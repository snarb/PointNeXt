import os, logging, csv, numpy as np, wandb
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.dataset import build_dataloader_from_cfg
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
# from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import furthest_point_sample, fps
from openpoints.utils.utils_3shape import homography_mae_robust#, homography_mae
from openpoints.utils.utils_3shape import get_mesh_and_plane, rotation_matrix_to_normal, get_plane
import trimesh
from scipy.spatial.transform import Rotation as R
import shutil

def get_features_by_keys(input_features_dim, data):
    if input_features_dim == 3:
        features = data['pos']
    elif input_features_dim == 4:
        features = torch.cat(
            (data['pos'], data['heights']), dim=-1)
        raise NotImplementedError("error")
    return features.transpose(1, 2).contiguous()


def write_to_csv(oa, macc, accs, best_epoch, cfg, write_header=True):
    accs_table = [f'{item:.2f}' for item in accs]
    header = ['method', 'OA', 'mAcc'] + \
        cfg.classes + ['best_epoch', 'log_path', 'wandb link']
    data = [cfg.exp_name, f'{oa:.3f}', f'{macc:.2f}'] + accs_table + [
        str(best_epoch), cfg.run_dir, wandb.run.get_url() if cfg.wandb.use_wandb else '-']
    with open(cfg.csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
        f.close()


def print_cls_results(oa, macc, accs, epoch, cfg):
    s = f'\nClasses\tAcc\n'
    for name, acc_tmp in zip(cfg.classes, accs):
        s += '{:10}: {:3.2f}%\n'.format(name, acc_tmp)
    s += f'E@{epoch}\tOA: {oa:3.2f}\tmAcc: {macc:3.2f}\n'
    logging.info(s)


def main(gpu, cfg, profile=False):
    if os.path.exists('visualisation'):
        shutil.rmtree('visualisation')

    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()
    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0 :
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        writer = SummaryWriter(log_dir=cfg.run_dir)
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    if not cfg.model.get('criterion_args', False):
        cfg.model.criterion_args = cfg.criterion_args
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    # criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    # build dataset
    val_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=cfg.distributed
                                           )
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
    test_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                            cfg.dataset,
                                            cfg.dataloader,
                                            datatransforms_cfg=cfg.datatransforms,
                                            split='test',
                                            distributed=cfg.distributed
                                            )

    num_points = val_loader.dataset.num_points if hasattr(
        val_loader.dataset, 'num_points') else None
    validate_fn = eval(cfg.get('val_fn', 'validate'))

    # optionally resume from a checkpoint
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler,
                              pretrained_path=cfg.pretrained_path)
            val_loss, val_rotation_mae, val_translation_dist, prev_losses = validate_fn(model, val_loader, cfg)
            print("val_loss: {}, val_rotation_mae: {} val_translation_dist: {}".format(val_loss, val_rotation_mae, val_translation_dist))
        else:
            if cfg.mode == 'test':
                # test mode
                epoch, best_val = load_checkpoint(
                    model, pretrained_path=cfg.pretrained_path)
                val_loss, val_rotation_mae, val_translation_dist, prev_losses = validate_fn(model, val_loader, cfg)
                print("val_loss: {}, val_rotation_mae: {} val_translation_dist: {}".format(val_loss, val_rotation_mae,
                                                                                           val_translation_dist))
                return True
            elif cfg.mode == 'val':
                # validation mode
                epoch, best_val = load_checkpoint(model, cfg.pretrained_path)
                val_loss, val_rotation_mae, val_translation_dist, prev_losses= validate_fn(model, val_loader, cfg)
                print("val_loss: {}, val_rotation_mae: {} val_translation_dist: {}".format(val_loss, val_rotation_mae,
                                                                                           val_translation_dist))
                return True
            elif cfg.mode == 'finetune':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model, cfg.pretrained_path)
            elif cfg.mode == 'finetune_encoder':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model.encoder, cfg.pretrained_path)
    else:
        logging.info('Training from scratch')
    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             )
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")

    # ===> start training
    val_macc, val_oa, val_accs, best_val, macc_when_best, best_epoch = 0., 0., [], 0., 0., 0
    model.zero_grad()
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):
            train_loader.dataset.epoch = epoch - 1
        train_loss, rotation_mae, translation_dist = train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg)

        is_best = False
        if epoch % cfg.val_freq == 0:
            val_loss, val_rotation_mae, val_translation_dist, prev_losses = validate_fn(model, val_loader, cfg, prev_losses)
            print("val_loss: {}, val_rotation_mae: {} val_translation_dist: {}".format(val_loss, val_rotation_mae, val_translation_dist))

            is_best = val_loss < best_val or best_val == 0.
            if is_best:
                best_val = val_loss
                best_epoch = epoch
                logging.info(f'Find a better ckpt @E{epoch}')

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train_loss {train_loss:.2f}, val_loss {val_loss:.2f}, best val loss {best_val:.2f}')
        if writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('val_translation_dist', translation_dist, epoch)
            writer.add_scalar('train_rotation_mae', rotation_mae, epoch)
            writer.add_scalar('val_translation_mae', val_translation_dist, epoch)
            writer.add_scalar('val_rotation_mae', val_rotation_mae, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('lr', lr, epoch)
            writer.add_scalar('val_oa', val_oa, epoch)
            writer.add_scalar('best_val', best_val, epoch)
            writer.add_scalar('epoch', epoch, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best
                            )
    # test the last epoch
    # val_loss, val_rotation_mae, val_translation_mae = validate(model, test_loader, cfg)
    # print_cls_results(test_oa, test_macc, test_accs, best_epoch, cfg)
    # if writer is not None:
    #     writer.add_scalar('test_oa', test_oa, epoch)
    #     writer.add_scalar('test_macc', test_macc, epoch)

    # test the best validataion model
    best_epoch, _ = load_checkpoint(model, pretrained_path=os.path.join(
        cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    test_loss, test_rotation_mae, test_translation_dist = validate(model, test_loader, cfg)
    if writer is not None:
        writer.add_scalar('test_loss', test_loss, best_epoch)
        writer.add_scalar('test_rotation_mae', test_rotation_mae, best_epoch)
        writer.add_scalar('test_translation_dist', test_translation_dist, best_epoch)

    print("test_loss: {}, test_rotation_mae: {} test_translation_dist: {} best_epoch: {}".format(test_loss, test_rotation_mae,
                                                                                   test_translation_dist, best_epoch))
    if writer is not None:
        writer.close()
    dist.destroy_process_group()

def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg):
    loss_meter = AverageMeter()
    rotation_mae_meter = AverageMeter()
    translation_dist_meter = AverageMeter()

    npoints = cfg.num_points

    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        for key in data.keys():
            if key not in ['mesh_path', 'plane_path']:
                data[key] = data[key].cuda(non_blocking=True)
        num_iter += 1
        points = data['x']
        target = data['y']
        """ bebug
        from openpoints.dataset import vis_points
        vis_points(data['pos'].cpu().numpy()[0])
        """
        num_curr_pts = points.shape[1]
        if num_curr_pts > npoints:  # point resampling strategy
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
            if  points.size(1) < point_all:
                point_all = points.size(1)
            fps_idx = furthest_point_sample(
                points[:, :, :3].contiguous(), point_all)
            fps_idx = fps_idx[:, np.random.choice(
                point_all, npoints, False)]
            points = torch.gather(
                points, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, points.shape[-1]))

        data['pos'] = points[:, :, :3].contiguous()
        data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
        predictions = model.forward(data)
        #logits, loss = model.get_loss(predictions, target) if not hasattr(model, 'module') else model.module.get_loss(predictions, target)
        loss = model.get_loss(predictions, target, data['normals'])
        loss.backward()

        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            model.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        #rotation_mae, translation_mae = homography_mae_robust(predictions, target)
        #rotation_mae, translation_mae = homography_mae_robust(torch.zeros_like(target), target)
        #rotation_mae, translation_mae = homography_mae(torch.zeros_like(target), target)
        rotation_mae, translation_dist = homography_mae_robust(predictions, target, data['normals'])




        # update confusion matrix
        rotation_mae_meter.update(rotation_mae.item())
        translation_dist_meter.update(translation_dist.item())

        loss_meter.update(loss.item())
        if idx % cfg.print_freq == 0:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} rot_mae {rotation_mae_meter.val:.1f} translation_dist {translation_dist_meter.val:.1f}")
    return loss_meter.avg, rotation_mae_meter.avg, translation_dist_meter.avg


@torch.no_grad()
def validate(model, val_loader, cfg, prev_losses = None):
    rotation_mae_meter = AverageMeter()
    translation_dist_meter = AverageMeter()
    loss_meter = AverageMeter()
    losses = []
    model.eval()  # set model to eval mode
    npoints = cfg.num_points
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    visualized_cnt = 0
    if prev_losses is not None:
        sorted_prev_losses = np.sort(prev_losses)
        bad_predictions_threshold = sorted_prev_losses[-cfg.vis_cnt]
        good_predictions_threshold = sorted_prev_losses[cfg.vis_cnt]
        half = cfg.vis_cnt // 2
        middle = len(sorted_prev_losses) // 2
        mid_preds_threshold_down = sorted_prev_losses[middle - half]
        mid_preds_threshold_up = sorted_prev_losses[middle + half]

    with torch.no_grad():
        for idx, data in pbar:
            for key in data.keys():
                if key not in ['mesh_path', 'plane_path']:
                    data[key] = data[key].cuda(non_blocking=True)
            target = data['y']
            points = data['x']
            points = points[:, :npoints]
            data['pos'] = points[:, :, :3].contiguous()
            data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
            predictions = model(data)
            loss, loss_full = model.get_val_loss(predictions, target, data['normals'])
            cur_loss = loss_full.cpu().numpy()
            losses.extend(cur_loss)
            rotation_mae, translation_dist = homography_mae_robust(predictions, target, data['normals'])
            rotation_mae_meter.update(rotation_mae.item())
            translation_dist_meter.update(translation_dist.item())

            for i in range(data['x'].shape[0]):
                if (prev_losses is None and visualized_cnt < cfg.vis_cnt_start):
                    visualise_predictions(data, i, predictions, 'random_predictions')
                    visualized_cnt += 1
                elif prev_losses is not None:
                    cur_loss_i = cur_loss[i]
                    if cur_loss_i < bad_predictions_threshold:
                        visualise_predictions(data, i, predictions, 'bad_predictions')
                    elif cur_loss_i > good_predictions_threshold:
                        visualise_predictions(data, i, predictions, 'good_predictions')
                    elif cur_loss_i > mid_preds_threshold_down and cur_loss_i < mid_preds_threshold_up:
                        visualise_predictions(data, i, predictions, 'avg_predictions')




            loss_meter.update(loss.item())

    return loss_meter.avg, rotation_mae_meter.avg, translation_dist_meter.avg, losses


def visualise_predictions(data, i, predictions, output_dir):
    mesh_path = data['mesh_path'][i]
    plane_path = data['plane_path'][i]

    # Load mesh
    mesh_o = trimesh.load_mesh(mesh_path)
    # Ground truth plane
    origin_gt, normals_gt = get_plane(plane_path)
    rot_gt = R.from_rotvec(np.cross([0, 0, 1], normals_gt))
    matrix_4x4_gt = np.eye(4)  # start with a 4x4 identity matrix
    matrix_4x4_gt[:3, :3] = rot_gt.as_matrix()  # fill the upper 3x3 part with rotation matrix
    matrix_4x4_gt[:3, 3] = origin_gt  # fill the last column with translation
    gt_plane = trimesh.creation.box(extents=[20, 20, 0.1], transform=matrix_4x4_gt)
    gt_plane.visual.face_colors = [0, 255, 0, 100]  # Green color for ground truth plane

    # Predicted plane
    normals_pred = rotation_matrix_to_normal(predictions[i, :3, :3]).cpu().numpy()
    origin_pred = predictions[i, :3, 3].cpu().numpy()
    rot_pred = R.from_rotvec(np.cross([0, 0, 1], normals_pred))
    matrix_4x4_pred = np.eye(4)  # start with a 4x4 identity matrix
    matrix_4x4_pred[:3, :3] = rot_pred.as_matrix()  # fill the upper 3x3 part with rotation matrix
    matrix_4x4_pred[:3, 3] = origin_pred  # fill the last column with translation
    pred_plane = trimesh.creation.box(extents=[20, 20, 0.1], transform=matrix_4x4_pred)
    pred_plane.visual.face_colors = [255, 0, 0, 100]  # Red color for predicted plane

    # Create a scene with both planes
    scene = trimesh.Scene([mesh_o, gt_plane, pred_plane])

    # Show the scene
    #scene.show()
    file_name = os.path.splitext(os.path.basename(mesh_path))[0]
    out_path = os.path.join('visualisation', output_dir, file_name, 'scene.gltf')
    os.makedirs(out_path, exist_ok=True)
    scene.export(out_path) # https://gltf-viewer.donmccurdy.com/
    f = 2
    # # Save the scene as PNG
    # with open(output_dir, 'wb') as f:
    #     f.write(scene.save_image())
