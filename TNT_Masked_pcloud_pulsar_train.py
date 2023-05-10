from __future__ import print_function, division
import sys
sys.path.append('core')
sys.path.append('datasets')

import argparse
import os
import cv2
import imageio
import time
import numpy as np
import json
import math
import glob
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import transforms as tvT
from torchvision.transforms import functional as tvF

from raft import RAFT
from llff import LLFFViewsynTrain
from tnt_masked import TNTMaskedViewsynTrain

from projector import Shader

from modules.extractor import BasicEncoder, SameResEncoder
from modules.unet import SmallUNet, UNet, TwoLayersCNN
from modules.msssim import ssim, msssim
import frame_utils
from morphology import Dilation2d, Erosion2d
from basic_utils import smoothnessloss, get_gaussian_kernel, compute_ssim
from geom_utils import check_depth_consistency, get_view_dir_world, get_view_dir_world_per_ray, crop_operation, PtsUnprojector, Lie, get_animation_poses
from summ_utils import Logger

import projective_ops as pops


from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from collections import OrderedDict
import subprocess

import lpips

from pytorch3d.structures import Pointclouds

from pytorch3d.renderer import (
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRasterizer
)

from pulsar.unified import PulsarPointsRenderer

# from pytorch3d.renderer import compositing
# from pytorch3d.renderer.points import rasterize_points
# from pytorch3d.ops.knn import knn_gather, knn_points
#
# from pytorch3d.renderer import (
#     FoVOrthographicCameras,
#     PerspectiveCameras,
#     PointsRasterizationSettings,
#     PointsRasterizer,
#     PointsRenderer,
#     PulsarPointsRenderer,
#     AlphaCompositor,
# )

from plyfile import PlyData, PlyElement

EPS = 1e-2


def fetch_optimizer(args, model):
    # todo: enable the dict
    """ Create the optimizer and learning rate scheduler """
    # special_args_dicts has format {'vert_feat': 1e-3, ...}
    # my_list = ['vert_feat', 'vert_opy']
    # my_list = ['vert_feat']

    # free_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))))
    # net_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))))
    special_args_dict = args.special_args_dict

    named_params = dict(model.named_parameters())
    default_params = [x[1] for x in named_params.items() if x[0] not in list(special_args_dict.keys())]

    # default_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in list(special_args_dict.keys()), named_params))))

    param_group_lr_list = [args.lr]
    special_params_list = [{'params': default_params, 'lr': args.lr}, ] # use the default lr (args.lr)

    for (name, lr) in special_args_dict.items():
        if name in named_params.keys():
            special_params_list.append({'params': named_params[name], 'lr': lr})
            param_group_lr_list.append(lr)
        else:
            print('warning: param key %s does not exist in model' % name)

    optimizer = optim.AdamW(special_params_list, lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    # print('free_params', free_params)
    # print('net_parame', net_params)

    # print('free_params', free_params)
    # print('net_params', net_params)

    # optimizer = optim.AdamW([{'params': net_params, 'lr': 1e-4}, {'params': free_params, 'lr': 1e-2}], lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    # optimizer = optim.AdamW([{'params': free_params, 'lr': 1e-2}], lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    # optimizer = optim.SGD([{'params': net_params, 'lr': args.lr}, {'params': free_params, 'lr': 1e-2}], lr=args.lr, weight_decay=args.wdecay)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    # print(net_params)

    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
    #     pct_start=args.pct_start, cycle_momentum=False, anneal_strategy='linear')
    # scheduler = None
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, param_group_lr_list, args.num_steps + 100,
        pct_start=args.pct_start, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def sequence_loss_rgb(rgb_est,
                        rgb_gt,
                        mask_gt=None,
                        loss_type='l1',
                        lpips_vgg=None,
                        vgg_loss=None,
                        weight=None,
                        gradual_weight=None,
                        gamma=0.9):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(rgb_est)
    flow_loss = 0.0

    ht, wd = rgb_gt.shape[-2:]

    if mask_gt is None:
        mask_gt = torch.ones_like(rgb_gt)
    else:
        mask_gt = mask_gt.expand_as(rgb_gt)



    # for i in range(n_predictions):
    #     rgb_est[i] = F.interpolate(rgb_est[i], [ht, wd], mode='bilinear', align_corners=True)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        if loss_type == "l1":
            i_loss = (rgb_est[i]*mask_gt - rgb_gt*mask_gt).abs()
        elif loss_type == "l2" or loss_type == "linear_l2":
            i_loss = (rgb_est[i]*mask_gt - rgb_gt*mask_gt)**2
        elif loss_type == "msssim":
            i_loss = 1. - msssim(rgb_est[i]*mask_gt, rgb_gt*mask_gt, val_range=2, normalize="relu")
            # i_loss = ssim(rgb_est[i] * mask_gt, rgb_gt * mask_gt, val_range=2)
        elif loss_type == "mix_l1_msssim":
            msssim_loss = 1. - msssim(rgb_est[i]*mask_gt, rgb_gt*mask_gt, val_range=2, normalize="relu")
            l1_loss = (rgb_est[i]*mask_gt - rgb_gt*mask_gt).abs()
            # alpha value from this paper https://arxiv.org/pdf/1511.08861
            alpha = 0.84
            i_loss = alpha * msssim_loss + (1.-alpha) * l1_loss
        else:
            raise NotImplementedError

        if not weight is None:
            i_loss *= weight

        # flow_loss += i_weight * i_loss.mean()
        flow_loss += i_weight * i_loss.sum() / mask_gt.sum()

    rgb_gt_scaled = (rgb_gt + 1.0) / 2.0 # range [0,1]
    rgb_est_scaled = (rgb_est[-1].detach() + 1.0) / 2.0

    l1 = (rgb_est_scaled * mask_gt - rgb_gt_scaled * mask_gt).abs().sum() / torch.sum(mask_gt) # proper normalize, consider the size of the mask
    l2 = ((rgb_est_scaled * mask_gt - rgb_gt_scaled * mask_gt)**2).sum() / torch.sum(mask_gt)
    psnr = -10. * torch.log(l2) / np.log(10.0)

    ssim = 0.0

    B = rgb_gt_scaled.shape[0]
    for b in range(B):
        g = (rgb_gt_scaled * mask_gt)[b].permute(1,2,0).cpu().numpy()
        p = (rgb_est_scaled * mask_gt)[b].permute(1,2,0).cpu().numpy()
        m = mask_gt[b].permute(1,2,0).cpu().numpy()

        ssim_custom = compute_ssim(g*255.0, p*255.0)
        ssim_custom = np.mean(ssim_custom, axis=2)
        ssim_custom = np.pad(ssim_custom, ((5, 5), (5, 5)), mode='mean')
        ssim_custom = ssim_custom[..., None]
        ssim += np.sum(ssim_custom * m) / np.sum(m)

        # sanity check: same when no mask
        # print(np.sum(ssim_custom * m) / np.sum(m))
        # print(np.mean(compute_ssim(g*255.0, p*255.0)))

        # ssim += structural_similarity(g, p, data_range=1.0, multichannel=True)

    ssim = ssim / float(B)

    assert (not torch.isnan(rgb_est[-1]).any())
    assert (not torch.isinf(rgb_est[-1]).any())

    metrics = {
        'l1': l1.item(),
        'l2': l2.item(),
        'psnr': psnr.item(),
        'ssim': ssim,
    }

    if lpips_vgg is not None:
        with torch.no_grad():
            # input should have range [-1,1], which we already have; need bgr2rgb
            # have sanity check that psnr, ssim and lpips is exactly the same as the tf version https://github.com/bmild/nerf/issues/66
            lpips_val = lpips_vgg(rgb_gt[:, [2,1,0]], rgb_est[-1][:, [2,1,0]])
            lpips_val = lpips_val.mean().cpu().item()
            metrics['lpips'] = lpips_val

    if vgg_loss is not None:
        with torch.no_grad():
            vgg_dist = vgg_loss(rgb_gt_scaled[:, [2, 1, 0]], rgb_est_scaled[:, [2, 1, 0]])  # input should have range [-1,1]
            vgg_dist = vgg_dist.mean().cpu().item()
            metrics['vgg'] = vgg_dist

    # print(metrics)

    return flow_loss, metrics

def extract_error_map(model, tau_E, ref_images, ref_masks, ref_depths, ref_poses, ref_intrinsics, data_loader, dataset_args, logger):
    model.eval()

    factor = dataset_args['factor']
    render_scale = dataset_args['render_scale']

    e_op_2 = Erosion2d(1, 1, 2, soft_max=False).cuda()
    d_op_2 = Dilation2d(1, 1, 2, soft_max=False).cuda()

    with torch.no_grad():
        l1_err_maps = []
        all_imgs = []
        all_preds = []
        positive_areas = []

        for i_batch, data_blob in enumerate(data_loader):
            # faster parameter tuning
            # if i_batch % 3 != 0:
            #     continue


            # if dataset_args["return_mask"]:
            #     images, depths, masks, poses, intrinsics = data_blob
            # else:
            #     images, depths, poses, intrinsics = data_blob
            #     masks = torch.ones_like(images[:, :, 0]) # color mask
            images, depths, poses, intrinsics = data_blob
            masks = torch.ones_like(images[:, :, 0])  # color mask

            images = images.cuda()
            poses = poses.cuda()
            intrinsics = intrinsics.cuda()
            masks = masks.cuda()
            masks = masks.unsqueeze(2)

            rgb_gt = images[:, 0] * 2.0 / 255.0 - 1.0  # range [-1, 1], 1 x 3 x H x W
            rgb_gt = F.interpolate(rgb_gt, [dataset_args["crop_size"][0] // (factor // render_scale),
                                            dataset_args["crop_size"][1] // (factor // render_scale)], mode='bilinear',
                                   align_corners=True)
            mask_gt = F.interpolate(masks[:, 0], [dataset_args["crop_size"][0] // (factor // render_scale),
                                                  dataset_args["crop_size"][1] // (factor // render_scale)], mode='nearest')

            intrinsics_gt = intrinsics[:, 0]
            intrinsics_gt[:, 0] /= (images.shape[3] / (dataset_args["crop_size"][0] // factor))  # rescale according to the ratio between dataset images and render images
            intrinsics_gt[:, 1] /= (images.shape[4] / (dataset_args["crop_size"][1] // factor))

            # rgb_est = model(ref_images, poses[:, 0], intrinsics_gt, is_eval=True) # 1 x 3 x H x W
            rgb_est = model.evaluate(ref_images, poses[:, 0], intrinsics_gt, num_random_samples=5)

            l1_err_map = torch.mean(torch.abs(rgb_gt - rgb_est), dim=1, keepdim=True) # 1 x 1 x H x W
            l1_err_maps.append(l1_err_map)

            mean_err = torch.mean(l1_err_map)
            std_err = torch.std(l1_err_map)
            positive_area = (l1_err_map > mean_err + tau_E * mean_err).float()
            positive_area = d_op_2(e_op_2(positive_area))  # opening

            positive_areas.append(positive_area)

            all_imgs.append(rgb_gt)
            all_preds.append(rgb_est)

        all_imgs = torch.cat(all_imgs, dim=0) # N x 3 x H x W
        all_preds = torch.cat(all_preds, dim=0) # N x 3 x H x W

        gaussian_kernel = get_gaussian_kernel(kernel_size=11, sigma=1.5).cuda()
        all_imgs = gaussian_kernel(all_imgs)
        all_preds = gaussian_kernel(all_preds)

        l1_err_maps = torch.cat(l1_err_maps, dim=0) # N x 1 x H x W
        max_err = torch.max(l1_err_maps)
        l1_err_maps_normalized = l1_err_maps / max_err # N x 1 x H x W, range[0,1]
        global_step = logger.total_steps # backup
        logger.set_global_step(0)


        # post-process the error map
        # mean_err = torch.mean(l1_err_maps)
        # std_err = torch.std(l1_err_maps)

        # positive_areas = (l1_err_maps > mean_err + 3.0 * std_err).float() # may need to adjust
        positive_areas = torch.cat(positive_areas, dim=0)

        # if bkg_col is not None:
        #     # only add points to the non-bkg area
        #     bkg_col = torch.tensor(bkg_col).cuda()
        #     fg_area = (torch.mean(all_imgs-bkg_col.reshape(1, 3, 1, 1), dim=1, keepdim=True) > 0.05).float()
        #     positive_areas = positive_areas * fg_area # N x 1 x H x W

        print('total number of positive pixels:', torch.sum(positive_areas))


        for i in range(l1_err_maps.shape[0]):
            logger.set_global_step(i)
            # cat the things together
            gt_to_sum = all_imgs[i:i+1] # 1 x 3 x H x W
            pred_to_sum = all_preds[i:i + 1]  # 1 x 3 x H x W
            err_to_sum = (l1_err_maps_normalized[i:i+1].repeat(1, 3, 1, 1)) * 2.0 - 1.0 # 1 x 3 x H x W, ranging [-1,1]
            pos_to_sum = (positive_areas[i:i+1].repeat(1, 3, 1, 1)) * 2.0 - 1.0  # 1 x 3 x H x W, ranging [-1,1]

            final_sum = torch.cat([gt_to_sum, pred_to_sum, err_to_sum, pos_to_sum], dim=2) # cat in vertical direction
            logger.summ_rgb('error/l1_error_map', final_sum, force_save=True)


        logger.set_global_step(global_step)

    model.train()

    return positive_areas # N x 1 x H x W


def add_points(positive_area, ref_images, ref_depths, ref_depth_masks, ref_poses, ref_intrinsics, min_depth=800, max_depth=1400, pts_dropout_rate=0.9, shallowest_few=-1, lindisp=False):
    # positive_area is N x 1 x H x W
    # ref_images is N x 3 x H x W
    # ref_depths is N x 1 x H x W
    # ref_depth_masks is N x 1 x H x W
    depth_levels = 100

    # more conservative
    # tau_f = 0.95
    #
    # lambda_geo = 1.5e1
    # lambda_photo = 7.5e1

    t_vals = torch.linspace(0., 1., steps=depth_levels).cuda()
    if not lindisp:
        depth_array = min_depth * (1. - t_vals) + max_depth * (t_vals)
    else:
        depth_array = 1. / (1. / min_depth * (1. - t_vals) + 1. / max_depth * (t_vals))

    num_views, _, H, W = positive_area.shape
    depth_array = depth_array.reshape(-1, 1, 1, 1).repeat(1, 1, H, W) # d x 1 x H x W

    unprojector = PtsUnprojector()

    all_points_to_add = []
    all_buvs = []

    do_random_dropout = pts_dropout_rate > 0.0

    # if bkg_col is not None:
    #     bkg_col = torch.tensor(bkg_col).cuda()
    #     bkg_area = (torch.mean(ref_images - bkg_col.reshape(1, 3, 1, 1), dim=1, keepdim=True) <= 0.05) # N x 1 x H x W, boolean
    #     ref_depths[bkg_area] = 1e6 # don't add points in those area

    for view_id in range(num_views):
        # debug
        # 29, 24
        # if view_id != 23:
        #     continue

        positive_mask = positive_area[view_id:view_id+1].repeat(depth_levels, 1, 1, 1) # d x 1 x H x W
        pose = ref_poses[view_id:view_id+1].repeat(depth_levels, 1, 1)
        intrinsics = ref_intrinsics[view_id:view_id + 1].repeat(depth_levels, 1, 1)

        vert_pos, buv = unprojector(depth_array, pose, intrinsics, positive_mask, return_coord=True)
        vert_pos = vert_pos.permute(1, 0) # 3xN, where N is the cartesian product of all positive rays and all possible depths
        buv[:, 0] = view_id
        buv = buv.permute(1, 0) # 3xN


        xyz_world = torch.cat((vert_pos, torch.ones_like(vert_pos[0:1])), dim=0).unsqueeze(0).repeat(num_views, 1, 1)  # num_views x 4 x N, turned into homogeneous coord

        # target_pose is cam_T_world. turn into all other views
        xyz_target = ref_poses.bmm(xyz_world)
        xyz_target = xyz_target[:, 0:3]  # num_views x 3 x N, discard homogeneous dimension

        xy_proj = ref_intrinsics.bmm(xyz_target)  # num_views x 3 x N

        eps_mask = (xy_proj[:, 2:3, :].abs() < EPS).detach()

        # Remove invalid zs that cause nans
        zs = xy_proj[:, 2:3, :]
        zs[eps_mask] = EPS

        sampler = torch.cat((xy_proj[:, 0:2, :] / zs, xy_proj[:, 2:3, :]),
                            1)  # u, v, has range [0,W], [0,H] respectively
        sampler[eps_mask.repeat(1, 3, 1)] = -1e6
        # sampler is num_views x 3 x N
        # results should have shape num_views x N
        sampler = torch.round(sampler).to(torch.long)
        sampler_xs = sampler[:, 0] # num_views x N
        sampler_ys = sampler[:, 1] # num_views x N
        sampler_zs = sampler[:, 2] # num_views x N
        sampler_xs_bounded = torch.clamp(sampler_xs, 0, W-1)
        sampler_ys_bounded = torch.clamp(sampler_ys, 0, H-1)
        # do nn sample
        samples = []
        for i in range(num_views):
            sample = torch.zeros_like(zs[0]) # 1 x N

            # if the corresponding area's err is large enough, the fitness is 1.
            sample += positive_area[i, :, sampler_ys_bounded[i], sampler_xs_bounded[i]] # 1 x N

            # if out of bound, we say that view agrees
            sample[:, ((sampler_xs[i] < 0) | (sampler_xs[i] > W-1) | (sampler_ys[i] < 0) | (sampler_ys[i] > H-1))] += 1.0  # 1 x N

            # or if the sample's depth is DEEPER than the predicted depth, we also say "don't know".
            sample_depth = sampler_zs[i:i+1] # 1 x N
            pred_depth = ref_depths[i, :, sampler_ys_bounded[i], sampler_xs_bounded[i]] # 1 x N
            pred_depth_mask = ref_depth_masks[i, :, sampler_ys_bounded[i], sampler_xs_bounded[i]] # 1 x N
            pred_depth[pred_depth_mask < 0.5] = 1e6 # set to infinity, since predicetd pts in those areas are pre-filtered out

            sample[sample_depth > pred_depth] += 1.0
            # sample[sample_depth <= pred_depth] += (torch.exp(-lambda_geo * (1. - sample_depth / pred_depth)))[sample_depth <= pred_depth]
            #
            # # check the photometric consistency
            # sample_col = ref_images[view_id, :, buv[1], buv[2]]  # 3 x N
            # pred_col = ref_images[i, :, sampler_ys_bounded[i], sampler_xs_bounded[i]]  # 3 x N
            # col_diff = torch.mean(torch.abs(sample_col - pred_col) / 2.0, dim=0, keepdim=True)  # 1 x N, normalize the scale to [0,1]
            #
            # sample += torch.exp(-lambda_photo * col_diff)

            # return a large negative value for point not passing the check
            sample[sample > 0.5] = 1.0

            samples.append(sample)

        samples = torch.cat(samples, dim=0) # num_views x N

        # try the simplest thing first
        pos_samples = torch.sum(samples, dim=0) == num_views # N, binary indicator

        # # add points only to the fg area
        # if bkg_col is not None:
        #     bkg_col = torch.tensor(bkg_col).cuda()
        #     fg_area = (torch.mean(ref_images[view_id] - bkg_col.reshape(3, 1, 1), dim=0, keepdim=True) > 0.05).float() # 1 x H x W
        #     fg_samples = fg_area[0, buv[1], buv[2]] # N
        #     pos_samples = torch.logical_and(pos_samples, fg_samples)


        # take the shallowest few if too many points past the test, to save effort
        if shallowest_few > 0:
            pos_samples = pos_samples.reshape(depth_levels, -1) # 100 x N
            cum_pos_samples = torch.cumsum(pos_samples, dim=0, dtype=torch.long) # accumulate the number of positive samples along the depth dim
            shallowest_samples = cum_pos_samples <= shallowest_few
            pos_samples = torch.logical_and(pos_samples.reshape(-1), shallowest_samples.reshape(-1))

        points_to_add = vert_pos[:, pos_samples] # 3 x N_valid
        buv = buv[:, pos_samples]

        if do_random_dropout:
            num_pts_to_keep = round(points_to_add.shape[1] * (1.0 - pts_dropout_rate))
            pts_id_to_keep = torch.tensor(np.random.choice(np.arange(points_to_add.shape[1]), size=num_pts_to_keep, replace=False))
            points_to_add = points_to_add[:, pts_id_to_keep]
            buv = buv[:, pts_id_to_keep]

        all_points_to_add.append(points_to_add)
        all_buvs.append(buv)


    all_points_to_add = torch.cat(all_points_to_add, dim=1) # 3 x N
    all_buvs = torch.cat(all_buvs, dim=1)  # 3 x N

    return all_points_to_add, all_buvs.to(torch.long)

def make_animation(model, val_dataset, ref_intrinsics, logger, rasterize_rounds=1):
    model.eval()
    metrics = {}

    # make a video with varying cam pose
    render_poses = val_dataset.get_render_poses()
    # render_viewpose = val_dataset.get_render_poses(radius=20) # larger movement

    N_views = render_poses.shape[0]

    # pre-select subset to reduce flickering
    num_pts_to_keep = round(model.vert_pos.shape[1] * (1.0 - model.pts_dropout_rate))
    pts_id_to_keep = torch.multinomial(torch.ones_like(model.vert_pos[0]), num_pts_to_keep, replacement=False)

    all_frames = []
    with torch.no_grad():
        for i_batch in range(N_views):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            target_pose = render_poses[i_batch:i_batch+1] # 1 x 4 x 4

            start.record()
            rgb_est = model.evaluate(None, target_pose, ref_intrinsics[0:1], num_random_samples=rasterize_rounds, pts_to_use_list=None, fix_seed=True)  # 1 x 3 x H x W
            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()

            print('total render time for one image:', start.elapsed_time(end))

            all_frames.append(rgb_est)


    all_frames = torch.cat(all_frames)
    logger.summ_rgbs('animation/motion', all_frames, fps=20, force_save=True)

    all_frames = (all_frames + 1.0) / 2.0
    all_frames = torch.clamp(all_frames, 0.0, 1.0)
    all_frames = all_frames[:, [2, 1, 0]]  # bgr2rgb, N x 3 x H x W, range [0,1]
    all_frames = all_frames.permute(0, 2, 3, 1).cpu().numpy()  # N x H x W x 3, range [0,1]

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    imageio.mimwrite(os.path.join('./saved_videos/%s.wmv' % args.name), to8b(all_frames), format='FFMPEG', fps=20, quality=10)

    # # make a video with fixed cam pose and varying lighting dir
    # all_frames = []
    # with torch.no_grad():
    #     for i_batch in range(N_views):
    #         start = torch.cuda.Event(enable_timing=True)
    #         end = torch.cuda.Event(enable_timing=True)
    #
    #         target_viewpose = render_viewpose[i_batch:i_batch+1] # 1 x 4 x 4
    #         # target_pose = render_poses[int(3*N_views/4):int(3*N_views/4)+1] # 1 x 4 x 4
    #         # target_pose = render_poses[0:1]  # 1 x 4 x 4
    #         target_pose = render_poses[int(N_views/4):int(N_views/4)+1]  # 1 x 4 x 4
    #
    #         start.record()
    #         rgb_est = model.evaluate(None, target_pose, ref_intrinsics[0:1], num_random_samples=rasterize_rounds, pts_to_use_list=pts_id_to_keep, target_viewpose=target_viewpose)  # 1 x 3 x H x W
    #         end.record()
    #
    #         # Waits for everything to finish running
    #         torch.cuda.synchronize()
    #
    #         print('total render time for one image:', start.elapsed_time(end))
    #
    #         all_frames.append(rgb_est)
    #
    #
    # all_frames = torch.cat(all_frames)
    # logger.summ_rgbs('animation/viewdir', all_frames, fps=20, force_save=True)
    #
    # all_frames = (all_frames + 1.0) / 2.0
    # all_frames = torch.clamp(all_frames, 0.0, 1.0)
    # all_frames = all_frames[:, [2, 1, 0]]  # bgr2rgb, N x 3 x H x W, range [0,1]
    # all_frames = all_frames.permute(0, 2, 3, 1).cpu().numpy()  # N x H x W x 3, range [0,1]
    #
    # to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    # imageio.mimwrite(os.path.join('./saved_videos/view_%s.mp4' % args.name), to8b(all_frames), format='FFMPEG', fps=20, quality=10)

    model.train()

    return all_frames


def validate(model, ref_images, val_loader, valset_args, logger, pts_to_use_list=None, val_cam_noise=0.0, rasterize_rounds=2):
    model.eval()
    metrics = {}

    # lpips_vgg = lpips.LPIPS(net='vgg').cuda()
    lpips_vgg = lpips.LPIPS(net='alex', version='0.1').cuda() # follow nsvf
    # vgg_loss = VGGLoss().cuda()
    # lpips_vgg = None

    total_render_time = 0.0

    if val_cam_noise > 0.0:
        # generate the noise with a fixed random seed, so that we can compare fairly with nerf
        torch.manual_seed(0)
        lie = Lie()
        se3_noise = torch.randn(len(val_loader), 6, device=torch.device('cuda')) * val_cam_noise
        SE3_noise = lie.se3_to_SE3(se3_noise)  # 1 x 3 x 4
        SE3_noise = torch.cat([SE3_noise, torch.tensor([0.0, 0.0, 0.0, 1.0], device=torch.device('cuda')).reshape(1, 1, 4).repeat(len(val_loader), 1, 1)], dim=1)

    with torch.no_grad():
        for i_batch, data_blob in enumerate(val_loader):
            # if datasetname == "DTU":
            #     if valset_args["return_mask"]:
            #         images, depths, masks, poses, intrinsics = data_blob
            #     else:
            #         images, depths, poses, intrinsics = data_blob
            #         masks = torch.ones_like(images[:, :, 0])
            #
            # elif datasetname == "Blended":
            #     images, depths, poses, intrinsics, scene_id, indices, scale, SD, ED, ND = data_blob


            images, poses, intrinsics = data_blob
            masks = torch.ones_like(images[:, :, 0]) # color mask

            factor = valset_args['factor']
            render_scale = valset_args['render_scale']
            loss_type = valset_args['loss_type']

            images = images.cuda()
            poses = poses.cuda()
            intrinsics = intrinsics.cuda()
            masks = masks.cuda()
            # depths = depths.cuda()
            # depth_low_res = F.interpolate(depths, [valset_args["crop_size"][0] // factor, valset_args["crop_size"][1] // factor], mode='nearest')
            #
            # depths = depths.unsqueeze(2)
            # depth_low_res = depth_low_res.unsqueeze(2)
            masks = masks.unsqueeze(2)

            rgb_gt = images[:, 0] * 2.0 / 255.0 - 1.0  # range [-1, 1]
            rgb_gt = F.interpolate(rgb_gt, [valset_args["crop_size"][0] // (factor // render_scale), valset_args["crop_size"][1] // (factor // render_scale)], mode='bilinear',
                                   align_corners=True)
            mask_gt = F.interpolate(masks[:, 0], [valset_args["crop_size"][0] // (factor // render_scale), valset_args["crop_size"][1] // (factor // render_scale)], mode='nearest')

            intrinsics_gt = intrinsics[:, 0]
            intrinsics_gt[:, 0] /= (images.shape[3] / (valset_args["crop_size"][0] // factor))  # rescale according to the ratio between dataset images and render images
            intrinsics_gt[:, 1] /= (images.shape[4] / (valset_args["crop_size"][1] // factor))

            # gts, intrinsics_gt = crop_operation(torch.cat([rgb_gt, mask_gt], dim=1), intrinsics_gt, valset_args['resize_h'] // factor, valset_args['resize_w'] // factor, mod='center')
            # rgb_gt, mask_gt = torch.split(gts, [3, 1], dim=1)

            # rgb_est = model(images, poses, intrinsics, depth_low_res)
            # rgb_est = [model(ref_images, poses[:, 0], intrinsics_gt, is_eval=True), ]  # 1 x 3 x H x W
            # rgb_est = [model.evaluate(ref_images, poses[:, 0], intrinsics_gt), ]  # 1 x 3 x H x W

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            target_pose = poses[:, 0] # B x 4 x 4
            # for the psnr analysis only:
            # add random rotation and translation noise to the pose
            if val_cam_noise > 0.0:
                target_pose = target_pose.bmm(SE3_noise[i_batch:i_batch+1])

            start.record()
            rgb_est = [model.evaluate(ref_images, target_pose, intrinsics_gt, num_random_samples=rasterize_rounds, pts_to_use_list=pts_to_use_list), ]  # 1 x 3 x H x W
            # rgb_est = [model.evaluate(ref_images, poses[:, 0], intrinsics_gt, num_random_samples=1, pts_to_use_list=pts_to_use_list), ]  # 1 x 3 x H x W
            end.record()

            # # get the original TNT size
            # rgb_est[0] = F.interpolate(rgb_est[0], [2*valset_args["crop_size"][0] // (factor // render_scale), 2*valset_args["crop_size"][1] // (factor // render_scale)], mode='bilinear', align_corners=True)
            # rgb_gt = F.interpolate(rgb_gt, [2 * valset_args["crop_size"][0] // (factor // render_scale), 2 * valset_args["crop_size"][1] // (factor // render_scale)], mode='bilinear', align_corners=True)
            # mask_gt = F.interpolate(mask_gt, [2 * valset_args["crop_size"][0] // (factor // render_scale), 2 * valset_args["crop_size"][1] // (factor // render_scale)], mode='nearest')

            # Waits for everything to finish running
            torch.cuda.synchronize()

            print('total render time for one image:', start.elapsed_time(end))\

            if i_batch != 0:
                total_render_time += start.elapsed_time(end)

            # disp_loss, disp_metrics = sequence_loss(disp_est, disp_gt,
            #                                     depthloss_threshold=args.depthloss_threshold,
            #                                     loss_type=loss_type,
            #                                     weight=weight,
            #                                     gradual_weight=gradual_weight)
            # loss += disp_loss
            # metrics.update(disp_metrics)

            _, rgb_metrics = sequence_loss_rgb(rgb_est, rgb_gt, mask_gt, lpips_vgg=lpips_vgg,
                                                      loss_type=loss_type,
                                                      weight=None,
                                                      gradual_weight=None)

            if len(metrics) == 0: # init
                metrics.update(rgb_metrics)
            else: # update
                for (k, v) in metrics.items():
                    metrics[k] += rgb_metrics[k]

            print('finished rendering %d/%d' % (i_batch+1, len(val_loader)))
            print(rgb_metrics)

            logger.summ_rgb('eval/rgb_gt/%d' % i_batch, rgb_gt, mask_gt, force_save=True)
            logger.summ_rgb('eval/rgb_est/%d' % i_batch, rgb_est[-1], mask_gt, force_save=True)
            logger.summ_diff('eval/l1_diff/%d' % i_batch, rgb_gt, rgb_est[-1], force_save=True)

    # average
    for (k, v) in metrics.items():
        metrics[k] /= len(val_loader)

    # compute the "avg. metric" from mipnerf
    avg = (10.**(-metrics['psnr'] / 10.) * metrics['lpips'] * np.sqrt(1-metrics['ssim'])) ** (1./3.)
    metrics['avg'] = avg

    logger.write_dict(metrics, 'eval')

    print('finish eval on %d samples' % len(val_loader))
    print(metrics)

    print('average render time: %.1f' % (total_render_time / (len(val_loader) - 1)))

    if hasattr(model, 'vert_opy'):
        logger.summ_hist('vert_opy', torch.sigmoid(model.vert_opy), force_save=True)

    model.train()

    return metrics


class PulsarSceneModel(nn.Module):
    def __init__(self,
                 vert_pos,
                 dim_pointfeat=256,
                 pointfeat_init=None,
                 radius=7.5e-4,
                 render_size=(300, 400),
                 world_scale=400.,
                 render_scale=1,
                 bkg_col=(0,0,0),
                 gamma=1.0e-3,
                 max_n_points=5000000,
                 batch_size=1,
                 free_xyz=False,
                 free_opy=False,
                 free_rad=False,
                 do_2d_shading=False,
                 shader_arch='simple_unet',
                 pts_dropout_rate=0.0,
                 do_xyz_pos_encode=False,
                 basis_type='mlp',
                 shader_output_channel=128,
                 knn_3d_smoothing=0,
                 shader_norm='none',
                 ):
        super(PulsarSceneModel, self).__init__()
        # images: N x 3 x H x W
        # depth_low_res: N x 1 x h x w
        # masks_low_res: N x 1 x h x w

        self.free_opy = free_opy
        self.free_xyz = free_xyz
        self.free_rad = free_rad


        self.unprojector = PtsUnprojector()

        # vert_pos = self.unprojector(ref_depth_low_res, ref_poses, ref_intrinsics, mask=ref_masks_low_res)  # N_pts x 3
        # vert_pos = vert_pos.permute(1, 0)  # 3 x N

        # assert (not self.free_xyz)
        if args.free_xyz:
            self.register_parameter("vert_pos", nn.Parameter(vert_pos, requires_grad=True))
        else:
            self.register_buffer('vert_pos', vert_pos)  # 3 x N


        self.n_points = vert_pos.shape[1]

        self.knn_3d_smoothing = knn_3d_smoothing
        if self.knn_3d_smoothing > 0:
            print('computing knn...')
            idx = np.load('./saved_pointclouds/modified_pointclouds_shallow_fern_filter_2_knn.npy') # N x 20
            idx = torch.tensor(idx).long()
            idx = idx[:, :self.knn_3d_smoothing] # only take the first K
            # print('computing knn...')
            # x_nn = knn_points((vert_pos.permute(1,0).unsqueeze(0))[:, :10000], vert_pos.permute(1,0).unsqueeze(0), K=self.knn_3d_smoothing)
            # idx = x_nn.idx
            # # idx has shape 1 x N x K
            # idx = idx[0] # N x K;


            # assert (False)

            self.register_buffer('vert_knn_idx', idx) # N x K

        # points_init_loc has shape N x 3
        device = torch.device("cuda")

        self.do_xyz_pos_encode = do_xyz_pos_encode

        if pointfeat_init is not None:
            assert basis_type == 'mlp'
            assert pointfeat_init.shape[0] == self.n_points
            assert pointfeat_init.shape[1] == dim_pointfeat
            self.register_parameter("vert_feat", nn.Parameter(pointfeat_init, requires_grad=True))

        else:
            if do_xyz_pos_encode:
                vert_pos_normalized = vert_pos / world_scale  # value around 1, 3xN
                xyz_embed_fn, xyz_input_ch = get_embedder(multires=10)
                embedded_vert_pos = xyz_embed_fn(vert_pos_normalized.permute(1,0)).permute(1,0) # 60 x N
                self.register_buffer("embedded_vert_pos", embedded_vert_pos)

                if basis_type == 'mlp':
                    self.register_parameter("vert_feat", nn.Parameter(torch.randn(self.n_points, dim_pointfeat-xyz_input_ch), requires_grad=True))
                elif basis_type=='SH':
                    self.register_parameter("vert_feat", nn.Parameter(torch.zeros(self.n_points, dim_pointfeat - xyz_input_ch), requires_grad=True))
                elif basis_type=='none':
                    self.register_parameter("vert_feat", nn.Parameter(torch.zeros(self.n_points, dim_pointfeat- xyz_input_ch), requires_grad=True))
                else:
                    raise NotImplementedError
            else:
                if basis_type == 'mlp':
                    self.register_parameter("vert_feat", nn.Parameter(torch.randn(self.n_points, dim_pointfeat), requires_grad=True))
                elif basis_type=='SH':
                    self.register_parameter("vert_feat", nn.Parameter(torch.zeros(self.n_points, dim_pointfeat), requires_grad=True))
                elif basis_type=='none':
                    self.register_parameter("vert_feat", nn.Parameter(torch.zeros(self.n_points, dim_pointfeat), requires_grad=True))
                else:
                    raise NotImplementedError


        z_dir = torch.tensor([0, 0, 1, 0]).reshape(1, 4).float()  # the last element is 0, because we only care orientation
        self.register_buffer("z_dir", z_dir)

        if do_2d_shading:
            print('using shader arch:', shader_arch)
            self.shader_output_channel = shader_output_channel # 128
            assert shader_arch == 'simple_unet' # other render_scale option not supported yet

            if shader_arch == 'simple_unet':
                self.shader_2d = SmallUNet(n_channels=self.shader_output_channel, n_classes=3, bilinear=False, norm=shader_norm, render_scale=render_scale)
            elif shader_arch == 'full_unet':
                self.shader_2d = UNet(n_channels=self.shader_output_channel, n_classes=3, bilinear=False, norm=shader_norm)
            elif shader_arch == 'simple':
                self.shader_2d = TwoLayersCNN(n_channels=self.shader_output_channel, n_classes=3, norm=shader_norm)
            else:
                raise NotImplementedError

        else:
            self.shader_output_channel = 3 # override

        output_opacity = not free_opy # if not free, ouput opacity from the network

        # self.shader = Shader(feat_dim=dim_pointfeat, rgb_channel=self.shader_output_channel, output_opacity=self.free_opy, opacity_channel=1)
        self.shader = Shader(feat_dim=dim_pointfeat, rgb_channel=self.shader_output_channel, output_opacity=output_opacity, opacity_channel=1, basis_type=basis_type)

        if free_rad:
            raise NotImplementedError
        else:
            pass

        if free_opy:
            self.register_parameter("vert_opy", nn.Parameter(torch.ones(self.n_points), requires_grad=True))
        # else:
        #     self.register_buffer("vert_opy", torch.ones(self.n_points, dtype=torch.float32))  # don't update this


        # cameras = FoVOrthographicCameras(R=(torch.eye(3, dtype=torch.float32, device=device)[None, ...]).repeat(batch_size, 1, 1),
        #                                  T=torch.zeros((batch_size, 3), dtype=torch.float32, device=device),
        #                                  znear=[1.0]*batch_size,
        #                                  zfar=[1e5]*batch_size,
        #                                  device=device,
        #                                  )
        cameras = FoVOrthographicCameras(R=(torch.eye(3, dtype=torch.float32, device=device)[None, ...]),
                                         T=torch.zeros((1, 3), dtype=torch.float32, device=device),
                                         znear=[1.0],
                                         zfar=[1e5],
                                         device=device,
                                         )


        # raster_settings = PointsRasterizationSettings(
        #     image_size=render_size,
        #     radius=None,
        #     max_points_per_bin=50000
        #     # max_points_per_bin=5
        # )
        raster_settings = PointsRasterizationSettings(
                image_size=render_size,
                radius=None,
                # max_points_per_bin=5
                max_points_per_bin=50000
            )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        self.renderer = PulsarPointsRenderer(rasterizer=rasterizer, max_num_spheres=vert_pos.shape[1], n_channels=self.shader_output_channel, n_track=100).cuda()
        # self.renderer = PulsarPointsRenderer(rasterizer=rasterizer, max_num_spheres=vert_pos.shape[1], n_channels=self.shader_output_channel).cuda()

        if self.shader_output_channel==3:
            self.register_buffer('bkg_col', torch.tensor(bkg_col, dtype=torch.float32, device=device))
            # self.bkg_col = torch.tensor(bkg_col, dtype=torch.float32, device=device)
        else:
            # self.register_buffer('bkg_col', torch.randn(self.shader_output_channel, dtype=torch.float32, device=device))
            # make it learnable
            self.register_parameter('bkg_col', nn.Parameter(torch.randn(self.shader_output_channel, dtype=torch.float32, device=device), requires_grad=True))
            # self.bkg_col = torch.randn(self.shader_output_channel, dtype=torch.float32, device=device)

        self.render_size = render_size
        self.gamma = gamma
        self.dim_pointfeat = dim_pointfeat

        self.H, self.W = render_size[0], render_size[1]
        self.radius = radius

        self.do_2d_shading = do_2d_shading
        self.pts_dropout_rate = pts_dropout_rate
        self.world_scale = world_scale

    def forward(self, ref_images, target_pose, target_intrinsics, affine_params=None, is_eval=False):
        # target_pose: B x 4 X 4
        # target_intrinsics: B x 3 x 3
        # if is_eval: turn off random dropout.
        do_random_dropout = ((not is_eval) and (self.pts_dropout_rate > 0.0))
        if do_random_dropout:
            num_pts_to_keep = round(self.vert_pos.shape[1] * (1.0 - self.pts_dropout_rate))
            # pts_id_to_keep = torch.tensor(np.random.choice(np.arange(self.vert_pos.shape[1]), size=num_pts_to_keep, replace=False)).cuda()
            pts_id_to_keep = torch.multinomial(torch.ones_like(self.vert_pos[0]), num_pts_to_keep, replacement=False) # this version is much faster. 55ms vs 346ms

        B = target_pose.shape[0]

        # convert self.vert_pos in world coordinates into cam coordinates
        xyz_world = torch.cat((self.vert_pos, torch.ones_like(self.vert_pos[0:1])), dim=0).unsqueeze(0).repeat(B, 1, 1)  # 1 x 4 x N, turned into homogeneous coord

        # target_pose is cam_T_world
        xyz_target = target_pose.bmm(xyz_world)
        xyz_target = xyz_target[:, 0:3]  # B x 3 x N, discard homogeneous dimension

        xy_proj = target_intrinsics.bmm(xyz_target) # B x 3 x N

        eps_mask = (xy_proj[:, 2:3, :].abs() < EPS).detach()

        # Remove invalid zs that cause nans
        zs = xy_proj[:, 2:3, :]
        zs[eps_mask] = EPS

        sampler = torch.cat((xy_proj[:, 0:2, :] / zs, xy_proj[:, 2:3, :]), 1)  # u, v, has range [0,W], [0,H] respectively
        sampler[eps_mask.repeat(1, 3, 1)] = -1e6

        # compute the radius based on the distance of the points to the reference view camera center
        scale_pts = torch.norm(sampler, dim=1) / self.world_scale
        # radius = scale_pts.detach() * self.radius # B x N. detaching here avoids a backward problem
        radius = torch.ones_like(scale_pts) * self.radius
        radius = torch.clamp(radius, min=0.0, max=0.01)
        # radius = torch.ones_like(scale_pts) * self.radius
        if do_random_dropout:
            radius = radius[:, pts_id_to_keep]

        # normlaize to NDC space. flip xy because the ndc coord difinition
        sampler[:, 0, :] = -((sampler[:, 0, :] / self.H) * 2. - (self.W / self.H))
        sampler[:, 1, :] = -((sampler[:, 1, :] / self.H) * 2. - 1.)

        # sampler: B x 3 x num_pts
        xyz_ndc = sampler.permute(0, 2, 1).contiguous() # B x N x 3


        if do_random_dropout:
            xyz_ndc = xyz_ndc[:, pts_id_to_keep] # B x N_drop x 3

        # do shading
        # feat_img = self.reference_encoder(ref_images)  # N x feat_dim x h x w
        # pointfeat = self.unprojector.apply_mask(feat_img, ref_masks_low_res) # N_pts x C


        if self.do_xyz_pos_encode:
            pointfeat = torch.cat([self.vert_feat, self.embedded_vert_pos], dim=0) # dim_pointfeat x N
        else:
            pointfeat = self.vert_feat # N x feat_dim

        # do the smoothing
        if self.knn_3d_smoothing > 0:
            pointfeat = pointfeat.unsqueeze(0) # 1 x N x featdim
            knn_idx = self.vert_knn_idx.unsqueeze(0) # 1 x N x K
            knn_feat = knn_gather(pointfeat, knn_idx) # 1 x N x K x featdim
            knn_feat = knn_feat.squeeze(0) # N x K x featdim

            # try simple alg first, just do averaging
            pointfeat = torch.mean(knn_feat, dim=1) # N x feat_dim


        if do_random_dropout:
            pointfeat = pointfeat[pts_id_to_keep]

        points_feature_flatten = pointfeat.unsqueeze(0).repeat(B, 1, 1).reshape(-1, self.dim_pointfeat)  # (B*N_pts) x feat_dim

        # view_dir = get_view_dir_world(target_pose, self.z_dir)
        view_dir = get_view_dir_world_per_ray(target_pose, xyz_target.detach()) # B x 3 x N
        if do_random_dropout:
            view_dir = view_dir[:, :, pts_id_to_keep]

        view_dir = view_dir.permute(0, 2, 1).reshape(-1, 3)  # (B*N) x 3

        shaded_feature = self.shader(points_feature_flatten, view_dir)  # (B*N) x 3

        if not self.free_opy: # opy from the network
            shaded_feature, shaded_opy = torch.split(shaded_feature, [self.shader_output_channel, 1], dim=1)
            shaded_opy = shaded_opy.reshape(B, xyz_ndc.shape[1]) # B x N

        shaded_feature = shaded_feature.reshape(B, xyz_ndc.shape[1], self.shader_output_channel)
        shaded_feature = shaded_feature.permute(0, 2, 1).contiguous() # B x 3 x N

        # print(shaded_feature.shape)


        # do rendering
        assert xyz_ndc.size(2) == 3
        assert xyz_ndc.size(1) == shaded_feature.size(2)


        xyz_ndc[..., 0:2] *= (float(self.render_size[0]) / float(self.render_size[1]))
        pts3D = Pointclouds(points=xyz_ndc, features=shaded_feature.permute(0, 2, 1))

        # if self.free_opy:
        #     opacity = torch.sigmoid(shaded_opy.squeeze(1).repeat(B, 1)) # B x N
        # else:
        #     opacity = torch.ones_like(xyz_ndc[:, :, 0])

        if self.free_opy:
            opacity = torch.sigmoid(self.vert_opy.unsqueeze(0).repeat(B, 1))
            if do_random_dropout:
                opacity = opacity[:, pts_id_to_keep]
        else:
            # already dropout before feeding into the net, so no need to do here again.
            opacity = torch.sigmoid(shaded_opy) # B x N


        pred = self.renderer(
            pts3D,
            radius=radius,
            gamma=[self.gamma] * B,  # Renderer blending parameter gamma, in [1., 1e-5].
            znear=[1.0] * B,
            zfar=[1e5] * B,
            radius_world=True,
            bg_col=self.bkg_col,
            opacity=torch.clamp(opacity, 0.0, 1.0)
        )
        # pred: B x H x W x 3
        pred = pred.permute(0, 3, 1, 2).contiguous() # B x 3 x H x W

        # vert_valid = torch.sigmoid(self.vert_opy) > torch.mean(torch.sigmoid(self.vert_opy))
        # frame_utils.save_ply('./test.ply', (self.vert_pos.permute(1,0)), ((shaded_feature[0]).permute(1,0)))

        # if cropping_params is not None:
        #     # crop before the shader2d
        #     x0, y0, x1, y1 = cropping_params
        #     pred = pred[:, :, y0:y1, x0:x1]

        if affine_params is not None:
            # crop before the shader2d
            pred = tvF.affine(pred, *affine_params, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)

        if self.do_2d_shading: # do post-processing
            rgb_pred = self.shader_2d(pred)
        else:
            rgb_pred = pred

        return pred, rgb_pred

    def save_pointcloud(self, plyfilename, target_pose, opacity_thres=0.5):
        with torch.no_grad():
            self.eval()
            vert_opy = torch.sigmoid(self.vert_opy)
            # print(torch.max(vert_opy), torch.min(vert_opy), torch.mean(vert_opy))
            vert_valid = vert_opy>torch.mean(vert_opy)
            vert_pos = self.vert_pos.permute(1,0)[vert_valid] # N_valid x 3

            # print('orginal num points: %d, filtered num points: %d' % (len(vert_opy), int(torch.sum(vert_valid).item())))

            # evaluate color
            points_feature_flatten = self.vert_feat  # N x feat_dim

            view_dir = get_view_dir_world(target_pose, self.z_dir)

            shaded_feature = self.shader(points_feature_flatten, view_dir)  # N x 3

            vert_color = shaded_feature[vert_valid] # N_valid x 3

            frame_utils.save_ply(plyfilename, vert_pos, vert_color)

            self.train()

    def evaluate(self, ref_images, target_pose, target_intrinsics, num_random_samples=5, pts_to_use_list=None, target_viewpose=None, fix_seed=False):
        # target_pose: B x 4 X 4
        # target_intrinsics: B x 3 x 3

        if target_viewpose is None: # for making the animation with fixed cam and varying lighting
            target_viewpose = target_pose

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        do_random_dropout = (self.pts_dropout_rate > 0.0)
        if not do_random_dropout:
            num_random_samples = 1

        vert_pos = self.vert_pos
        vert_feat = self.vert_feat
        vert_opy = self.vert_opy

        if pts_to_use_list is not None:
            vert_pos = vert_pos[:, pts_to_use_list]
            vert_feat = vert_feat[pts_to_use_list, :]
            vert_opy = vert_opy[pts_to_use_list]

        num_pts_to_keep = round(self.vert_pos.shape[1] * (1.0 - self.pts_dropout_rate))

        B = target_pose.shape[0]


        # convert self.vert_pos in world coordinates into cam coordinates
        xyz_world = torch.cat((vert_pos, torch.ones_like(vert_pos[0:1])), dim=0).unsqueeze(0).repeat(B, 1, 1)  # 1 x 4 x N, turned into homogeneous coord

        # tagget_pose is cam_T_world
        xyz_target = target_pose.bmm(xyz_world)
        xyz_target = xyz_target[:, 0:3]  # B x 3 x N, discard homogeneous dimension

        xy_proj = target_intrinsics.bmm(xyz_target) # B x 3 x N

        eps_mask = (xy_proj[:, 2:3, :].abs() < EPS).detach()

        # Remove invalid zs that cause nans
        zs = xy_proj[:, 2:3, :]
        zs[eps_mask] = EPS

        sampler = torch.cat((xy_proj[:, 0:2, :] / zs, xy_proj[:, 2:3, :]), 1)  # u, v, has range [0,W], [0,H] respectively
        sampler[eps_mask.repeat(1, 3, 1)] = -1e6

        # compute the radius based on the distance of the points to the reference view camera center
        scale_pts = torch.norm(sampler, dim=1) / self.world_scale
        # radius = scale_pts * self.radius # B x N
        radius = scale_pts.detach() * self.radius  # B x N. detaching here avoids a backward problem
        radius = torch.clamp(radius, min=0.0, max=0.01)

        # normlaize to NDC space. flip xy because the ndc coord difinition
        sampler[:, 0, :] = -((sampler[:, 0, :] / self.H) * 2. - (self.W / self.H))
        sampler[:, 1, :] = -((sampler[:, 1, :] / self.H) * 2. - 1.)

        # sampler: B x 3 x num_pts
        xyz_ndc = sampler.permute(0, 2, 1).contiguous() # B x N x 3

        # do shading
        # feat_img = self.reference_encoder(ref_images)  # N x feat_dim x h x w
        # pointfeat = self.unprojector.apply_mask(feat_img, ref_masks_low_res) # N_pts x C


        if self.do_xyz_pos_encode:
            pointfeat = torch.cat([vert_feat, self.embedded_vert_pos], dim=0) # dim_pointfeat x N
        else:
            pointfeat = vert_feat # N x feat_dim

        if self.knn_3d_smoothing > 0:
            pointfeat = pointfeat.unsqueeze(0) # 1 x N x featdim
            knn_idx = self.vert_knn_idx.unsqueeze(0) # 1 x N x K
            knn_feat = knn_gather(pointfeat, knn_idx) # 1 x N x K x featdim
            knn_feat = knn_feat.squeeze(0) # N x K x featdim
            # try simple alg first, just do averaging
            pointfeat = torch.mean(knn_feat, dim=1) # N x feat_dim

        points_feature_flatten = pointfeat.unsqueeze(0).repeat(B, 1, 1).reshape(-1, self.dim_pointfeat)  # (B*N_pts) x feat_dim

        # view_dir = get_view_dir_world_per_ray(target_pose, xyz_target)
        view_dir = get_view_dir_world_per_ray(target_viewpose, xyz_target)
        view_dir = view_dir.permute(0, 2, 1).reshape(-1, 3)  # (B*N) x 3
        # view_dir = get_view_dir_world(target_pose, self.z_dir)

        end.record()
        torch.cuda.synchronize()
        print('geometry preparation time:', start.elapsed_time(end))



        start.record()

        shaded_feature = self.shader(points_feature_flatten, view_dir)  # (B*N) x 3
        
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        print('shading time:', start.elapsed_time(end))

        if not self.free_opy: # opy from the network
            shaded_feature, shaded_opy = torch.split(shaded_feature, [self.shader_output_channel, 1], dim=1)
            shaded_opy = shaded_opy.reshape(B, xyz_ndc.shape[1]) # B x N

        shaded_feature = shaded_feature.reshape(B, xyz_ndc.shape[1], self.shader_output_channel)
        shaded_feature = shaded_feature.permute(0, 2, 1).contiguous() # B x 3 x N


        # do rendering
        assert xyz_ndc.size(2) == 3
        assert xyz_ndc.size(1) == shaded_feature.size(2)


        xyz_ndc[..., 0:2] *= (float(self.render_size[0]) / float(self.render_size[1]))


        # if self.free_opy:
        #     opacity = torch.sigmoid(shaded_opy.squeeze(1).repeat(B, 1)) # B x N
        # else:
        #     opacity = torch.ones_like(xyz_ndc[:, :, 0])

        if self.free_opy:
            opacity = torch.sigmoid(vert_opy.unsqueeze(0).repeat(B, 1))
        else:
            # already dropout before feeding into the net, so no need to do here again.
            opacity = torch.sigmoid(shaded_opy) # B x N

        all_preds = None
        for _ in range(num_random_samples):
            start.record()
            if fix_seed:
                torch.manual_seed(0)
                np.random.seed(0)

            # pts_id_to_keep = torch.tensor(np.random.choice(np.arange(xyz_ndc.shape[1]), size=num_pts_to_keep, replace=False)).cuda()
            pts_id_to_keep = torch.multinomial(torch.ones_like(vert_pos[0]), num_pts_to_keep, replacement=False) # this version is much faster. 55ms vs 346ms

            end.record()
            torch.cuda.synchronize()
            print('subsampling time:', start.elapsed_time(end))

            xyz_ndc_sampled = xyz_ndc[:, pts_id_to_keep]  # B x N_drop x 3
            shaded_feature_sampled = shaded_feature[..., pts_id_to_keep]
            shaded_opy_sampled = opacity[:, pts_id_to_keep]
            radius_sampled = radius[:, pts_id_to_keep]

            start.record()

            pts3D = Pointclouds(points=xyz_ndc_sampled, features=shaded_feature_sampled.permute(0, 2, 1))

            end.record()
            torch.cuda.synchronize()
            print('pts3d object creation time:', start.elapsed_time(end))

            start.record()

            pred = self.renderer(
                pts3D,
                radius=radius_sampled,
                gamma=[self.gamma] * B,  # Renderer blending parameter gamma, in [1., 1e-5].
                znear=[1.0] * B,
                zfar=[1e5] * B,
                radius_world=True,
                bg_col=self.bkg_col,
                opacity=torch.clamp(shaded_opy_sampled, 0.0, 1.0)
            )
            # pred: B x H x W x 3
            pred = pred.permute(0, 3, 1, 2).contiguous()  # B x 3 x H x W

            # vert_valid = torch.sigmoid(self.vert_opy) > torch.mean(torch.sigmoid(self.vert_opy))
            # frame_utils.save_ply('./test.ply', (self.vert_pos.permute(1,0)), ((shaded_feature[0]).permute(1,0)))

            if all_preds is None:
                all_preds = pred
            else:
                all_preds += pred

            end.record()
            torch.cuda.synchronize()
            print('rasterization time:', start.elapsed_time(end))

        pred = all_preds / float(num_random_samples)



        if self.do_2d_shading: # do post-processing
            start.record()
            rgb_pred = self.shader_2d(pred)
            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()

            print('render time:', start.elapsed_time(end))
        else:
            rgb_pred = pred

        return rgb_pred



def train(args):
    params = {
        "corr_len": 2 if args.pooltype in ['maxmean', "meanvar"] else 1,
        "inference": 0
    }
    for k in list(vars(args).keys()):
        params[k] = vars(args)[k]

    if args.tb_log_dir is not None:
        args.tb_log_dir = os.path.join(args.tb_log_dir, args.name)
        if not os.path.isdir(args.tb_log_dir):
            os.mkdir(args.tb_log_dir)

    with open('dir.json') as f:
        d = json.load(f)
    d = d[args.setting]

    # d["testing_dir"] = os.path.join(d["testing_dir"], args.single)

    HR = params["HR"]
    factor = 4

    unprojector = PtsUnprojector()

    render_scale = args.render_scale

    # factor = int(factor / args.render_scale)


    # extract pts for all views
    gpuargs = {'num_workers': 0, 'drop_last': False, 'shuffle': False}

    datasetname = d["dataset"]
    trainset_args = {
                     "resize": [args.crop_h, args.crop_w],
                     "scan": args.single
                     }

    if datasetname == "LLFF":
        total_num_views = len(sorted(glob.glob(os.path.join(d["testing_dir"], args.single, "DTU_format", "images", "*.jpg"))))

        indicies = np.arange(total_num_views)

        trainset_args["data_augmentation"] = False
        trainset_args["source_views"] = list(indicies[np.mod(np.arange(len(indicies), dtype=int), 8) != 0])
        # trainset_args["source_views"] = list(indicies[np.mod(np.arange(len(indicies), dtype=int), 5) != 2])
        # trainset_args["target_views"] = indicies[np.mod(np.arange(len(indicies), dtype=int), 5) != 2]

    elif datasetname == "DTU":
        trainset_args["return_mask"] = False
        # trainset_args["foreground_mask_path"] = args.foreground_mask_path

        indicies = np.arange(49)

        # trainset_args["source_views"] = indicies[np.mod(np.arange(len(indicies), dtype=int), 8) != 0]
        # trainset_args["target_views"] = indicies[np.mod(np.arange(len(indicies), dtype=int), 8) != 0]

        trainset_args["source_views"] = indicies[np.mod(np.arange(len(indicies), dtype=int), 7) != 2]
        trainset_args["target_views"] = indicies[np.mod(np.arange(len(indicies), dtype=int), 7) != 2]

    elif datasetname == "TNTMasked":
        trainset_args["split"] = "train"
        # trainset_args["target_views"] = indicies[np.mod(np.arange(len(indicies), dtype=int), 20) != 0]


    else:
        raise NotImplementedError

    valset_args = trainset_args.copy()

    if datasetname == "LLFF":
        valset_args["source_views"] = list(indicies[np.mod(np.arange(len(indicies), dtype=int), 8) == 0])
        # valset_args["source_views"] = list(indicies[np.mod(np.arange(len(indicies), dtype=int), 5) == 2])
    elif datasetname == "DTU":
        valset_args["target_views"] = indicies[np.mod(np.arange(len(indicies), dtype=int), 7) == 2]
    elif datasetname == "TNTMasked":
        valset_args["split"] = "test"
    else:
        raise NotImplementedError

    # turn off random scale and crop
    valset_args["resize"] = [args.crop_h, args.crop_w]

    train_dataset = eval(datasetname+'ViewsynTrain')(d["testing_dir"], args.pointcloud_dir, **trainset_args)
    val_dataset = eval(datasetname+'ViewsynTrain')(d["testing_dir"], args.pointcloud_dir, **valset_args)

    valset_args['factor'] = factor
    valset_args['render_scale'] = render_scale
    valset_args['loss_type'] = args.fe_loss_type
    valset_args["crop_size"] = [args.crop_h, args.crop_w] # 2x to get original TNT size
    # valset_args['resize_h'] = args.resize_h
    # valset_args['resize_w'] = args.resize_w

    train_loader = DataLoader(train_dataset, batch_size=1, **gpuargs)
    val_loader = DataLoader(val_dataset, batch_size=1, **gpuargs)


    trainset_images = []
    # trainset_depth_masks = []
    trainset_loss_masks = []
    # trainset_depths = []
    trainset_poses = []
    trainset_intrinsics = []
    # trainset_vgg_features = []
    #
    # vgg_model = torchvision.models.vgg16(pretrained=True).cuda()
    # def hook(module, input, output):
    #     trainset_vgg_features.append(output.detach())

    # print(vgg_model.features[15])
    # vgg_model.features[15].register_forward_hook(hook) # this is the last relu layer before the 3-rd downsampling, so the spatial resolution is 1/4, which matches our downsampling rate.

    # put all training samples into memory
    for i_batch, data_blob in enumerate(train_loader):

        images, poses, intrinsics = data_blob
        loss_masks = torch.ones_like(images[:, :, 0]) # everything

        # now just simple filtering. later we may use more clever pre-filtering
        images = images.cuda()
        poses = poses.cuda()
        intrinsics = intrinsics.cuda()
        loss_masks = loss_masks.cuda()
        loss_masks = loss_masks.unsqueeze(2)

        rgb_gt = images[:, 0] * 2.0 / 255.0 - 1.0  # range [-1, 1], 1 x H x W
        rgb_gt = F.interpolate(rgb_gt, [params['crop_h'] // (factor // render_scale), params['crop_w'] // (factor // render_scale)], mode='bilinear', align_corners=True)
        loss_mask_gt = F.interpolate(loss_masks[:, 0], [params['crop_h'] // (factor // render_scale), params['crop_w'] // (factor // render_scale)], mode='nearest') # B x 1 x H x W



        # mask_lowres = F.interpolate(depth_masks[:, 0], [params['crop_h'] // factor, params['crop_w'] // factor], mode='nearest')  # B x 1 x H x W

        intrinsics_gt = intrinsics[:, 0] # B x 4 x 4
        intrinsics_gt[:, 0] /= (images.shape[3] / (params['crop_h'] // factor)) # rescale according to the ratio between dataset images and render images
        intrinsics_gt[:, 1] /= (images.shape[4] / (params['crop_w'] // factor))

        trainset_images.append(rgb_gt)
        trainset_loss_masks.append(loss_mask_gt)
        # trainset_masks.append(mask_gt)
        trainset_poses.append(poses[:, 0])
        trainset_intrinsics.append(intrinsics_gt)

        # xyzs, buvs = unprojector(depth_low_res[:, 0], poses[:, 0], intrinsics_gt, mask=depth_mask_gt, return_coord=True) # N x 3
        # buvs[:, 0] = i_batch
        #
        # trainset_xyzs.append(xyzs)
        # trainset_buvs.append(buvs)

        # with torch.no_grad():
        #     inp = images[:, 0] / 255.0
        #     inp = tvF.normalize(inp, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     _ = vgg_model(inp) # add feature to output

    # stack
    trainset_images = torch.cat(trainset_images, dim=0)
    trainset_loss_masks = torch.cat(trainset_loss_masks, dim=0)
    trainset_poses = torch.cat(trainset_poses, dim=0)
    trainset_intrinsics = torch.cat(trainset_intrinsics, dim=0)
    # trainset_vgg_features = torch.cat(trainset_vgg_features, dim=0) # N x 256 x H x W

    # trainset_xyzs = []
    # trainset_buvs = []
    #
    # for i in range(len(trainset_images)):
    #     xyzs, buvs = unprojector(trainset_depths[i:i+1], trainset_poses[i:i+1], trainset_intrinsics[i:i+1], mask=trainset_depth_masks[i:i+1], return_coord=True)  # N x 3
    #     buvs[:, 0] = i
    #
    #     trainset_xyzs.append(xyzs)
    #     trainset_buvs.append(buvs)
    #
    # trainset_xyzs = torch.cat(trainset_xyzs, dim=0) # N x 3
    # trainset_buvs = torch.cat(trainset_buvs, dim=0) # N x 3. these are cooresponding to xyzs, so that we can index into the images/feature for each point

    trainset_xyzs = train_dataset.get_pointclouds()
    trainset_xyzs = trainset_xyzs.permute(1, 0) # N x 3

    # # debug only
    # color_orginal = trainset_images[trainset_buvs[:,0], :, trainset_buvs[:,1], trainset_buvs[:,2]].cpu()
    # frame_utils.save_ply('./pts_full_scan48.ply', trainset_xyzs.cpu(), color_orginal)
    # return


    if args.restore_pointclouds is None:
        vert_pos = trainset_xyzs.permute(1,0) # 3xN
        # buvs = trainset_buvs.permute(1,0) # 3xN
    else:
        print('loading points from %s' % args.restore_pointclouds)
        if args.restore_pointclouds.endswith(".pt"):
            tmp = torch.load(args.restore_pointclouds)
            vert_pos = tmp['xyzs'].permute(1,0) # 3xN
            buvs = tmp['buvs'].permute(1,0) # 3xN
        elif args.restore_pointclouds.endswith(".ply"):
            vert_pos = frame_utils.load_ply(args.restore_pointclouds) # N x 3
            vert_pos = torch.tensor(vert_pos).permute(1,0) # 3xN
            buvs = None
        else:
            raise NotImplementedError

    print('total points we gonna use: %d' % vert_pos.shape[1])
    # print(trainset_xyzs.shape)

    max_num_pts = args.max_num_pts
    # max_num_pts = 100000000000
    # if datasetname == "LLFF": # seems DTU doesn't have the mem issue
    #     max_num_pts = 4500000  # safe number for dropout rate=0.5
    # elif datasetname == "DTU":
    #     max_num_pts = 4800000
    # else:
    #     raise NotImplementedError

    if vert_pos.shape[1] > max_num_pts:
        print('warning: random dropping points to %d due to memory limit...' % max_num_pts)
        vert_id_to_keep = np.random.choice(np.arange(vert_pos.shape[1]), size=max_num_pts, replace=False)
        vert_pos = vert_pos[:, vert_id_to_keep]

    # if args.cnn_feat:
    #     pointfeat_init = trainset_vgg_features[buvs[0], :, buvs[1], buvs[2]]  # N x feat_dim
    #     print('pointfeat_init:', pointfeat_init)
    #     print(torch.max(pointfeat_init), torch.min(pointfeat_init), torch.mean(pointfeat_init))
    # else:
    #     pointfeat_init = None
    pointfeat_init = None

    if datasetname == "LLFF":
        bkg_col = (-1, -1, 1)
        tau_E = 4.0
        shallowest_few = 5
        pointadd_dropout = 0.0
        min_depth = 800
        max_depth = 1e4 # this number can be arbitrarily large, as we will sample in the disparity space
        lindisp=True

    elif datasetname == "DTU":
        bkg_col = (-1, -1, -1)
        tau_E = 4.0
        shallowest_few = 5
        pointadd_dropout = 0.0
        min_depth = 800
        max_depth = 1400
        lindisp = False

    elif datasetname == "TNT":
        bkg_col = (-1, -1, -1)

    elif datasetname == "TNTMasked":
        bkg_col = (1, 1, 1)

    else:
        raise NotImplementedError

    model = PulsarSceneModel(vert_pos=vert_pos, pointfeat_init=pointfeat_init, dim_pointfeat=args.dim_pointfeat,
                             render_size=(params['crop_h'] // factor, params['crop_w'] // factor),
                             render_scale=render_scale, gamma=args.blend_gamma, batch_size=args.batch_size, radius=args.sphere_radius,
                             free_xyz=args.free_xyz, free_opy=args.free_opy, free_rad=args.free_rad, bkg_col=bkg_col,  # green for debugging, red for llff
                             do_2d_shading=args.do_2d_shading, shader_arch=args.shader_arch, pts_dropout_rate=args.pts_dropout_rate, do_xyz_pos_encode=args.do_xyz_pos_encode,
                             basis_type=args.basis_type, shader_output_channel=args.shader_output_channel, knn_3d_smoothing=args.knn_3d_smoothing, shader_norm=args.shader_norm).cuda()

    if args.restore_ckpt is not None:
        tmp = torch.load(args.restore_ckpt)
        if list(tmp.keys())[0][:7] == "module.":
            model = nn.DataParallel(model)
        model.load_state_dict(tmp, strict=False)

    if args.freeze_shader:
        for param in model.shader.parameters():
            param.requires_grad = False

    # optimizer
    optimizer, scheduler = fetch_optimizer(args, model)

    scaler = GradScaler(enabled=True)
    logger = Logger(model, scheduler, args.outputfile, args.SUM_FREQ, args.img_log_freq, args.tb_log_dir)

    if args.render_only:
        assert (args.restore_ckpt is not None)
        # this gives worse performance now, so turn it off.
        # pts_to_use_list = check_pts_gradient(model, trainset_images, train_loader, valset_args, logger)
        pts_to_use_list=None
        # make_animation(model, val_dataset, trainset_intrinsics, logger, rasterize_rounds=1)

        validate(model, trainset_images, val_loader, valset_args, logger, pts_to_use_list, val_cam_noise=args.val_cam_noise, rasterize_rounds=args.rasterize_rounds)

        logger.close()
        return

    point_add_debug = False

    if args.eval_only:
        assert (args.restore_ckpt is not None)
        # model.save_pointcloud(args.plyfilename, trainset_poses[0:1]) # select a middle view



        positive_area = extract_error_map(model, tau_E, trainset_images, trainset_depth_masks, trainset_depths, trainset_poses, trainset_intrinsics, train_loader, valset_args, logger)

        if point_add_debug:
            tmp = {'positive_area': positive_area}
            torch.save(tmp, './temp_%s.pt' % args.single)



        if point_add_debug:
            tmp = torch.load('./temp_%s.pt' % args.single)
            positive_area = tmp['positive_area']

        points_added, buvs_added = add_points(positive_area, trainset_images, trainset_depths, trainset_depth_masks, trainset_poses, trainset_intrinsics, min_depth=min_depth, max_depth=max_depth, pts_dropout_rate=pointadd_dropout, shallowest_few=shallowest_few, lindisp=lindisp)

        points_added = points_added.permute(1,0)  # N x 3
        buvs_added = buvs_added.permute(1,0)  # N x 3

        print('number of points to add: %d' % points_added.shape[0])
        color_added = torch.tensor([0, 0, 1]).reshape(1, 3).repeat(points_added.shape[0], 1) # red
        # color_added = trainset_images[buvs_added[:,0], :, buvs_added[:,1], buvs_added[:,2]].cpu()

        points_original = trainset_xyzs # N x 3
        # index into colors
        # print(trainset_buvs.shape)
        # print(torch.min(trainset_buvs[:, 0]), torch.max(trainset_buvs[:, 0]))
        # print(torch.min(trainset_buvs[:, 1]), torch.max(trainset_buvs[:, 1]))
        # print(torch.min(trainset_buvs[:, 2]), torch.max(trainset_buvs[:, 2]))
        # print(trainset_images.shape)
        color_orginal = trainset_images[trainset_buvs[:,0], :, trainset_buvs[:,1], trainset_buvs[:,2]].cpu() # N x 3
        # color_orginal = torch.tensor([1, 0, 0]).reshape(1, 3).repeat(points_original.shape[0], 1) # blue

        frame_utils.save_ply('./pointclouds/%s.ply' % args.name, torch.cat([points_added, points_original]), torch.cat([color_added, color_orginal]))

        # for debugging only
        frame_utils.save_ply('./pointclouds/after_pruning_%s.ply' % args.name, points_original, color_orginal)
        frame_utils.save_ply('./pointclouds/aded_%s.ply' % args.name, points_added, color_added)

        tmp = {'xyzs': torch.cat([points_added, points_original]).cpu(), 'buvs': torch.cat([buvs_added, trainset_buvs]).cpu()} # N x 3
        torch.save(tmp, './pointclouds/%s.pt' % args.name)

        logger.close()
        return

    VAL_FREQ = args.VAL_FREQ

    tic = None
    total_time = 0

    best_score = 1.0 # this is high enough

    for total_steps in range(1, args.num_steps+1):

        optimizer.zero_grad()

        # train_id = np.random.randint(len(trainset_images))
        # train_id = np.array([0, ]) # debug

        train_id = np.random.choice(np.arange(len(trainset_images)), args.batch_size, replace=False)

        rgb_gt = trainset_images[train_id] # b x 3 x H x W
        target_pose = trainset_poses[train_id] # b x 4 x 4
        target_intrinsics = trainset_intrinsics[train_id] # b x 3 x 3
        mask_gt = trainset_loss_masks[train_id] # b x 1 x H x W

        # random crop the target
        # get cropping params in format (x0, y0, x1, y1)
        # x0 = np.random.randint(0, params['crop_w']//factor - params['resize_w']//factor + 1)
        # y0 = np.random.randint(0, params['crop_h']//factor - params['resize_h']//factor + 1)
        # x1 = x0 + params['resize_w']//factor
        # y1 = y0 + params['resize_h']//factor
        #
        # mask_gt = mask_gt[:, :, y0:y1, x0:x1]
        # rgb_gt = rgb_gt[:, :, y0:y1, x0:x1]
        #
        # cropping_params = (x0, y0, x1, y1)

        if args.do_random_affine:
            affine_params = tvT.RandomAffine(0).get_params(degrees=(-30, 30), translate=(0.1, 0.1), scale_ranges=(0.9, 1.1), shears=None, img_size=(rgb_gt.shape[-1], rgb_gt.shape[-2]))
            rgb_gt, mask_gt = tvF.affine(rgb_gt, *affine_params, interpolation=torchvision.transforms.InterpolationMode.BILINEAR), tvF.affine(mask_gt, *affine_params, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        else:
            affine_params = None

        # random crop the reference
        # ref_inputs = torch.cat([trainset_images, trainset_depth_masks, trainset_depths], dim=1)
        # ref_inputs, ref_intrinsics = crop_operation(ref_inputs, trainset_intrinsics, params['resize_h']//factor, params['resize_w']//factor)
        # ref_images, ref_masks, ref_depths = torch.split(ref_inputs, [3,1,1], dim=1)
        # ref_poses = trainset_poses

        # # debugging exp.
        # val_id = np.array([0, ])
        # rgb_gt = valset_images[val_id]
        # mask_gt = valset_masks[val_id]
        # target_pose = valset_poses[val_id]
        # target_intrinsics = valset_intrinsics[val_id]

        # do the rendering
        # rgb_est = model(target_pose, target_intrinsics, ref_images, ref_masks, ref_depths, ref_poses, ref_intrinsics, cropping_params) # b x 3 x H x W
        # feat_est, rgb_est = model(trainset_images, target_pose, target_intrinsics, cropping_params)  # b x 3 x H x W
        feat_est, rgb_est = model(trainset_images, target_pose, target_intrinsics, affine_params)  # b x 3 x H x W


        loss_type = args.fe_loss_type

        loss, metrics = sequence_loss_rgb([rgb_est,], rgb_gt, mask_gt, loss_type=loss_type)
        # rgb_gt = rgb_gt * mask_gt + 1.0 * (1.0 - mask_gt)
        # loss, metrics = sequence_loss_rgb([rgb_est, ], rgb_gt, loss_type=

        # add a smoothness loss here
        if args.feat_smooth_loss_coeff > 0.0:
            if render_scale != 1:
                ht, wd = rgb_est.shape[-2:]
                feat_est = F.interpolate(feat_est, [ht, wd], mode='bilinear', align_corners=True)

            feat_smooth_loss = smoothnessloss(feat_est, mask_gt)
            loss += args.feat_smooth_loss_coeff * feat_smooth_loss
            metrics['unscaled_feat_smooth_loss'] = feat_smooth_loss

        scaler.scale(loss).backward()

        # print(torch.min(model.vert_pos.grad), torch.max(model.vert_pos.grad))
        # print(torch.min(model.vert_feat.grad), torch.max(model.vert_feat.grad))

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        if scheduler is not None:
            scheduler.step()
        scaler.update()

        logger.push(metrics, 'train')
        logger.summ_rgb('train/rgb_gt_unmasked', rgb_gt)
        logger.summ_rgb('train/rgb_gt', rgb_gt, mask_gt)
        logger.summ_rgb('train/rgb_est', rgb_est, mask_gt)
        logger.summ_rgb('train/rgb_est_unmasked', rgb_est)
        # logger.summ_rgb('train/rgb_gt', rgb_gt)
        # logger.summ_rgb('train/rgb_est', rgb_est)


        if total_steps % VAL_FREQ == VAL_FREQ - 1:
            res = validate(model, trainset_images, val_loader, valset_args, logger, val_cam_noise=args.val_cam_noise, rasterize_rounds=args.rasterize_rounds)
            cur_score = res['avg']
            if cur_score < best_score: # best. avg the lower the better
                best_score = cur_score
                PATH = 'checkpoints/model_best_%s.pth' % args.name
                torch.save(model.state_dict(), PATH)


        logger.set_global_step(total_steps)

        if not tic is None:
            total_time += time.time() - tic
            print(
                f"time per step: {total_time / (total_steps - 1)}, expected: {total_time / (total_steps - 1) * args.num_steps / 24 / 3600} days")
            print(args.name)
        tic = time.time()

    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    validate(model, trainset_images, val_loader, valset_args, logger, val_cam_noise=args.val_cam_noise, rasterize_rounds=args.rasterize_rounds)

    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # #================= not used =======================
    # parser.add_argument('--stage', help="determines which dataset to use for training")
    # parser.add_argument('--mode', default='stereo')
    # parser.add_argument('--small', action='store_true', help='use small model')
    # parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    # parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    # parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    # parser.add_argument('--clip', type=float, default=1.0)
    # parser.add_argument('--dropout', type=float, default=0.0)
    # parser.add_argument('--gamma', type=float, default=0.9, help='exponential weighting')
    # parser.add_argument('--add_noise', action='store_true')
    # parser.add_argument('--validation', type=str, nargs='+')
    # parser.add_argument('--debug', type=int, default=False)
    # #================= not used =======================

    ''' training args'''
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--restore_ckpt', default=None, help="restore checkpoint")
    parser.add_argument('--restore_raft_only', default=None, help="restore cer-mvs raft ckpt")
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--pct_start', type=float, default=0.001)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--pause_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--shuffle', type=int, default=True)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--SUM_FREQ', type=int, default=100)
    parser.add_argument('--VAL_FREQ', type=int, default=5000)
    parser.add_argument('--outputfile', type=str,
                        default=None)  # in case stdoutput is buffered (don't know how to disable buffer...)
    parser.add_argument('--tb_log_dir', type=str, default=None)  # tensorboard log dir
    parser.add_argument('--img_log_freq', type=int, default=100)  # tensorboard log dir
    parser.add_argument('--eval_only', type=int, default=False)
    parser.add_argument('--render_only', type=int, default=False)

    '''loss args'''
    parser.add_argument('--loss_type', type=str, default='depth_gradual',
                        choices=['disp', "depth", "depth_softloss", 'depth_gradual'])
    parser.add_argument('--depthloss_threshold', type=float, default=100)
    parser.add_argument('--disp_runup', type=int, default=10000)
    parser.add_argument('--Euclidean', type=int, default=True)

    '''dataset args'''
    parser.add_argument('--setting', type=str, default='DTU')
    parser.add_argument('--num_frames', type=int, default=5)
    parser.add_argument('--light_number', type=int, default=-1)
    parser.add_argument('--crop_h', type=int, default=448)
    parser.add_argument('--crop_w', type=int, default=576)
    parser.add_argument('--resize_h', type=int, default=-1)
    parser.add_argument('--resize_w', type=int, default=-1)
    parser.add_argument('--pairs_provided', type=int, default=0)
    parser.add_argument('--scaling', type=str, default="median")
    parser.add_argument('--image_aug', type=int, default=False)
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--single', type=str, default=None)  # train on a single scene
    parser.add_argument('--precomputed_depth_path', type=str, default=None)  # train on a single scene
    parser.add_argument('--pointcloud_dir', type=str, default=None)  # train on a single scene

    '''model args'''
    '''prediction range'''
    parser.add_argument('--Dnear', type=float, default=.0025)
    parser.add_argument('--DD', type=int, default=128)
    parser.add_argument('--Dfar', type=float, default=.0)
    parser.add_argument('--DD_fine', type=int, default=320)
    # parser.add_argument('--len_dyna', type=int, default=44)
    '''layer and kernel size'''
    parser.add_argument('--kernel_z', type=int, default=3)
    parser.add_argument('--kernel_r', type=int, default=3)
    parser.add_argument('--kernel_q', type=int, default=3)
    parser.add_argument('--kernel_corr', type=int, default=3)
    parser.add_argument('--dim0_corr', type=int, default=128)
    parser.add_argument('--dim1_corr', type=int, default=128)
    parser.add_argument('--dim_net', type=int, default=128)
    parser.add_argument('--dim_inp', type=int, default=128)
    parser.add_argument('--dim0_delta', type=int, default=256)
    parser.add_argument('--kernel0_delta', type=int, default=3)
    parser.add_argument('--kernel1_delta', type=int, default=3)
    parser.add_argument('--dim0_upmask', type=int, default=256)
    parser.add_argument('--kernel_upmask', type=int, default=3)
    parser.add_argument('--num_levels', type=int, default=5)
    # parser.add_argument('--radius', type=int, default=5)
    parser.add_argument('--s_disp_enc', type=int, default=7)
    parser.add_argument('--dim_fmap', type=int, default=128)
    '''variants'''
    parser.add_argument('--num_iters', type=int, default=16)
    parser.add_argument('--HR', type=int, default=False)
    parser.add_argument('--HRv2', type=int, default=False)
    parser.add_argument('--cascade', type=int, default=False)
    parser.add_argument('--cascade_v2', type=int, default=False)
    parser.add_argument('--num_iters1', type=int, default=8)
    parser.add_argument('--num_iters2', type=int, default=5)
    parser.add_argument('--slant', type=int, default=False)
    parser.add_argument('--invariance', type=int, default=False)
    parser.add_argument('--pooltype', type=str, default="maxmean")
    parser.add_argument('--no_upsample', type=int, default=False)
    parser.add_argument('--merge', type=int, default=False)
    parser.add_argument('--visibility', type=int, default=False)
    parser.add_argument('--visibility_v2', type=int, default=False)
    # parser.add_argument('--merge_permute', type=int, default=0)

    '''render args'''
    parser.add_argument('--output_appearance_features', type=int, default=True)
    parser.add_argument('--detach_depth', type=int, default=False)

    parser.add_argument('--fe_render_iters', type=int, default=8)
    parser.add_argument('--fe_dim_pointfeat', type=int, default=256)
    parser.add_argument('--fe_n_neighbors', type=int, default=4)
    parser.add_argument('--fe_dim_inp', type=int, default=128)
    parser.add_argument('--fe_dim_net', type=int, default=128)
    parser.add_argument('--fe_dim0_delta', type=int, default=256)
    parser.add_argument('--fe_kernel0_delta', type=int, default=3)
    parser.add_argument('--fe_kernel1_delta', type=int, default=3)
    parser.add_argument('--fe_output_opacity', type=int, default=False)
    parser.add_argument('--fe_loss_type', type=str, default='l1')

    parser.add_argument('--dtu_return_mask', type=int, default=False)
    parser.add_argument('--foreground_mask_path', type=str, default=None)
    parser.add_argument('--freeze_shader', type=int, default=False)
    # parser.add_argument('--do_shading', type=int, default=True)
    parser.add_argument('--sphere_radius', type=float, default=7.5e-4)
    parser.add_argument('--free_xyz', type=int, default=False)
    parser.add_argument('--free_opy', type=int, default=False)
    parser.add_argument('--free_rad', type=int, default=False)
    parser.add_argument('--cnn_feat', type=int, default=False, help='extract feature with cnn. else treat as free params')
    parser.add_argument('--blend_gamma', type=float, default=1e-4, help='gamma for blending')
    parser.add_argument('--render_scale', type=int, default=1, help='generate higher resolution images')
    # parser.add_argument('--do_color_residual', type=int, default=False)
    parser.add_argument('--do_2d_shading', type=int, default=False)
    parser.add_argument('--shader_arch', type=str, default='simple_unet')
    parser.add_argument('--shader_norm', type=str, default='none')
    parser.add_argument('--basis_type', type=str, default='mlp', help="the basis type to use for modeling the non-Lambertian effect. option: mlp;SH")
    parser.add_argument('--shader_output_channel', type=int, default=128)
    parser.add_argument('--pts_dropout_rate', type=float, default=0.0)
    parser.add_argument('--dim_pointfeat', type=int, default=16)
    parser.add_argument('--do_xyz_pos_encode', type=int, default=False)
    parser.add_argument('--plyfilename', type=str, default='')
    parser.add_argument('--restore_pointclouds', type=str, default=None)
    parser.add_argument('--knn_3d_smoothing', type=int, default=0, help="do the k-nearest-neighbor smoothing of point features in 3D")
    parser.add_argument('--feat_smooth_loss_coeff', type=float, default=0.0)
    parser.add_argument('--do_random_affine', type=int, default=False)
    parser.add_argument('--do_check_depth_consistency', type=int, default=True, help="do filtering based on view consistency. default is True. turn off only for ablation study")
    parser.add_argument('--max_num_pts', type=int, default=1000000000)
    parser.add_argument('--val_cam_noise', type=float, default=0.0, help='add noise to the cam pose during eval. for debug only')
    parser.add_argument('--rasterize_rounds', type=int, default=5)
    # parser.add_argument('--special_args_dict', type=dict, default={})
    parser.add_argument(
        '--special_args_dict',
        type=lambda x: {k: float(v) for k, v in (i.split(':') for i in x.split(','))},
        default={},
        help='comma-separated field:position pairs, e.g. Date:0,Amount:2,Payee:5,Memo:9'
    )

    # parser.add_argument('--fix_mistake', type=int, default=False)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)





