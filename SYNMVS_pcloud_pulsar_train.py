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

from raft import RAFT
from syn import SYNViewsynTrain
from projector import Shader
from modules.extractor import BasicEncoder, SameResEncoder
from modules.unet import SmallUNet, UNet
import frame_utils
from geom_utils import Lie

import projective_ops as pops

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from collections import OrderedDict
import subprocess

# from skimage.metrics import structural_similarity
import lpips

from pytorch3d.structures import Pointclouds

from pytorch3d.renderer import (
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRasterizer
)

from pulsar.unified import PulsarPointsRenderer

from plyfile import PlyData, PlyElement

EPS = 1e-2

# https://en.wikipedia.org/wiki/Rotation_matrix
# ==> Rotation matrix from axis and angle
def rotate_points_around_axis_center(xyzs, center, axis, angle):
    # xyzs is 3 x N
    axis = axis / np.linalg.norm(axis)
    ux, uy, uz = axis
    R = np.array([[np.cos(angle) + ux**2 * (1 - np.cos(angle)),
          ux * uy * (1 - np.cos(angle)) - uz * np.sin(angle),
          ux * uz * (1 - np.cos(angle)) + uy * np.sin(angle)],
         [uy * ux * (1 - np.cos(angle)) + uz * np.sin(angle),
          np.cos(angle) + uy**2 * (1 - np.cos(angle)),
          uy * uz * (1 - np.cos(angle)) - ux * np.sin(angle)],
         [uz * ux * (1 - np.cos(angle)) - uy * np.sin(angle),
          uz * uy * (1 - np.cos(angle)) + ux * np.sin(angle),
          np.cos(angle) + uz**2 * (1 - np.cos(angle))]], dtype=np.float32)

    xyzs_centered = xyzs - center.reshape(3, 1) # 3 x N
    xyzs_centered_rotated = R @ xyzs_centered # 3 x N
    xyzs_rotated = xyzs_centered_rotated + center.reshape(3, 1)
    return xyzs_rotated

def get_view_dir_world(ref_pose, view_dir_cam):
    view_dir = (torch.inverse(ref_pose) @ view_dir_cam.view(1, 4, 1))  # B x 4 x 1
    view_dir = view_dir[:, 0:3]  # B x 3 x 1
    view_dir = view_dir.repeat(1, 1, xyz_ndc.shape[1])  # B x 3 x N
    # view_dir = view_dir.permute(0, 2, 1).reshape(-1, 3)  # (B*N) x 3

    return view_dir

def get_view_dir_world_per_ray(ref_pose, view_dir_cam):
    # view_dir_cam is B x 3 x N
    view_dir_cam = view_dir_cam / torch.linalg.norm(view_dir_cam, dim=1, keepdim=True) # B x 3 x N
    view_dir_cam = torch.cat([view_dir_cam, torch.zeros_like(view_dir_cam[:, 0:1])], dim=1) # B x 4 x N. turn into homogeneous coord. last dim zero because we want dir only
    view_dir = (torch.inverse(ref_pose) @ view_dir_cam) # B x 4 x N
    view_dir = view_dir[:, 0:3]  # B x 3 x N

    return view_dir

def compute_ssim(img1, img2):
    # img1, img2: [0, 255]

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map

def crop_operation(images, intrinsics, crop_h, crop_w, mod='random'):
    B, _, H, W = images.shape
    # concat all things together on feat dim if you want to crop, say, depth as well.

    new_images = []
    new_intrinsics = []

    for b in range(B):
        if mod == 'random':
            x0 = np.random.randint(0, W - crop_w + 1)
            y0 = np.random.randint(0, H - crop_h + 1)
        elif mod == 'center':
            x0 = (wd1 - crop_w) // 2
            y0 = (ht1 - crop_h) // 2
        else:
            raise NotImplementedError

        x1 = x0 + crop_w
        y1 = y0 + crop_h
        new_image = images[b, :, y0:y1, x0:x1]
        new_intrinsic = torch.clone(intrinsics[b])

        new_intrinsic[0, 2] -= x0
        new_intrinsic[1, 2] -= y0

        new_images.append(new_image)
        new_intrinsics.append(new_intrinsic)

    new_images = torch.stack(new_images, dim=0)
    new_intrinsics = torch.stack(new_intrinsics, dim=0)

    return new_images, new_intrinsics

def fetch_optimizer(args, model):
    # todo: enable the dict
    """ Create the optimizer and learning rate scheduler """
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

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, param_group_lr_list, args.num_steps + 100,
        pct_start=args.pct_start, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def sequence_loss_rgb(rgb_est,
                        rgb_gt,
                        mask_gt=None,
                        loss_type='l1',
                        lpips_vgg=None,
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
        else:
            raise NotImplementedError

        if not weight is None:
            i_loss *= weight

        flow_loss += i_weight * i_loss.mean()

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
            lpips_val = lpips_vgg(rgb_gt[:, [2,1,0]], rgb_est[-1][:, [2,1,0]]) # input should have range [-1,1]
            lpips_val = lpips_val.mean().cpu().item()
            metrics['lpips'] = lpips_val

    print(metrics)

    return flow_loss, metrics

def make_animation_simple(model, vert_pos, val_dataset, ref_intrinsics, logger, rasterize_rounds=1):
    model.eval()
    metrics = {}

    # make a video with varying cam pose
    render_poses = val_dataset.get_render_poses()
    # render_viewpose= val_dataset.get_render_poses(radius=20) # larger movement

    # num_poses = vert_poses.shape[0]
    num_poses = 60

    trans_t = lambda t: torch.Tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]]).float()

    rot_phi = lambda phi: torch.Tensor([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]]).float()

    rot_theta = lambda th: torch.Tensor([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]]).float()

    def pose_spherical(theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi / 180. * np.pi) @ c2w
        c2w = rot_theta(theta / 180. * np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
        w2c = torch.linalg.inv(c2w)
        w2c = torch.Tensor(np.array([[1., 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])) @ w2c

        return w2c

    # vert_poses is n_poses x 3 x N tensor
    # create the animation elements
    # first move around at the original pose
    all_cam_extrinsics = []
    all_view_extrinsics = []

    # center_azimuth = 180.0
    center_azimuth = 135.0
    azimuth_range = 30

    ele = 45.0
    cam_extrinsics = pose_spherical(center_azimuth, -ele, 4.0).reshape(1, 4, 4).repeat(num_poses, 1, 1)  # N x 4 x 4
    # view_extrinsics = torch.stack([pose_spherical(angle, -ele, 4.0) for angle in np.linspace(-180 + 45, 180 + 45, num_poses + 1)[:-1]], 0)  # N x 4 x 4
    view_extrinsics = torch.stack(
        [pose_spherical(angle, -ele, 4.0) for angle in np.linspace(center_azimuth - azimuth_range, center_azimuth + azimuth_range, num_poses//2 + 1)[:-1]] +
        [pose_spherical(angle, -ele, 4.0) for angle in np.linspace(center_azimuth + azimuth_range, center_azimuth - azimuth_range, num_poses//2 + 1)[:-1]],
        0)  # N x 4 x 4

    all_cam_extrinsics.append(cam_extrinsics)
    all_view_extrinsics.append(view_extrinsics)

    all_cam_extrinsics = torch.cat(all_cam_extrinsics, 0).cuda()
    all_view_extrinsics = torch.cat(all_view_extrinsics, 0).cuda()

    all_cam_extrinsics[:, 0:3, 3] *= 400.
    all_view_extrinsics[:, 0:3, 3] *= 400.


    N_views = render_poses.shape[0]

    # pre-select subset to reduce flickering
    # num_pts_to_keep = round(vert_pos.shape[1] * (1.0 - model.pts_dropout_rate))
    # pts_id_to_keep = torch.multinomial(torch.ones_like(vert_pos[0]), num_pts_to_keep, replacement=False)

    all_frames = []
    with torch.no_grad():
        for i_batch in range(N_views):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            target_pose = render_poses[i_batch:i_batch+1] # 1 x 4 x 4

            start.record()
            rgb_est = model.evaluate(vert_pos, None, target_pose, ref_intrinsics[0:1], num_random_samples=rasterize_rounds, pts_to_use_list=None, fix_seed=True)  # 1 x 3 x H x W
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

    # make a video with fixed cam pose and varying lighting dir
    all_frames = []
    with torch.no_grad():
        for i_batch in range(num_poses):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            target_viewpose = all_view_extrinsics[i_batch:i_batch+1] # 1 x 4 x 4
            # target_pose = render_poses[int(3*N_views/4):int(3*N_views/4)+1] # 1 x 4 x 4
            # target_pose = render_poses[0:1]  # 1 x 4 x 4
            target_pose = all_cam_extrinsics[i_batch:i_batch+1]  # 1 x 4 x 4

            start.record()
            rgb_est = model.evaluate(vert_pos, None, target_pose, ref_intrinsics[0:1], num_random_samples=rasterize_rounds, pts_to_use_list=None, target_viewpose=target_viewpose, fix_seed=True)  # 1 x 3 x H x W
            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()

            print('total render time for one image:', start.elapsed_time(end))

            all_frames.append(rgb_est)


    all_frames = torch.cat(all_frames)
    logger.summ_rgbs('animation/viewdir', all_frames, fps=20, force_save=True)

    all_frames = (all_frames + 1.0) / 2.0
    all_frames = torch.clamp(all_frames, 0.0, 1.0)
    all_frames = all_frames[:, [2, 1, 0]]  # bgr2rgb, N x 3 x H x W, range [0,1]
    all_frames = all_frames.permute(0, 2, 3, 1).cpu().numpy()  # N x H x W x 3, range [0,1]

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    imageio.mimwrite(os.path.join('./saved_videos/view_%s.wmv' % args.name), to8b(all_frames), format='FFMPEG', fps=20, quality=10)

    model.train()

    return all_frames

def validate(model, vert_pos, ref_images, val_loader, valset_args, logger, pts_to_use_list=None, val_cam_noise=0.0):
    model.eval()
    metrics = {}

    lpips_vgg = lpips.LPIPS(net='vgg').cuda()
    # lpips_vgg = None

    if val_cam_noise > 0.0:
        # generate the noise with a fixed random seed, so that we can compare fairly with nerf
        torch.manual_seed(0)
        lie = Lie()
        se3_noise = torch.randn(len(val_loader), 6, device=torch.device('cuda')) * val_cam_noise
        SE3_noise = lie.se3_to_SE3(se3_noise)  # 1 x 3 x 4
        SE3_noise[:, :, 3] = 0.0 # only take the SO3 part, to save some effort on scaling
        SE3_noise = torch.cat([SE3_noise, torch.tensor([0.0, 0.0, 0.0, 1.0], device=torch.device('cuda')).reshape(1, 1, 4).repeat(len(val_loader), 1, 1)], dim=1)

        print(SE3_noise[0])

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


            images, poses, intrinsics, _ = data_blob
            masks = torch.ones_like(images[:, :, 0]) # color mask

            factor = valset_args['factor']
            render_scale = valset_args['render_scale']
            loss_type = valset_args['loss_type']

            images = images.cuda()
            poses = poses.cuda()
            intrinsics = intrinsics.cuda()
            masks = masks.cuda()
            masks = masks.unsqueeze(2)

            rgb_gt = images[:, 0] * 2.0 / 255.0 - 1.0  # range [-1, 1]
            rgb_gt = F.interpolate(rgb_gt, [valset_args["crop_size"][0] // (factor // render_scale), valset_args["crop_size"][1] // (factor // render_scale)], mode='bilinear',
                                   align_corners=True)
            mask_gt = F.interpolate(masks[:, 0], [valset_args["crop_size"][0] // (factor // render_scale), valset_args["crop_size"][1] // (factor // render_scale)], mode='nearest')

            intrinsics_gt = intrinsics[:, 0]
            intrinsics_gt[:, 0] /= factor
            intrinsics_gt[:, 1] /= factor

            target_pose = poses[:, 0]

            if val_cam_noise > 0.0:
                target_pose = target_pose.bmm(SE3_noise[i_batch:i_batch+1])

            # gts, intrinsics_gt = crop_operation(torch.cat([rgb_gt, mask_gt], dim=1), intrinsics_gt, valset_args['resize_h'] // factor, valset_args['resize_w'] // factor, mod='center')
            # rgb_gt, mask_gt = torch.split(gts, [3, 1], dim=1)

            # rgb_est = model(images, poses, intrinsics, depth_low_res)
            # rgb_est = [model(ref_images, poses[:, 0], intrinsics_gt, is_eval=True), ]  # 1 x 3 x H x W
            # rgb_est = [model.evaluate(ref_images, poses[:, 0], intrinsics_gt), ]  # 1 x 3 x H x W

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            rgb_est = [model.evaluate(vert_pos, ref_images, target_pose, intrinsics_gt, num_random_samples=1, pts_to_use_list=pts_to_use_list), ]  # 1 x 3 x H x W
            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()

            print('time for one frame (ms):', start.elapsed_time(end))

            # disp_loss, disp_metrics = sequence_loss(disp_est, disp_gt,
            #                                     depthloss_threshold=args.depthloss_threshold,
            #                                     loss_type=loss_type,
            #                                     weight=weight,
            #                                     gradual_weight=gradual_weight)
            # loss += disp_loss
            # metrics.update(disp_metrics)
            if loss_type == 'l2':
                # normalize to [-1,1]
                for i, im in enumerate(rgb_est):
                    rgb_est[i] = torch.sigmoid(im) * 2.0 - 1.0

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
            # print(rgb_metrics)

            logger.summ_rgb('eval/rgb_gt/%d' % i_batch, rgb_gt, mask_gt, force_save=True)
            logger.summ_rgb('eval/rgb_est/%d' % i_batch, rgb_est[-1], mask_gt, force_save=True)
            logger.summ_diff('eval/l1_diff/%d' % i_batch, rgb_gt, rgb_est[-1], force_save=True)

    # average
    for (k, v) in metrics.items():
        metrics[k] /= len(val_loader)

    logger.write_dict(metrics, 'eval')

    print('finish eval on %d samples' % len(val_loader))
    print(metrics)

    if hasattr(model, 'vert_opy'):
        logger.summ_hist('vert_opy', torch.sigmoid(model.vert_opy), force_save=True)

    model.train()

    return metrics

class PtsUnprojector(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super(PtsUnprojector, self).__init__()
        self.device=device

    def forward(self, depth, pose, intrinsics, mask=None, return_coord=False):
        # take depth and convert into world pts
        # depth: B x 1 x H x W
        # pose: B x 4 x 4
        # intrinsics: B x 3 x 3
        # mask: B x 1 x H x W
        # return coord: return the corresponding [b,y,x] coord for each point, so that we can index into the vertex feature

        B, _, H, W = depth.shape

        # assert(h==self.H)
        # assert(w==self.W)
        xs = torch.linspace(0, W - 1, W).float()
        ys = torch.linspace(0, H - 1, H).float()

        xs = xs.view(1, 1, 1, W).repeat(1, 1, H, 1)
        ys = ys.view(1, 1, H, 1).repeat(1, 1, 1, W)

        xyzs = torch.cat((xs, ys, torch.ones(xs.size())), 1).view(1, 3, -1).to(self.device)  # 1 x 3 x N

        depth = depth.reshape(B, 1, -1)

        projected_coors = xyzs * depth # B x 3 x N

        xyz_source = torch.inverse(intrinsics).bmm(projected_coors)  # B x 3 x N, xyz in cam1 space
        xyz_source = torch.cat((xyz_source, torch.ones_like(xyz_source[:, 0:1])), dim=1) # B x 4 x N

        # pose is cam_T_world
        xyz_world = torch.inverse(pose).bmm(xyz_source) # B x 4 x N
        xyz_world = xyz_world[:, 0:3]  # B x 3 x N, discard homogeneous dimension

        # xyz_world = xyz_world.reshape(B, 3, H, W)
        xyz_world = xyz_world.permute(0, 2, 1).reshape(-1, 3)  # B*N x 3

        if return_coord:
            bs = torch.linspace(0, B-1, B).float()
            bs = bs.view(B, 1, 1).repeat(1, H, W)
            xs = xs.view(1, H, W).repeat(B, 1, 1)
            ys = ys.view(1, H, W).repeat(B, 1, 1)

            buvs = torch.stack((bs, ys, xs), dim=-1).view(-1, 3).to(self.device) # B*N x 3

        # if mask not none, we prune the xyzs by only selecting the valid ones
        if mask is not None:
            mask = mask.reshape(-1)
            nonzeros = torch.where(mask>0.5)[0]
            xyz_world = xyz_world[nonzeros, :] # n_valid x 3
            if return_coord:
                buvs = buvs[nonzeros, :]

        if return_coord:
            return xyz_world, buvs.to(torch.long)
        else:
            return xyz_world

    def get_dists(self, depth, intrinsics, mask=None):
        # take depth and convert into world pts
        # depth: B x 1 x H x W
        # intrinsics: B x 3 x 3
        # mask: B x 1 x H x W
        B, _, H, W = depth.shape

        xs = torch.linspace(0, W - 1, W).float()
        ys = torch.linspace(0, H - 1, H).float()

        xs = xs.view(1, 1, 1, W).repeat(1, 1, H, 1)
        ys = ys.view(1, 1, H, 1).repeat(1, 1, 1, W)

        xyzs = torch.cat((xs, ys, torch.ones(xs.size())), 1).view(1, 3, -1).to(self.device)  # 1 x 3 x N

        depth = depth.reshape(B, 1, -1)

        projected_coors = xyzs * depth  # B x 3 x N

        xyz_source = torch.inverse(intrinsics).bmm(projected_coors)  # B x 3 x N, xyz in cam1 space

        l2_dists =  torch.norm(xyz_source, p=2, dim=1, keepdim=True)  # B x 1 x N
        l2_dists = l2_dists.permute(0, 2, 1).reshape(-1, 1)  # B*N x 1

        if mask is not None:
            mask = mask.reshape(-1)
            nonzeros = torch.where(mask > 0.5)[0]
            l2_dists = l2_dists[nonzeros, :]  # n_valid x 3

        return l2_dists

    def apply_mask(self, feat, mask):
        # feat: B x C x H x W
        # mask: B x 1 x H x W
        B, C, H, W = feat.shape
        feat = feat.reshape(B, C, -1)
        feat = feat.permute(0, 2, 1).reshape(-1, C)  # B*N x C

        mask = mask.reshape(-1)
        nonzeros = torch.where(mask > 0.5)[0]
        feat = feat[nonzeros, :]  # n_valid x C

        return feat

class PulsarSceneModel(nn.Module):
    def __init__(self,
                 n_points,
                 dim_pointfeat=256,
                 radius=7.5e-4,
                 render_size=(300, 400),
                 render_scale=1,
                 world_scale=400.,
                 bkg_col=(0,0,0),
                 gamma=1.0e-3,
                 batch_size=1,
                 do_2d_shading=False,
                 shader_arch='simple_unet',
                 pts_dropout_rate=0.0,
                 basis_type='mlp',
                 shader_output_channel=128,
                 free_opy=1,
                 shader_norm='none',
                 ):
        super(PulsarSceneModel, self).__init__()
        # images: N x 3 x H x W
        # depth_low_res: N x 1 x h x w
        # masks_low_res: N x 1 x h x w

        self.unprojector = PtsUnprojector()

        self.n_points = n_points

        # points_init_loc has shape N x 3
        device = torch.device("cuda")

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

            if shader_arch == 'simple_unet':
                self.shader_2d = SmallUNet(n_channels=self.shader_output_channel, n_classes=3, bilinear=False, norm=shader_norm, render_scale=render_scale)
            elif shader_arch == 'full_unet':
                self.shader_2d = UNet(n_channels=self.shader_output_channel, n_classes=3, bilinear=False, norm=shader_norm)
            elif shader_arch == 'simple':
                self.shader_2d = TwoLayersCNN(n_channels=self.shader_output_channel, n_classes=3, norm=shader_norm)
            else:
                raise NotImplementedError

        else:
            self.shader_output_channel = 3 # override to rgb channel

        self.shader = Shader(feat_dim=dim_pointfeat, rgb_channel=self.shader_output_channel, output_opacity=False, opacity_channel=1, basis_type=basis_type)

        if free_opy:
            self.register_parameter("vert_opy", nn.Parameter(torch.ones(self.n_points), requires_grad=True))
        else:
            self.register_buffer("vert_opy", torch.ones(self.n_points))
        self.free_opy = free_opy

        cameras = FoVOrthographicCameras(R=(torch.eye(3, dtype=torch.float32, device=device)[None, ...]).repeat(batch_size, 1, 1),
                                         T=torch.zeros((batch_size, 3), dtype=torch.float32, device=device),
                                         znear=[1.0]*batch_size,
                                         zfar=[1e5]*batch_size,
                                         device=device,
                                         )

        raster_settings = PointsRasterizationSettings(
            image_size=render_size,
            radius=None,
            max_points_per_bin=50000
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        self.renderer = PulsarPointsRenderer(rasterizer=rasterizer, max_num_spheres=n_points, n_channels=self.shader_output_channel, n_track=100).cuda()

        if self.shader_output_channel==3:
            self.register_buffer('bkg_col', torch.tensor(bkg_col, dtype=torch.float32, device=device))
        else:
            self.register_buffer('bkg_col', torch.randn(self.shader_output_channel, dtype=torch.float32, device=device))

        self.render_size = render_size
        self.gamma = gamma
        self.dim_pointfeat = dim_pointfeat

        self.H, self.W = render_size[0], render_size[1]
        self.radius = radius

        self.render_scale = render_scale
        self.do_2d_shading = do_2d_shading
        self.pts_dropout_rate = pts_dropout_rate
        self.world_scale = world_scale

    def forward(self, vert_pos, ref_images, target_pose, target_intrinsics, cropping_params=None, is_eval=False):
        # note that in this version, the vertex postions need to be assigned in forward pass, to model the articulated object
        # vert_pos: torch.cuda() tensor with shape 3 x N
        # target_pose: B x 4 X 4
        # target_intrinsics: B x 3 x 3

        # if is_eval: turn off random dropout.
        do_random_dropout = ((not is_eval) and (self.pts_dropout_rate > 0.0))
        if do_random_dropout:
            num_pts_to_keep = round(self.n_points * (1.0 - self.pts_dropout_rate))
            pts_id_to_keep = torch.tensor(np.random.choice(np.arange(self.n_points), size=num_pts_to_keep, replace=False)).cuda()

        B = target_pose.shape[0]

        # target_intrinsics[:, 0] *= self.render_scale
        # target_intrinsics[:, 1] *= self.render_scale

        # convert self.vert_pos in world coordinates into cam coordinates
        xyz_world = torch.cat((vert_pos, torch.ones_like(vert_pos[0:1])), dim=0).unsqueeze(0).repeat(B, 1, 1)  # 1 x 4 x N, turned into homogeneous coord

        # tagget_pose is cam_T_world
        xyz_target = target_pose.bmm(xyz_world)
        xyz_target = xyz_target[:, 0:3]  # B x 3 x N, discard homogeneous dimension

        # xyz_target[:, 1:3] = -xyz_target[:, 1:3] # adjust coord here, flip y and z

        xy_proj = target_intrinsics.bmm(xyz_target) # B x 3 x N

        eps_mask = (xy_proj[:, 2:3, :].abs() < EPS).detach()

        # Remove invalid zs that cause nans
        zs = xy_proj[:, 2:3, :]
        zs[eps_mask] = EPS


        sampler = torch.cat((xy_proj[:, 0:2, :] / zs, xy_proj[:, 2:3, :]), 1)  # u, v, has range [0,W], [0,H] respectively
        sampler[eps_mask.repeat(1, 3, 1)] = -1e6

        # compute the radius based on the distance of the points to the reference view camera center
        scale_pts = torch.norm(sampler, dim=1) / self.world_scale
        radius = torch.ones_like(scale_pts) * self.radius  # B x N
        # radius = scale_pts * self.radius # B x N
        # radius = torch.ones_like(sampler[:, 0]) * self.radius

        if do_random_dropout:
            radius = radius[:, pts_id_to_keep]

        # normlaize to NDC space. flip xy because the ndc coord difinition
        sampler[:, 0, :] = -((sampler[:, 0, :] / self.H) * 2. - (self.W / self.H))
        sampler[:, 1, :] = -((sampler[:, 1, :] / self.H) * 2. - 1.)

        # sampler: B x 3 x num_pts
        xyz_ndc = sampler.permute(0, 2, 1).contiguous() # B x N x 3


        if do_random_dropout:
            xyz_ndc = xyz_ndc[:, pts_id_to_keep] # B x N_drop x 3

        pointfeat = self.vert_feat # N x feat_dim

        if do_random_dropout:
            pointfeat = pointfeat[pts_id_to_keep]

        points_feature_flatten = pointfeat.unsqueeze(0).repeat(B, 1, 1).reshape(-1, self.dim_pointfeat)  # (B*N_pts) x feat_dim

        # view_dir = get_view_dir_world(target_pose, self.z_dir)
        view_dir = get_view_dir_world_per_ray(target_pose, xyz_target) # B x 3 x N
        if do_random_dropout:
            view_dir = view_dir[:, :, pts_id_to_keep]

        view_dir = view_dir.permute(0, 2, 1).reshape(-1, 3)  # (B*N) x 3

        shaded_feature = self.shader(points_feature_flatten, view_dir)  # (B*N) x 3
        shaded_feature = shaded_feature.reshape(B, xyz_ndc.shape[1], self.shader_output_channel)
        shaded_feature = shaded_feature.permute(0, 2, 1).contiguous() # B x 3 x N

        # do rendering
        assert xyz_ndc.size(2) == 3
        assert xyz_ndc.size(1) == shaded_feature.size(2)


        xyz_ndc[..., 0:2] *= (float(self.render_size[0]) / float(self.render_size[1]))
        pts3D = Pointclouds(points=xyz_ndc, features=shaded_feature.permute(0, 2, 1))

        if self.free_opy:
            opacity = torch.sigmoid(self.vert_opy.unsqueeze(0).repeat(B, 1))
        else:
            opacity = self.vert_opy.unsqueeze(0).repeat(B, 1)

        if do_random_dropout:
            opacity = opacity[:, pts_id_to_keep]

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

        if cropping_params is not None:
            # crop before the shader2d
            x0, y0, x1, y1 = cropping_params
            pred = pred[:, :, y0:y1, x0:x1]

        if self.do_2d_shading: # do post-processing
            pred = self.shader_2d(pred)

        return pred

    def evaluate(self, vert_pos, ref_images, target_pose, target_intrinsics, num_random_samples=5, pts_to_use_list=None, fix_seed=False, target_viewpose=None):
        # target_pose: B x 4 X 4
        # target_intrinsics: B x 3 x 3
        if target_viewpose is None:
            target_viewpose = target_pose

        do_random_dropout = (self.pts_dropout_rate > 0.0)
        if not do_random_dropout:
            num_random_samples = 1

        vert_feat = self.vert_feat
        vert_opy = self.vert_opy

        if pts_to_use_list is not None:
            vert_pos = vert_pos[:, pts_to_use_list]
            vert_feat = vert_feat[pts_to_use_list, :]
            vert_opy = vert_opy[pts_to_use_list]

        num_pts_to_keep = round(vert_pos.shape[1] * (1.0 - self.pts_dropout_rate))

        B = target_pose.shape[0]

        # target_intrinsics[:, 0] *= self.render_scale
        # target_intrinsics[:, 1] *= self.render_scale

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
        radius = torch.ones_like(scale_pts) * self.radius # B x N

        # normlaize to NDC space. flip xy because the ndc coord difinition
        sampler[:, 0, :] = -((sampler[:, 0, :] / self.H) * 2. - (self.W / self.H))
        sampler[:, 1, :] = -((sampler[:, 1, :] / self.H) * 2. - 1.)

        # sampler: B x 3 x num_pts
        xyz_ndc = sampler.permute(0, 2, 1).contiguous() # B x N x 3

        pointfeat = vert_feat # N x feat_dim

        points_feature_flatten = pointfeat.unsqueeze(0).repeat(B, 1, 1).reshape(-1, self.dim_pointfeat)  # (B*N_pts) x feat_dim

        view_dir = get_view_dir_world_per_ray(target_viewpose, xyz_target)
        view_dir = view_dir.permute(0, 2, 1).reshape(-1, 3)  # (B*N) x 3
        # view_dir = get_view_dir_world(target_pose, self.z_dir)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        shaded_feature = self.shader(points_feature_flatten, view_dir)  # (B*N) x 3
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        # print('shading time:', start.elapsed_time(end))

        shaded_feature = shaded_feature.reshape(B, xyz_ndc.shape[1], self.shader_output_channel)
        shaded_feature = shaded_feature.permute(0, 2, 1).contiguous() # B x 3 x N


        # do rendering
        assert xyz_ndc.size(2) == 3
        assert xyz_ndc.size(1) == shaded_feature.size(2)


        xyz_ndc[..., 0:2] *= (float(self.render_size[0]) / float(self.render_size[1]))

        if self.free_opy:
            opacity = torch.sigmoid(vert_opy.unsqueeze(0).repeat(B, 1))
        else:
            opacity = vert_opy.unsqueeze(0).repeat(B, 1)

        all_preds = None

        for _ in range(num_random_samples):
            if fix_seed:
                torch.manual_seed(0)
                np.random.seed(0)
            pts_id_to_keep = torch.tensor(np.random.choice(np.arange(xyz_ndc.shape[1]), size=num_pts_to_keep, replace=False)).cuda()
            xyz_ndc_sampled = xyz_ndc[:, pts_id_to_keep]  # B x N_drop x 3
            shaded_feature_sampled = shaded_feature[..., pts_id_to_keep]
            shaded_opy_sampled = opacity[:, pts_id_to_keep]
            radius_sampled = radius[:, pts_id_to_keep]


            pts3D = Pointclouds(points=xyz_ndc_sampled, features=shaded_feature_sampled.permute(0, 2, 1))

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

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

            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()

            # print('rasterization time:', start.elapsed_time(end))


            # vert_valid = torch.sigmoid(self.vert_opy) > torch.mean(torch.sigmoid(self.vert_opy))
            # frame_utils.save_ply('./test.ply', (self.vert_pos.permute(1,0)), ((shaded_feature[0]).permute(1,0)))

            if all_preds is None:
                all_preds = pred
            else:
                all_preds += pred

        pred = all_preds / float(num_random_samples)

        if self.do_2d_shading: # do post-processing
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            pred = self.shader_2d(pred)
            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()

            # print('render time:', start.elapsed_time(end))

        return pred

class Logger:
    def __init__(self, model, scheduler, output, SUM_FREQ, img_log_freq=100, tb_logdir=None):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=tb_logdir)
        self.output = output
        self.SUM_FREQ = SUM_FREQ
        self.img_log_freq = img_log_freq

    def set_global_step(self, global_step):
        self.total_steps = global_step

    def _print_training_status(self):
        SUM_FREQ = self.SUM_FREQ
        metrics_data = [self.running_loss[k] / SUM_FREQ for k in sorted(self.running_loss.keys())]
        # training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        training_str = "[{:6d}]".format(self.total_steps + 1)
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)
        if not self.output is None:
            f = open(self.output, "a")
            f.write(f"{training_str + metrics_str}\n")
            f.close()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics, prefix=None):
        SUM_FREQ = self.SUM_FREQ

        for (key, v) in metrics.items():
            if prefix is not None:
                key = '%s/%s' % (prefix, key)

            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += v

        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results, prefix=None):
        for (key, v) in results.items():
            if prefix is not None:
                key = '%s/%s' % (prefix, key)

            self.writer.add_scalar(key, v, self.total_steps)

    def summ_rgb(self, tag, rgb, mask=None, bgr2rgb=True, force_save=False):
        # rgb should have shape B x 3 x H x W, and be in range [-0.5, 0.5]
        if force_save or self.total_steps % self.img_log_freq == self.img_log_freq - 1:
            rgb = (rgb + 1.0) / 2.0
            rgb = torch.clamp(rgb, 0.0, 1.0)
            if bgr2rgb:
                rgb = rgb[:, [2, 1, 0]]
            if mask is not None:
                rgb = rgb * mask + 1.0 * (1.0 - mask)  # make bkg white
            self.writer.add_image(tag, rgb[0], self.total_steps)
            return True
        else:
            return False

    def summ_rgbs(self, tag, rgbs, fps=10, bgr2rgb=True, force_save=False):
        # rgbs should have shape N x 3 x H x W, and be in range [-1, 1]
        if force_save or self.total_steps % self.img_log_freq == self.img_log_freq - 1:
            rgbs = (rgbs + 1.0) / 2.0
            rgbs = torch.clamp(rgbs, 0.0, 1.0)
            if bgr2rgb:
                rgbs = rgbs[:, [2, 1, 0]]
            self.writer.add_video(tag, rgbs.unsqueeze(0), self.total_steps, fps=fps)
            return True
        else:
            return False

    def summ_oned(self, tag, img, force_save=False):
        # make sure the img has range [0,1]
        if force_save or self.total_steps % self.img_log_freq == self.img_log_freq - 1:
            img = torch.clamp(img, 0.0, 1.0)
            self.writer.add_image(tag, img[0, 0], self.total_steps, dataformats='HW')
            return True
        else:
            return False

    def summ_diff(self, tag, im1, im2, vmin=0, vmax=100, force_save=False):
        if force_save or self.total_steps % self.img_log_freq == self.img_log_freq - 1:
            im1 = (im1[0]).permute(1, 2, 0).cpu().numpy() # H x W x 3
            im2 = (im2[0]).permute(1, 2, 0).cpu().numpy()

            im1 = (im1 + 1.0) / 2.0 * 255.0
            im2 = (im2 + 1.0) / 2.0 * 255.0
            vis = frame_utils.grayscale_visualization(np.mean(np.abs((im1 - im2)), axis=2), 'L1 diff', vmin=vmin, vmax=vmax)

            self.writer.add_image(tag, vis, self.total_steps, dataformats='HWC')
            return True

        else:
            return False

    def summ_hist(self, tag, tensor, force_save=False):
        if force_save or self.total_steps % self.img_log_freq == self.img_log_freq - 1:
            self.writer.add_histogram(tag, tensor, self.total_steps)
            return True
        else:
            return False

    def close(self):
        self.writer.close()

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

    d["testing_dir"] = os.path.join(d["testing_dir"], args.single)

    # HR = params["HR"]
    # factor = 8 if not HR else 4
    factor = 2 # downsample from 800x800 to 400x400

    unprojector = PtsUnprojector()

    render_scale = args.render_scale

    # factor = int(factor / args.render_scale)


    # extract pts for all views
    gpuargs = {'num_workers': 0, 'drop_last': False, 'shuffle': False}

    datasetname = d["dataset"]
    trainset_args = {
                     "crop_size": [args.crop_h, args.crop_w],
                     "resize": [args.crop_h, args.crop_w],
                     "split": 'train',
                     "return_frame_ids": True
                     }

    valset_args = trainset_args.copy()
    valset_args['split'] = 'test'

    train_dataset = SYNViewsynTrain(d["testing_dir"], args.pointcloud_dir, **trainset_args)
    val_dataset = SYNViewsynTrain(d["testing_dir"], args.pointcloud_dir, **valset_args)

    valset_args['factor'] = factor
    valset_args['render_scale'] = render_scale
    valset_args['loss_type'] = args.fe_loss_type


    train_loader = DataLoader(train_dataset, batch_size=1, **gpuargs)
    val_loader = DataLoader(val_dataset, batch_size=1, **gpuargs)

    trainset_images = []
    trainset_loss_masks = []
    trainset_poses = []
    trainset_intrinsics = []
    trainset_frame_ids = []
    # trainset_xyzs = []

    # put all training samples into memory
    for i_batch, data_blob in enumerate(train_loader):

        images, poses, intrinsics, frame_id = data_blob
        loss_masks = torch.ones_like(images[:, :, 0]) # everything

        images = images.cuda()
        poses = poses.cuda()
        intrinsics = intrinsics.cuda()
        loss_masks = loss_masks.cuda()

        loss_masks = loss_masks.unsqueeze(2)

        rgb_gt = images[:, 0] * 2.0 / 255.0 - 1.0  # range [-1, 1]
        rgb_gt = F.interpolate(rgb_gt, [params['crop_h'] // (factor // render_scale), params['crop_w'] // (factor // render_scale)], mode='bilinear', align_corners=True)
        loss_mask_gt = F.interpolate(loss_masks[:, 0], [params['crop_h'] // (factor // render_scale), params['crop_w'] // (factor // render_scale)], mode='nearest') # B x 1 x H x W

        intrinsics_gt = intrinsics[:, 0] # B x 4 x 4
        intrinsics_gt[:, 0] /= factor
        intrinsics_gt[:, 1] /= factor

        trainset_images.append(rgb_gt)
        trainset_loss_masks.append(loss_mask_gt)
        trainset_poses.append(poses[:, 0])
        trainset_intrinsics.append(intrinsics_gt)
        trainset_frame_ids.append(frame_id[:, 0])

    # stack
    trainset_images = torch.cat(trainset_images, dim=0)
    trainset_loss_masks = torch.cat(trainset_loss_masks, dim=0)
    trainset_poses = torch.cat(trainset_poses, dim=0)
    trainset_intrinsics = torch.cat(trainset_intrinsics, dim=0)
    trainset_frame_ids = torch.cat(trainset_frame_ids, dim=0)

    # print(trainset_frame_ids)

    # TODO: load pointclouds from ply file
    trainset_xyzs = train_dataset.get_pointclouds()
    trainset_xyzs = trainset_xyzs.cuda()

    max_num_pts = args.max_num_pts

    if trainset_xyzs.shape[-1] > max_num_pts:
        print('original num pts: %d' % trainset_xyzs.shape[-1])
        print('warning: random dropping points to %d due to memory limit...' % max_num_pts)
        vert_id_to_keep = np.random.choice(np.arange(trainset_xyzs.shape[-1]), size=max_num_pts, replace=False)
        trainset_xyzs = trainset_xyzs[..., vert_id_to_keep]

    # vert_pos = trainset_xyzs[0] # shape 3 x n_points
    n_points = trainset_xyzs.shape[-1]

    print('total points we gonna use: %d' % n_points)

    model = PulsarSceneModel(n_points=n_points, dim_pointfeat=args.dim_pointfeat,
                             render_size=(params['crop_h'] // (factor), params['crop_w'] // (factor)),
                             render_scale=render_scale, gamma=args.blend_gamma, batch_size=args.batch_size, radius=args.sphere_radius,
                             bkg_col=(1, 1, 1), # syn data has white bkg
                             do_2d_shading=args.do_2d_shading, shader_arch=args.shader_arch, pts_dropout_rate=args.pts_dropout_rate,
                             basis_type=args.basis_type, shader_output_channel=args.shader_output_channel, free_opy=args.free_opy, shader_norm=args.shader_norm).cuda()

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

    if args.eval_only or args.anim_only:
        assert (args.restore_ckpt is not None)
        # this gives worse performance now, so turn it off.
        # pts_to_use_list = check_pts_gradient(model, trainset_images, train_loader, valset_args, logger)
        pts_to_use_list=None
        if args.anim_only:
            make_animation_simple(model, trainset_xyzs[0], val_dataset, trainset_intrinsics[0:1], logger)
        if args.eval_only:
            validate(model, trainset_xyzs[0], trainset_images, val_loader, valset_args, logger, pts_to_use_list, val_cam_noise=args.val_cam_noise)
        logger.close()
        return

    VAL_FREQ = args.VAL_FREQ

    tic = None
    total_time = 0

    for total_steps in range(1, args.num_steps+1):

        optimizer.zero_grad()

        # train_id = np.random.randint(len(trainset_images))
        # train_id = np.array([0, ]) # debug

        train_id = np.random.choice(np.arange(len(trainset_images)), args.batch_size, replace=False)

        rgb_gt = trainset_images[train_id] # b x 3 x H x W
        target_pose = trainset_poses[train_id] # b x 4 x 4
        target_intrinsics = trainset_intrinsics[train_id] # b x 3 x 3
        mask_gt = trainset_loss_masks[train_id] # b x 1 x H x W
        vert_pos = trainset_xyzs[(trainset_frame_ids[train_id])[0]]

        # vert_pos = trainset_xyzs[(trainset_frame_ids[train_id] // 2)[0]] # don't do this! just for debug

        # # random crop the target
        # # get cropping params in format (x0, y0, x1, y1)
        # x0 = np.random.randint(0, params['crop_w']//factor - params['resize_w']//factor + 1)
        # y0 = np.random.randint(0, params['crop_h']//factor - params['resize_h']//factor + 1)
        # x1 = x0 + params['resize_w']//factor
        # y1 = y0 + params['resize_h']//factor
        #
        # mask_gt = mask_gt[:, :, y0:y1, x0:x1]
        # rgb_gt = rgb_gt[:, :, y0:y1, x0:x1]
        #
        # cropping_params = (x0, y0, x1, y1)
        #
        # # random crop the reference
        # ref_inputs = torch.cat([trainset_images, trainset_depth_masks, trainset_depths], dim=1)
        # ref_inputs, ref_intrinsics = crop_operation(ref_inputs, trainset_intrinsics, params['resize_h']//factor, params['resize_w']//factor)
        # ref_images, ref_masks, ref_depths = torch.split(ref_inputs, [3,1,1], dim=1)
        # ref_poses = trainset_poses

        ref_images = trainset_images

        # do the rendering
        rgb_est = model(vert_pos, ref_images, target_pose, target_intrinsics, cropping_params=None)  # b x 3 x H x W

        loss_type = args.fe_loss_type
        if loss_type == 'l2':
            rgb_est = torch.sigmoid(rgb_est) * 2.0 - 1.0

        loss, metrics = sequence_loss_rgb([rgb_est,], rgb_gt, mask_gt, loss_type=loss_type)
        # rgb_gt = rgb_gt * mask_gt + 1.0 * (1.0 - mask_gt)
        # loss, metrics = sequence_loss_rgb([rgb_est, ], rgb_gt, loss_type=loss_type)

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


        if total_steps % VAL_FREQ == VAL_FREQ - 1:
            validate(model, vert_pos, trainset_images, val_loader, valset_args, logger)
            PATH = 'checkpoints/%d_%s.pth' % (total_steps + 1, args.name)

        logger.set_global_step(total_steps)

        if not tic is None:
            total_time += time.time() - tic
            print(
                f"time per step: {total_time / (total_steps - 1)}, expected: {total_time / (total_steps - 1) * args.num_steps / 24 / 3600} days")
            print(args.name)
        tic = time.time()

    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    validate(model, vert_pos, trainset_images, val_loader, valset_args, logger)

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
    parser.add_argument('--anim_only', type=int, default=False)

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
    parser.add_argument('--fe_loss_type', type=str, default='l2')

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
    parser.add_argument('--basis_type', type=str, default='mlp', help="the basis type to use for modeling the non-Lambertian effect. option: mlp;SH")
    parser.add_argument('--shader_output_channel', type=int, default=128)
    parser.add_argument('--pts_dropout_rate', type=float, default=0.0)
    parser.add_argument('--dim_pointfeat', type=int, default=16)
    parser.add_argument('--do_xyz_pos_encode', type=int, default=False)
    parser.add_argument('--plyfilename', type=str, default='')
    parser.add_argument('--restore_pointclouds', type=str, default=None)
    parser.add_argument('--anim_type', type=str, default='deformation')
    parser.add_argument('--max_num_pts', type=int, default=500000)
    parser.add_argument('--shader_norm', type=str, default='none')
    parser.add_argument('--val_cam_noise', type=float, default=0.0,
                        help='add noise to the cam pose during eval. for debug only')
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