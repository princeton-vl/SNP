import numpy as np
import glob
import os
import cv2
import json
import imageio

# import matplotlib.pyplot as plt
import frame_utils

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from collections import OrderedDict


# training_set = [2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
#                     45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
#                     74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
#                     101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
#                     121, 122, 123, 124, 125, 126, 127, 128]

def scale_operation(images, intrinsics, s):
    ht1 = images.shape[2]
    wd1 = images.shape[3]
    ht2 = int(s * ht1)
    wd2 = int(s * wd1)
    intrinsics[:, 0] *= s
    intrinsics[:, 1] *= s
    images = F.interpolate(images, [ht2, wd2], mode='bilinear', align_corners=True)
    return images, intrinsics


def crop_operation(images, intrinsics, crop_h, crop_w):
    ht1 = images.shape[2]
    wd1 = images.shape[3]
    x0 = (wd1 - crop_w) // 2
    y0 = (ht1 - crop_h) // 2
    x1 = x0 + crop_w
    y1 = y0 + crop_h
    images = images[:, :, y0:y1, x0:x1]
    intrinsics[:, 0, 2] -= x0
    intrinsics[:, 1, 2] -= y0
    return images, intrinsics


def random_scale_and_crop(images, masks, intrinsics, depths=None, resize=[-1, -1], crop_size=[448, 576]):
    s = 2 ** np.random.uniform(-0.1, 0.4)

    ht1 = images.shape[2]
    wd1 = images.shape[3]
    if resize == [-1, -1]:
        ht2 = int(s * ht1)
        wd2 = int(s * wd1)
    else:
        ht2 = int(resize[0])
        wd2 = int(resize[1])

    intrinsics[:, 0] *= float(wd2) / wd1
    intrinsics[:, 1] *= float(ht2) / ht1

    if depths is not None:
        depths = depths.unsqueeze(1)
        depths = F.interpolate(depths, [ht2, wd2], mode='nearest')

    images = F.interpolate(images, [ht2, wd2], mode='bilinear', align_corners=True)

    x0 = np.random.randint(0, wd2 - crop_size[1] + 1)
    y0 = np.random.randint(0, ht2 - crop_size[0] + 1)
    x1 = x0 + crop_size[1]
    y1 = y0 + crop_size[0]

    images = images[:, :, y0:y1, x0:x1]

    if depths is not None:
        depths = depths[:, :, y0:y1, x0:x1]
        depths = depths.squeeze(1)

    intrinsics[:, 0, 2] -= x0
    intrinsics[:, 1, 2] -= y0

    if masks is not None:
        masks = masks.unsqueeze(1)
        masks = F.interpolate(masks, [ht2, wd2], mode='nearest')
        masks = masks[:, :, y0:y1, x0:x1]
        masks = masks.squeeze(1)

    return images, depths, masks, intrinsics


def load_pair(file: str):
    with open(file) as f:
        lines = f.readlines()
    n_cam = int(lines[0])
    pairs = {}
    img_ids = []
    for i in range(1, 1 + 2 * n_cam, 2):
        pair = []
        score = []
        img_id = lines[i].strip()
        pair_str = lines[i + 1].strip().split(' ')
        n_pair = int(pair_str[0])
        for j in range(1, 1 + 2 * n_pair, 2):
            pair.append(pair_str[j])
            score.append(float(pair_str[j + 1]))
        img_ids.append(img_id)
        pairs[img_id] = {'id': img_id, 'index': i // 2, 'pair': pair, 'score': score}
    pairs['id_list'] = img_ids
    return pairs


class SYNViewsynTrain(Dataset):
    def __init__(self, dataset_path, pointcloud_path, split='train', crop_size=[800, 800], resize=[-1, -1], return_frame_ids=False, data_augmentation=False, start=0, end=9999):
        self.dataset_path = dataset_path
        self.pointcloud_path = pointcloud_path
        self.split = split

        self.crop_size = crop_size
        self.resize = resize

        self.data_augmentation = data_augmentation
        self.return_frame_ids = return_frame_ids

        self._build_dataset_index()
        self._load_and_rescale_points()

        self.start = start
        self.end = end

    def _load_and_rescale_points(self):
        # get the orginal depth file for scaling
        # first load pointclouds
        # pointcloud_list = [os.path.join(self.dataset_path, "pointclouds", "%s.ply" % os.path.basename(self.dataset_path)), ]
        # pointcloud_list = [os.path.join(self.dataset_path, "pointclouds", "%s_v1.ply" % os.path.basename(self.dataset_path)), ]
        pointcloud_list = [os.path.join(self.pointcloud_path, "%s_v1.ply" % os.path.basename(self.dataset_path)), ]
        pointcloud_factor = 150.0 # ad-hoc

        self.depth_scale = 400. # this may need change later

        all_pointclouds = []

        for file_name in pointcloud_list:
            # pointcloud = np.load(file_name) # np array of shape N x 3
            pointcloud = frame_utils.load_ply(file_name)  # np array of shape N x 3
            pointcloud = np.transpose(pointcloud) # 3 x N

            # this is no longer needed, as we have adjusted the orientation when saving the raw data.
            # # adjust the world coord
            # T = np.array(
            #         ((1, 0, 0),
            #          (0, 0, -1),
            #          (0, 1, 0)))
            #
            # pointcloud = T @ pointcloud
            # pointcloud = pointcloud.astype(np.float32)
            
            all_pointclouds.append(pointcloud)

        all_pointclouds = np.stack(all_pointclouds, axis=0) # n_poses x 3 x N

        # do the scaling
        all_pointclouds *= self.depth_scale
        all_pointclouds /= pointcloud_factor
        self.poses[:, :3, 3] = self.poses[:, :3, 3] * self.depth_scale

        self.all_pointclouds = all_pointclouds

    def _build_dataset_index(self):
        splits = [self.split, ]

        testskip = 1 # same as nerf

        metas = {}
        for s in splits:
            with open(os.path.join(self.dataset_path, 'transforms_{}.json'.format(s)), 'r') as fp:
                metas[s] = json.load(fp)

        all_imgs = []
        all_poses = []
        all_frame_ids = []

        for s in splits:
            meta = metas[s]
            if s == 'train' or testskip == 0:
                skip = 1
            else:
                skip = testskip

            for frame in meta['frames'][::skip]:
                fname = os.path.join(self.dataset_path, frame['file_path'] + '.png')
                # img = frame_utils.read_gen(fname)
                # img = imageio.imread(fname)
                # img = img[..., [2,1,0]] # to cv2 format
                img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
                img = np.where(img[..., 3:]==0, 255*np.ones_like(img), img)
                img = img[..., 0:3]

                c2w = np.array(frame['transform_matrix'])

                rotation, location = c2w[0:3, 0:3], c2w[0:3, 3]
                R_world2bcam = np.transpose(rotation)

                # Convert camera location to translation vector used in coordinate changes
                # T_world2bcam = -1*R_world2bcam*cam.location
                # Use location from matrix_world to account for constraints:
                T_world2bcam = -1 * np.dot(R_world2bcam, location)

                R_bcam2cv = np.array(
                    ((1, 0, 0),
                     (0, -1, 0),
                     (0, 0, -1)))

                # Build the coordinate transform matrix from world to computer vision camera
                # NOTE: Use * instead of @ here for older versions of Blender
                # TODO: detect Blender version
                R_world2cv = np.dot(R_bcam2cv, R_world2bcam)
                T_world2cv = np.dot(R_bcam2cv, T_world2bcam)

                # put into 3x4 matrix
                RT = np.concatenate((R_world2cv, T_world2cv[:, None]), axis=1)
                RT = np.concatenate((RT, np.array([0, 0, 0, 1])[None, :]), axis=0)

                # w2c = np.linalg.inv(c2w)
                w2c = RT

                # print(np.array(frame['transform_matrix']))
                #
                # c2w = np.array(frame['transform_matrix'])
                # c2w[0:3, 0:3] = c2w[0:3, 0:3] @ np.array([[1., 0, 0],[0, -1, 0],[0, 0, -1]])
                # print(c2w)
                #
                # c2w = np.array(frame['transform_matrix']) @ np.array([[1., 0, 0, 0],[0, -1, 0, 0],[0, 0, -1, 0],[0, 0, 0, 1]])
                # print(c2w)
                #
                # assert (False)
                # w2c = np.linalg.inv(c2w)


                # w2c = np.array([[1., 0, 0, 0],[0, -1, 0, 0],[0, 0, -1, 0],[0, 0, 0, 1]]) @ w2c
                # w2c = w2c @ np.array([[1., 0, 0, 0],[0, -1, 0, 0],[0, 0, -1, 0],[0, 0, 0, 1]])

                all_imgs.append(img)
                all_poses.append(w2c)

                if 'frame_id' in frame:
                    all_frame_ids.append(frame['frame_id'])
                else:
                    all_frame_ids.append(0)

        self.images = np.stack(all_imgs, 0).astype(np.float32)  # N x H x W x 3
        self.poses = np.stack(all_poses, 0).astype(np.float32)
        self.all_frame_ids = np.array(all_frame_ids).astype(np.int64)

        H, W = self.images[0].shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)

        self.total_num_views = len(self.images)

        # get the intrinsics
        K = np.array([[focal, 0, H/2],[0, focal, W/2],[0, 0, 1]]).reshape(1, 3, 3)
        K = np.repeat(K, self.total_num_views, axis=0)

        self.intrinsics = K

        print('Dataset length:', self.total_num_views)

    def __len__(self):
        return self.total_num_views

    def __getitem__(self, ix1):
        if ix1 < self.start or ix1 >= self.end: return []
        # randomly sample neighboring frame

        indices = [ix1, ]

        images, poses, intrinsics, frame_ids = [], [], [], []
        for i in indices:
            # image = frame_utils.read_gen(self.image_list[i])
            # depth = frame_utils.read_gen(self.depth_list[i])
            image = self.images[i]
            pose = self.poses[i]
            calib = self.intrinsics[i].copy()
            frame_id = self.all_frame_ids[i]

            images.append(image)
            poses.append(pose)
            intrinsics.append(calib)
            frame_ids.append(frame_id)

        images = np.stack(images, 0).astype(np.float32)  # N x H x W x 3
        poses = np.stack(poses, 0).astype(np.float32)
        intrinsics = np.stack(intrinsics, 0).astype(np.float32)
        frame_ids = np.array(frame_ids).astype(np.int64)

        images = torch.from_numpy(images)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)
        frame_ids = torch.from_numpy(frame_ids)

        # channels first
        images = images.permute(0, 3, 1, 2)  # N x 3 x H x W
        images = images.contiguous()

        if self.data_augmentation:
            images, depths, _, intrinsics = \
                random_scale_and_crop(images, None, intrinsics, None, self.resize, self.crop_size)

        # for op, param in self.size_operations:
        #     if op == "scale":
        #         images, intrinsics = scale_operation(images, intrinsics, param)
        #     elif op == "crop":
        #         images, intrinsics = crop_operation(images, intrinsics, *param)

        if self.return_frame_ids:
            return images, poses, intrinsics, frame_ids
        else:
            return images, poses, intrinsics

    def get_pointclouds(self):
        return torch.from_numpy(self.all_pointclouds)

    def get_render_poses(self):
        splits = ['test']

        metas = {}
        for s in splits:
            with open(os.path.join(self.dataset_path, 'transforms_{}.json'.format(s)), 'r') as fp:
                metas[s] = json.load(fp)

        all_poses = []

        for s in splits:
            meta = metas[s]
            for frame in meta['frames']:
                c2w = np.array(frame['transform_matrix'])

                rotation, location = c2w[0:3, 0:3], c2w[0:3, 3]
                R_world2bcam = np.transpose(rotation)

                # Convert camera location to translation vector used in coordinate changes
                # T_world2bcam = -1*R_world2bcam*cam.location
                # Use location from matrix_world to account for constraints:
                T_world2bcam = -1 * np.dot(R_world2bcam, location)

                R_bcam2cv = np.array(
                    ((1, 0, 0),
                     (0, -1, 0),
                     (0, 0, -1)))

                # Build the coordinate transform matrix from world to computer vision camera
                # NOTE: Use * instead of @ here for older versions of Blender
                # TODO: detect Blender version
                R_world2cv = np.dot(R_bcam2cv, R_world2bcam)
                T_world2cv = np.dot(R_bcam2cv, T_world2bcam)

                # put into 3x4 matrix
                RT = np.concatenate((R_world2cv, T_world2cv[:, None]), axis=1)
                RT = np.concatenate((RT, np.array([0, 0, 0, 1])[None, :]), axis=0)

                # w2c = np.linalg.inv(c2w)
                w2c = RT

                all_poses.append(w2c)

        render_poses = np.stack(all_poses, 0).astype(np.float32)
        render_poses[:, :3, 3] = render_poses[:, :3, 3] * self.depth_scale

        return torch.tensor(render_poses).float().cuda()

