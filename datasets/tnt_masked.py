import numpy as np
import os
import frame_utils
import torch
from torch.utils.data import Dataset
import glob
import torch.nn.functional as F

training_set = ['Barn', 'Truck', 'Caterpillar', "Ignatius", 'Meetingroom', 'Church', 'Courthouse']
intermediate_set = ['Family', 'Francis', 'Horse', 'Lighthouse', 'M60', 'Panther', 'Playground', 'Train']
advanced_set = ["Auditorium", "Ballroom", "Courtroom", "Museum", "Palace", "Temple"]


def scale(images, intrinsics, resize=[-1, -1]):
    ht1 = images.shape[2]
    wd1 = images.shape[3]

    ht2 = int(resize[0])
    wd2 = int(resize[1])

    intrinsics[:, 0] *= float(wd2) / wd1
    intrinsics[:, 1] *= float(ht2) / ht1

    images = F.interpolate(images, [ht2, wd2], mode='bilinear', align_corners=True)

    return images, intrinsics

class TNTMaskedViewsynTrain(Dataset):
    def __init__(self, dataset_path, pointcloud_path, split='train', scan=None, resize=[-1, -1]):
        self.scan = scan
        self.root_path = dataset_path
        self.pointcloud_path = pointcloud_path

        print(split)

        if scan in training_set:
            self.dataset_path = f"{dataset_path}/training_input/{scan}"
        elif scan in intermediate_set:
            self.dataset_path = f"{dataset_path}/tankandtemples/intermediate/{scan}"
        else:
            self.dataset_path = f"{dataset_path}/tankandtemples/advanced/{scan}"

        if split == "train":
            split_fn = f"{dataset_path}/processed_masks/{scan}/train_list.txt"
        else:
            split_fn = f"{dataset_path}/processed_masks/{scan}/test_list.txt"

        f = open(split_fn, "r")
        data = f.read()
        source_views = data.split("\n")
        source_views = [int(s) for s in source_views if s != '']

        self.source_views = source_views
        # self.dataset_path = f"{dataset_path}/{scan}"
        cams_path0 = os.path.join(self.dataset_path, "cams", "%08d_cam.txt" % 0)
        scale_info = np.loadtxt(cams_path0, skiprows=11, dtype=np.float)
        self.scale = 400 / scale_info[0]
        # self.num_frames = num_frames + 1
        # self.pair_list = load_pair(os.path.join(self.dataset_path, 'pair.txt'))
        self._build_dataset_index()
        self._load_and_rescale_points()
        self.resize = resize

        print('Dataset length:', len(self.source_views))

    def _build_dataset_index(self):
        self.all_img_paths = sorted(glob.glob(os.path.join(self.dataset_path, "images", "*.jpg")))
        self.total_num_views = len(self.all_img_paths)


    def __len__(self):
        return len(self.source_views)

    def _load_and_rescale_points(self):
        file_name = os.path.join(self.pointcloud_path, "%s_subsampled.ply" % self.scan)
        pointcloud = frame_utils.load_ply(file_name)  # np array of shape N x 3
        pointcloud = np.transpose(pointcloud)  # 3 x N

        # do the scaling
        pointcloud *= self.scale
        # self.poses[:, :3, 3] = self.poses[:, :3, 3] * self.depth_scale

        self.pointcloud = pointcloud # 3 x N

    def __getitem__(self, index):
        images, poses, intrinsics = [], [], []
        i = self.source_views[index]
        image = frame_utils.read_gen(os.path.join(self.dataset_path, "images", "%08d.jpg" % i))

        # apply the mask
        mask = frame_utils.read_gen(os.path.join(self.root_path, "processed_masks", self.scan, "masks", "%08d.png" % i))
        mask = mask / 255.0
        image = image * mask + 255 * np.ones_like(image) * (1. - mask)

        cams_path = os.path.join(self.dataset_path, "cams", "%08d_cam.txt" % i)
        pose = np.loadtxt(cams_path, skiprows=1, max_rows=4, dtype=np.float)
        calib = np.loadtxt(cams_path, skiprows=7, max_rows=3, dtype=np.float)
        images.append(image)
        poses.append(pose)
        intrinsics.append(calib)

        images = np.stack(images, 0).astype(np.float32)
        poses = np.stack(poses, 0).astype(np.float32)
        intrinsics = np.stack(intrinsics, 0).astype(np.float32)

        poses[:, :3, 3] *= self.scale

        images = torch.from_numpy(images)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)

        # channels first
        images = images.permute(0, 3, 1, 2)
        images = images.contiguous()

        images, intrinsics = scale(images, intrinsics, self.resize)

        return images, poses, intrinsics

    def get_pointclouds(self):
        return torch.from_numpy(self.pointcloud)

    def get_render_poses(self):
        render_poses_raw = np.load(os.path.join(self.root_path, "render_poses", f"{self.scan}_render_pose.npy")) # 1 x 4 x 4

        # re-scale depth
        render_poses_raw[:, :3, 3] *= self.scale

        return torch.tensor(render_poses_raw).float().cuda()