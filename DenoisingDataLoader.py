import torch
import os
import numpy as np
import pandas as pd
from skimage import io
from torch.utils.data import Dataset


class NoisyDataset(Dataset):
    def __init__(self, csv_file, noisy_dir, gt_dir, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.noisy_dir, self.metadata.iloc[idx, 2])  # Index of image name in CSV
        img = io.imread(img_name)
        gt_img_name = os.path.join(self.gt_dir, self.metadata.iloc[idx, 3])  # Index of GT img name in CSV
        gt_img = io.imread(gt_img_name)
        sample = {"image": img, "gt_img": gt_img}
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    def __call__(self, sample):
        image, gt_img = sample['image'], sample['gt_img']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if np.amax(image) > 1:
            # If the image is in 0-255 range, scale it to 0-1
            image = image / 255.0
            gt_img = gt_img / 255.0
        if len(image.shape) == 2:
            #  For loading grayscale images
            image = np.expand_dims(image, axis=3)
            gt_img = np.expand_dims(gt_img, axis=3)
        image = image.transpose((2, 0, 1))
        gt_img = gt_img.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'gt_img': torch.from_numpy(gt_img)}
