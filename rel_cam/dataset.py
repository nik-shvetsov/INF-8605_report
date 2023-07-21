import os
from glob import glob
from pathlib import Path
from collections import Counter
import config

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import albumentations as alb

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import lovely_tensors as lt
lt.monkey_patch()


class PatchDataset(Dataset):
    def __init__(self, data_path, tf, preproc=True, augment=True):
        self.data_path = data_path
        classes = sorted([x.name for x in Path(data_path).iterdir() if x.is_dir()])
        self.class_to_idx = {class_label: idx  for idx, class_label in enumerate(classes)} # aka data_classes, {'necrosis': 0, 'normal_lung': 1, 'stroma_tls': 2}
        self.idx_to_class = {idx: class_label  for idx, class_label in enumerate(classes)}
        self.imgs = sorted(glob(f"{os.path.join(data_path)}/*/*"))
        self.targets = [self.class_to_idx[Path(img).parent.name] for img in self.imgs]
        
        self.transforms = tf
        self.augment = augment
        self.preproc = preproc

        self.imgs_per_class = {}
        for class_label in classes:
            self.imgs_per_class[class_label] = len(glob(f"{os.path.join(data_path, class_label)}/*"))
        
        assert len(classes) == config.NUM_CLASSES, f"Number of classes in dataset ({len(classes)}) does not match config ({config.NUM_CLASSES})"
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        preproc_img = {'image': cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)}
        if self.preproc:
            if isinstance(self.transforms['preproc'], transforms.Compose):
                preproc_img['image'] = Image.fromarray(preproc_img['image'])
                preproc_img['image'] = self.transforms['preproc'](preproc_img['image'])
                preproc_img['image'] = preproc_img['image'].permute(1, 2, 0)
                preproc_img['image'] = np.asarray(preproc_img['image'])
            elif isinstance(self.transforms['preproc'], alb.core.composition.Compose):
                preproc_img = self.transforms['preproc'](image=preproc_img['image'])
        if self.augment:
            preproc_img = self.transforms['aug'](image=preproc_img['image'])

        preproc_img = self.transforms[f'resize_to_tensor'](image=preproc_img['image'])
        img = preproc_img['image']
        label = torch.tensor(self.class_to_idx[Path(self.imgs[idx]).parent.name])

        return img, label

def get_subset_targets(subset):
    targets = [subset.dataset.targets[i] for i in subset.indices]
    targets_d = {list(subset.dataset.class_to_idx.keys())[list(subset.dataset.class_to_idx.values()).index(k)]: v for k, v in dict(Counter(targets)).items()}
    return dict(sorted(targets_d.items()))


if __name__ == '__main__':
    n_show = 4
    
    dataset = PatchDataset(config.DATA_DIR, config.ATF, preproc=True, augment=True)
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset, test_subset = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)

    train_dataloader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True if config.ACCELERATOR == 'cuda' else False)
    val_dataloader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True if config.ACCELERATOR == 'cuda' else False)
    test_dataloader = DataLoader(test_subset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True if config.ACCELERATOR == 'cuda' else False)
    
    print("=====================================")
    print("Dataset info:")
    print(f"Labels: ", dataset.class_to_idx)
    print("-------------------------------------")
    print(f"Train dataset: {len(train_subset)}")
    print(f"Number of images per class: ", get_subset_targets(train_subset))
    print("-------------------------------------")
    print(f"Val dataset: {len(val_subset)}")
    print(f"Number of images per class: ", get_subset_targets(val_subset))
    print("-------------------------------------")
    print(f"Test dataset: {len(test_subset)}")
    print(f"Number of images per class: ", get_subset_targets(test_subset))
    print("=====================================")

    for batch_idx, (imgs, labels) in enumerate(test_dataloader):
        print(f"Imgs tensor: {imgs}")
        print(f"Image labels: {labels[:n_show]}")
        x = imgs[:n_show] if n_show < config.BATCH_SIZE else imgs[:config.BATCH_SIZE]
        grid = make_grid(x.view(-1, 3, config.INPUT_SIZE[0], config.INPUT_SIZE[1]))
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.savefig('dataloader_pbatch.png')
        break

