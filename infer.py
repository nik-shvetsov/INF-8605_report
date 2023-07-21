import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import random_split
import numpy as np

from dataset import PatchDataset
from model import PatchClassifier, ModelHandler
import config
import utils

from PIL import Image

import lovely_tensors as lt

lt.monkey_patch()
lt.set_config(sci_mode=False)
torch.set_printoptions(sci_mode=False)


if __name__ == '__main__':
    
    INF_DICE = 87.36
    model = PatchClassifier(
        features_only=config.FEATURES_ONLY, 
        out_classes=config.NUM_CLASSES,
        freeze_encoder=config.FREEZE_ENCODER,
    )
    
    dataset = PatchDataset(config.DATA_DIR, config.ATF, preproc=False, augment=False)
    generator = torch.Generator().manual_seed(42)
    _, _, test_subset = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)

    mh = ModelHandler(model, device=config.ACCELERATOR)
    mh.load(os.path.join(config.MODEL_FOLDER, f'{config.CONFIG_ID}_{INF_DICE}.pt'))
    mh.model.eval()
    
    img_idx = np.random.randint(0, len(test_subset))
    if not config.FEATURES_ONLY:
        result = mh.predict(transforms.ToPILImage()(test_subset[img_idx][0]), show_results=False, label_dict=test_subset.dataset.idx_to_class, fname='_test.png')
        print (f"Correct? {test_subset[img_idx][1] == result[0]} | {result[2] if len(result) == 3 else result[0]} {result[1]:.2f}")
    else:
        print ("config.FEATURES_ONLY is set to True")
