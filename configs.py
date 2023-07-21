import torch
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
from pathlib import Path

CONFIGS = {
    'resnet18_ssl': {
        "MODEL_NAME" :'resnet18',
        "FEATURES_ONLY": False,
        "NUM_CLASSES": 5,
        "FREEZE_ENCODER": True,

        "NUM_EPOCHS": 100,
        "ACCUM_GRAD_BATCHES": 1,
        "USE_AMP": True,
        "USE_GRAD_SCALER": True,
        "EARLY_STOP": True,
        "PATIENCE": 15,
        "OPTIMIZER": torch.optim.Adam,
        "OPTIMIZER_PARAMS": {
            "lr": 0.001,
            "weight_decay": 5e-4,
        },
        "SCHEDULER": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "SCHEDULER_PARAMS": {
            "mode": 'min', 
            "factor": 0.1,
            "patience": 5, 
            "verbose": True,
        },
        "CRITERTION": torch.nn.CrossEntropyLoss,

        "BATCH_SIZE": 128,
        "INPUT_SIZE": (224, 224),
        "NUM_WORKERS": 8,
        "PREPROC": True,
        "AUGMENT": True,
        "MODEL_FOLDER": Path('models'),

        "MATMUL_PRECISION": "medium",
        "ACCELERATOR": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "CUDNN_BENCHMARK": False,
        "CUDNN_DETERMINISTIC": True,

        "DATA_DIR": '/mnt/wd8d/pathology/ROI/LC25000/combined/',
        "CHECKPOINT_FILE": '/code/INF-8605/self-supervised-histopathology/ssl_histo/tenpercent_resnet18.ckpt',
        "ATF": {
            'aug': alb.Compose([
                alb.HorizontalFlip(),
                alb.VerticalFlip(),
            ]),
            'preproc': alb.Compose([
                alb.Resize(height=256, width=256),
                alb.CenterCrop(224, 224, always_apply=False, p=1.0),
                alb.Normalize(always_apply=True),
            ]),
            'resize_to_tensor': alb.Compose([
                alb.Resize(height=224, width=224),
                ToTensorV2(),
            ]),
            'tn': alb.Compose([
                alb.Normalize(always_apply=True),
                ToTensorV2(),
            ]),
            'unnorm': alb.Compose([
                alb.Normalize(always_apply=True, mean=[ 0., 0., 0. ],
                          std=(1.0 / torch.tensor([0.229, 0.224, 0.225])).tolist()),
                alb.Normalize(always_apply=True, mean=(-1 * torch.tensor([0.485, 0.456, 0.406])).tolist(),
                          std= [ 1., 1., 1. ]),
            ]),
        },
    },
    #########################################################################################
    'resnet18_ssl_aug': {
        "MODEL_NAME" :'resnet18',
        "FEATURES_ONLY": False,
        "NUM_CLASSES": 5,
        "FREEZE_ENCODER": True,

        "NUM_EPOCHS": 100,
        "ACCUM_GRAD_BATCHES": 1,
        "USE_AMP": True,
        "USE_GRAD_SCALER": True,
        "EARLY_STOP": True,
        "PATIENCE": 15,
        "OPTIMIZER": torch.optim.Adam,
        "OPTIMIZER_PARAMS": {
            "lr": 0.001,
            "weight_decay": 5e-4,
        },
        "SCHEDULER": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "SCHEDULER_PARAMS": {
            "mode": 'min', 
            "factor": 0.1,
            "patience": 5, 
            "verbose": True,
        },
        "CRITERTION": torch.nn.CrossEntropyLoss,

        "BATCH_SIZE": 128,
        "INPUT_SIZE": (224, 224),
        "NUM_WORKERS": 8,
        "PREPROC": True,
        "AUGMENT": True,
        "MODEL_FOLDER": Path('models'),

        "MATMUL_PRECISION": "medium",
        "ACCELERATOR": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "CUDNN_BENCHMARK": False,
        "CUDNN_DETERMINISTIC": True,

        # "DATA_DIR": '/mnt/wd8d/pathology/ROI/LC25000/lung_colon_image_set/lung_image_sets/'
        # "DATA_DIR": '/mnt/wd8d/pathology/ROI/LC25000/lung_colon_image_set/colon_image_sets/'
        "DATA_DIR": '/mnt/wd8d/pathology/ROI/LC25000/combined/',
        "CHECKPOINT_FILE": '/code/INF-8605/self-supervised-histopathology/ssl_histo/tenpercent_resnet18.ckpt',
        "ATF": {
            'aug': alb.Compose([
                alb.HorizontalFlip(),
                alb.VerticalFlip(),
                alb.Rotate(180),
                alb.Transpose(),
                alb.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                alb.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), p=0.5),
                alb.OneOf([
                    alb.OpticalDistortion(distort_limit=1.0),
                    alb.GridDistortion(num_steps=5, distort_limit=1.),
                    alb.ElasticTransform(alpha=3),
                ], p=0.2),
                alb.OneOf([
                    alb.HueSaturationValue(10, 15, 10),
                    alb.RandomBrightnessContrast(),
                ], p=0.3),
                alb.OneOf([
                    alb.GaussNoise(),
                    alb.MotionBlur(),
                    alb.MedianBlur(5),
                    alb.GaussianBlur(),
                ], p=0.1),
            ]),
            'preproc': alb.Compose([
                alb.Resize(height=256, width=256),
                alb.CenterCrop(224, 224, always_apply=False, p=1.0),
                alb.Normalize(always_apply=True),
            ]),
            'resize_to_tensor': alb.Compose([
                alb.Resize(height=224, width=224),
                ToTensorV2(),
            ]),
            'tn': alb.Compose([
                alb.Normalize(always_apply=True),
                ToTensorV2(),
            ]),
            'unnorm': alb.Compose([
                alb.Normalize(always_apply=True, mean=[ 0., 0., 0. ],
                          std=(1.0 / torch.tensor([0.229, 0.224, 0.225])).tolist()),
                alb.Normalize(always_apply=True, mean=(-1 * torch.tensor([0.485, 0.456, 0.406])).tolist(),
                          std= [ 1., 1., 1. ]),
            ]),
        },
    },
    #########################################################################################
}
