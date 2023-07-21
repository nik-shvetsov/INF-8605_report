import torch
from model import PatchClassifier, ModelHandler
from dataset import PatchDataset
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config
from torch.utils.data import DataLoader, random_split
from utils import construct_cm


if __name__ == "__main__":
    torch.set_float32_matmul_precision(config.MATMUL_PRECISION)
    torch.backends.cudnn.benchmark = config.CUDNN_BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN_DETERMINISTIC

    dataset = PatchDataset(config.DATA_DIR, config.ATF, preproc=True, augment=True)
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset, test_subset = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)

    train_dataloader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True if config.ACCELERATOR == 'cuda' else False)
    val_dataloader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True if config.ACCELERATOR == 'cuda' else False)
    test_dataloader = DataLoader(test_subset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True if config.ACCELERATOR == 'cuda' else False)

    # Model
    model = PatchClassifier(
        out_classes=config.NUM_CLASSES
    ).model

    optimizer = config.OPTIMIZER(model.parameters(), **config.OPTIMIZER_PARAMS)
    scheduler = config.SCHEDULER(optimizer, **config.SCHEDULER_PARAMS)
    
    mh = ModelHandler(
        model=model,
        loss_fn=config.CRITERTION(),
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        valid_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        device=config.ACCELERATOR
    )

    # Train model
    mh.train(accumulate_grad_batches=config.ACCUM_GRAD_BATCHES, save_final=True)

    # Evaluate on test set
    out_eval = mh.evaluate(dataloader=test_dataloader)
    loss = out_eval["avg_loss"]
    accuracy = out_eval["accuracy"]
    precision = out_eval["precision"]
    recall = out_eval["recall"]
    roc_auc = out_eval["roc_auc"]

    print(f"""
    Model evaluation: \t
    Loss = [{loss:0.5f}] \t
    Accuracy = [{(accuracy * 100):0.2f}%] \t
    Precision = [{(precision * 100):0.2f}%] \t
    Recall = [{(recall * 100):0.2f}%] \t
    ROC AUC = [{(roc_auc * 100):0.2f}%] \t
    """)

    targets = torch.tensor([dataset.targets[i] for i in test_subset.indices])
    construct_cm(targets, out_eval["preds"], dataset.class_to_idx.keys(), save_dir=config.MODEL_FOLDER)


