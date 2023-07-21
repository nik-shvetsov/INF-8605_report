import torch
import torchvision
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import albumentations as alb
import timm
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_auroc, multiclass_precision, multiclass_recall, multiclass_stat_scores
from pathlib import Path
from pprint import pprint
from tqdm import tqdm, trange
import config
from utils import test_net, save_run_config, construct_model, load_model_weights, inspect_model
import matplotlib.pyplot as plt

import numpy as np
import lovely_tensors as lt

lt.monkey_patch()
lt.set_config(sci_mode=False)
torch.set_printoptions(sci_mode=False)


class PatchClassifier(nn.Module):
    def __init__(self, features_only=True, out_classes=None, freeze_encoder=None):
        super().__init__()
        self.model = construct_model(
            load_model_weights(
                torchvision.models.__dict__['resnet18'](weights=None), 
                config.CHECKPOINT_FILE
            ),
        return_features=features_only,
        num_classes=out_classes,
        freeze_encoder=freeze_encoder)

    def forward(self, image):
        output = self.model(image)
        return output

class ModelHandler():
    def __init__(self, model, loss_fn=None, optimizer=None, scheduler=None, train_dataloader=None, 
                    valid_dataloader=None, test_dataloader=None, device=None):
        super().__init__()
        self.device = device if device is not None else 'cpu'
        self.model = model.to(device=self.device)

        self.criterion = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = torch.cuda.amp.GradScaler()

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        if config.EARLY_STOP:
            self.patience = config.PATIENCE
            self.best_trained_model = {
                "score": 0,
                "state_dict": None,
                "current_pateince": self.patience,
            }

    def test_step(self, imgs, targets):
        self.model.eval()
        logits = self.model(imgs)
        (pred_logits, pred_idxs) = logits.max(1)

        return {
            "logits": logits,
            "preds": pred_idxs
        }

    def eval_step(self, imgs, targets):
        self.model.eval()

        if config.USE_AMP:
            with torch.amp.autocast(device_type=config.ACCELERATOR, cache_enabled=True):
                logits = self.model(imgs)
                loss = self.criterion(logits, targets)
        else:
            logits = self.model(imgs)
            loss = self.criterion(logits, targets)

        (pred_logits, pred_idxs) = logits.max(1)

        return {
            "loss": float(loss.item()),
            "preds": pred_idxs,
            "class_probs": F.softmax(logits, dim=1)
        }

    def evaluate(self, dataloader):
        total_loss = 0
        preds = []
        class_probs = []

        with torch.no_grad():
            for batch_idx, (imgs, targets) in enumerate(dataloader):
                imgs = imgs.type(torch.FloatTensor).to(device=self.device)
                targets = targets.type(torch.LongTensor).to(device=self.device)

                out_eval_step = self.eval_step(imgs, targets)
                
                batch_loss = out_eval_step["loss"]
                total_loss += batch_loss

                preds.extend(out_eval_step["preds"].tolist())

                class_probs.append(out_eval_step["class_probs"])
        
        if isinstance(dataloader.dataset, torch.utils.data.Subset):
            targets = torch.tensor([dataloader.dataset.dataset.targets[i] for i in dataloader.dataset.indices])
        else:
            targets = torch.tensor(dataloader.dataset.targets)

        accuracy = multiclass_accuracy(torch.tensor(preds), targets, num_classes=config.NUM_CLASSES, average='micro')
        precision = multiclass_precision(torch.tensor(preds), targets, num_classes=config.NUM_CLASSES, average='macro')
        recall = multiclass_recall(torch.tensor(preds), targets, num_classes=config.NUM_CLASSES, average='macro')
        roc_auc = multiclass_auroc(torch.tensor(torch.cat(class_probs, dim=0).numpy(force=True)), targets, num_classes=config.NUM_CLASSES, average='macro')

        return {
            'avg_loss': total_loss / len(dataloader),
            'accuracy': float(accuracy.item()),
            'precision': float(precision.item()),
            'recall': float(recall.item()),
            'roc_auc': float(roc_auc.item()),
            'preds': preds,
        }

    def train_step(self, imgs, targets, grad_step=True, accumulate_norm_factor=1):
        self.model.train()

        # Forward
        if config.USE_AMP:
            with torch.amp.autocast(device_type=config.ACCELERATOR, cache_enabled=True):
                logits = self.model(imgs)
                loss = self.criterion(logits, targets)
        else:
            logits = self.model(imgs)
            loss = self.criterion(logits, targets)

        if config.USE_GRAD_SCALER:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if grad_step: self.optimizer.step()
                
        if grad_step: self.optimizer.zero_grad()
        return float(loss.item()) / accumulate_norm_factor
    
    def train(self, train_dataloader=None, valid_dataloader=None, accumulate_grad_batches=1, save_final=False):
        if train_dataloader is None: train_dataloader = self.train_dataloader
        if valid_dataloader is None: valid_dataloader = self.valid_dataloader

        self.model = self.model.to(self.device)
        for epoch in trange(1, config.NUM_EPOCHS + 1, desc='Epochs'):
            train_loss = 0 # train epoch loss
            for batch_idx, (imgs, targets) in enumerate(tqdm(train_dataloader, desc='Batches', leave=False)):

                imgs = imgs.type(torch.FloatTensor).to(device=self.device)
                targets = targets.type(torch.LongTensor).to(device=self.device)
                do_grad_step = ((batch_idx + 1) % accumulate_grad_batches == 0) or (batch_idx + 1 == len(train_dataloader))

                # train batch loss
                train_loss += self.train_step(imgs, targets, grad_step=do_grad_step, accumulate_norm_factor=accumulate_grad_batches) 

            out_eval = self.evaluate(valid_dataloader)

            loss = out_eval["avg_loss"]
            accuracy = out_eval["accuracy"]
            precision = out_eval["precision"]
            recall = out_eval["recall"]
            roc_auc = out_eval["roc_auc"]
   
            tqdm.write(f"""
            Epoch [{epoch}]: \t 
            train loss = [{train_loss:0.5f}] \t 
            val loss = [{loss:0.5f}] \t 
            val accuracy = [{(accuracy * 100):0.2f}%] \t
            val precision = [{(precision * 100):0.2f}%] \t
            val recall = [{(recall * 100):0.2f}%] \t
            val ROC AUC = [{(roc_auc * 100):0.2f}%] \t
            """) 
            
            # Scheduler step
            if self.scheduler is not None: 
                if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                    self.scheduler.step(loss)
                elif type(self.scheduler) == torch.optim.lr_scheduler.MultiStepLR:
                    self.scheduler.step()
            
            # Early stopping
            if self.best_trained_model is not None:
                if accuracy > self.best_trained_model['score']:
                    self.best_trained_model['score'] = accuracy
                    self.best_trained_model['model_state_dict'] = self.model.state_dict().copy()
                    self.best_trained_model['current_pateince'] = self.patience
                else:
                    self.best_trained_model['current_pateince'] -= 1
                if self.best_trained_model['current_pateince'] < 0:
                    tqdm.write(f"Early stopping at epoch {epoch}")
                    break
                
        if save_final:
            if self.best_trained_model is not None:
                self.model.load_state_dict(self.best_trained_model['model_state_dict'])
            
            out_eval = self.evaluate(self.test_dataloader if self.test_dataloader is not None else self.valid_dataloader)
            accuracy = out_eval["accuracy"]
            
            print(f"Saving model {config.CONFIG_ID}: \"{config.MODEL_NAME}\" with test set accuracy score = {(accuracy * 100):0.2f}")
            self.save(f'{config.MODEL_FOLDER}/{config.CONFIG_ID}_{(accuracy * 100):0.2f}.pt')
        
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_run_config(path)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=config.ACCELERATOR))
    
    def predict(self, pil_img, show_results=False, label_dict=None, fname=None):
        self.model = self.model.to(self.device)
        self.model.eval()

        if isinstance(config.ATF['preproc'], transforms.Compose):
            preproc_img = {'image': pil_img}
            preproc_img['image'] = config.ATF['preproc'](preproc_img['image'])
            preproc_img['image'] = np.asarray(preproc_img['image'].permute(1, 2, 0))
        elif isinstance(config.ATF['preproc'], alb.core.composition.Compose):
            preproc_img = {'image': np.asarray(pil_img)}
            preproc_img = config.ATF['preproc'](image=preproc_img['image'])
        preproc_img = config.ATF['resize_to_tensor'](image=preproc_img['image'])['image']

        img = preproc_img.unsqueeze(0).type(torch.FloatTensor).to(device=self.device)
        with torch.no_grad():            
            logits = self.model(img)
            probs = F.softmax(logits, dim=1)
            (prob, prediction) = probs.max(1)
            # (_, prediction) = logits.max(1)
            
            prob = float(prob)
            prediction = int(prediction)
            
            plt.figure(figsize=(10, 5))
            plt.title(prediction) if label_dict is None else plt.title(f"{label_dict[prediction]} ({prediction}): {prob:.2f}")
            plt.imshow(pil_img)
            if fname is not None: plt.savefig(fname)
            if show_results: plt.show()
            
            return (prediction, prob, label_dict[prediction]) if label_dict is not None else (prediction, prob)

    def get_probs(self, pil_img, label_dict=None):
        self.model = self.model.to(self.device)
        self.model.eval()

        if isinstance(config.ATF['preproc'], transforms.Compose):
            preproc_img = {'image': pil_img}
            preproc_img['image'] = config.ATF['preproc'](preproc_img['image'])
            preproc_img['image'] = np.asarray(preproc_img['image'].permute(1, 2, 0))
        elif isinstance(config.ATF['preproc'], alb.core.composition.Compose):
            preproc_img = {'image': np.asarray(pil_img)}
            preproc_img = config.ATF['preproc'](image=preproc_img['image'])
        preproc_img = config.ATF['resize_to_tensor'](image=preproc_img['image'])['image']

        img = preproc_img.unsqueeze(0).type(torch.FloatTensor).to(device=self.device)
        with torch.no_grad():            
            logits = self.model(img)
            probs = F.softmax(logits, dim=1).squeeze(0).tolist()
            return probs if label_dict is None else {label_dict[i]: probs[i] for i in range(len(probs))}


if __name__ == '__main__':
    model = PatchClassifier(
        features_only=False, 
        out_classes=5, 
        freeze_encoder=True
    ).model.eval().to(config.ACCELERATOR)

    inspect_model(model, output='state')
    # test_net(model, size=(3,224,224), n_batch=1, use_lt=True)
    
    
