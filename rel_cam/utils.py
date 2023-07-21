import json
import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
from torchinfo import summary
import numpy as np
import cv2
import config
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import lovely_tensors as lt

lt.monkey_patch()

def load_model_weights(model, checkpoint_path):
    state = torch.load(checkpoint_path)
    state_dict = state['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

    model_dict = model.state_dict()
    weights = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(weights)
    model.load_state_dict(model_dict)
    return model

def construct_model(model, return_features=True, num_classes=None, freeze_encoder=False):
    if return_features:
        model.fc = torch.nn.Sequential()
    elif num_classes is not None:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError('Must specify either return_features or num_classes')

    if freeze_encoder:
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    return model

def test_net(model, device='cpu', size=(3,224,224), n_batch=32, use_lt=False):
    model = model.to(device)
    model.eval()
    x = torch.randn(n_batch, *size, device=device)
    with torch.no_grad():
        output = model(x)
    if use_lt:
        print('==========================================================================================')
        print(f'Input shape: {x}')
        print(f'Output shape: {output}')
    else:
        print('==========================================================================================')
        print(f'Input shape: {x.shape}')
        print(f'Output shape: {output.shape}')
    summary(model, input_size=(n_batch,*(size)), device=device)

def inspect_model(model, output='params'):
    """
    output: 'params' or 'state'
    """
    if output == 'state': pprint (model.state_dict)
    elif output == 'params':
        for idx, (name, param) in enumerate(model.named_parameters()):
            print (f"{idx}: {name} \n{param}")
            print ('------------------------------------------------------------------------------------------')
    else:
        raise ValueError("Output must be either 'params' or 'state'")

def save_run_config(fname):
    consts = {}
    for k in dir(config):
        if k.isupper() and not k.startswith('_'):
            consts[k] = str(getattr(config, k))
    with open(f'{fname}.conf', 'w') as f:
        f.write(json.dumps(obj=consts, indent=4))

def construct_cm(targets, preds, labels, save_dir=config.MODEL_FOLDER):
    cm = metrics.confusion_matrix(targets, preds)
    cm_df = pd.DataFrame(cm, labels, labels)
    plt.figure(figsize = (9,6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("Predicted")
    plt.ylabel("Ground truth")
    plt.title(f"{config.MODEL_NAME} confusion matrix")
    plt.savefig(os.path.join(save_dir, f'confusion_matrix.png'))

def threshold(x):
    thresh = x.mean() + x.std()
    return (x > thresh)

def normalize(Ac):
    AA = Ac.view(Ac.size(0), -1)
    AA -= AA.min(1, keepdim=True)[0]
    AA /= AA.max(1, keepdim=True)[0]
    return AA.view(Ac.shape)

def tensor2image(x, i=0):
    x = normalize(x)
    x = x[i].detach().cpu().numpy()
    x = cv2.resize(np.transpose(x, (1, 2, 0)), config.INPUT_SIZE)
    return x
    
#################################################################

class ScoreCAM(object):
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = dict()
        self.activations = dict()

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        def backward_hook(module,input,output):
            self.gradients['value'] = output[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        score_weight = []
        # predication on raw input
        logit = self.model(input).cuda()

        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()

        logit = F.softmax(logit, dim=1)

        if torch.cuda.is_available():
          predicted_class= predicted_class.cuda()
          score = score.cuda()
          logit = logit.cuda()

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        activations = self.activations['value']
        b, k, u, v = activations.size()

        score_saliency_map = torch.zeros((1, 1, h, w))

        if torch.cuda.is_available():
          activations = activations.cuda()
          score_saliency_map = score_saliency_map.cuda()

        with torch.no_grad():
          for i in range(k):

              # upsampling
              saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
              saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

              if saliency_map.max() == saliency_map.min():
                  score_weight.append(0)
                  continue

              # normalize to 0-1
              norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

              # how much increase if keeping the highlighted region
              # predication on masked input
              output = self.model(input * norm_saliency_map)
              output = F.softmax(output, dim=1)
              score = output[0][predicted_class]

              score_saliency_map += score * saliency_map
              score_weight.append(score.detach().cpu().numpy())

        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_map, score_weight

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)

def LRP(output, max_index):
    if max_index == None:
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        print(f'Pred cls : {str(pred)}')
        max_index = pred.squeeze().cpu().numpy()
    tmp = np.zeros(output.shape)
    tmp[:, max_index] = 1
    T = tmp
    T = torch.from_numpy(T).type(torch.FloatTensor)
    Tt = Variable(T).cuda()
    return Tt

def SGLRP(output, max_index = None):
    output = F.softmax(output, dim=1)
    if max_index == None:
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        print(f'Pred cls : {str(pred)}')
        max_index = pred.squeeze().cpu().numpy()

    output = output.detach().cpu().numpy()
    tmp = output.copy()
    y_t = tmp[:, max_index].copy()
    tmp *= y_t
    tmp[:, max_index] = 0

    Tn = tmp
    Tn = torch.from_numpy(Tn).type(torch.FloatTensor)
    Tn = Variable(Tn).cuda()

    tmp = np.zeros(output.shape)
    tmp[:, max_index] = y_t*(1-y_t)
    T = tmp
    T = torch.from_numpy(T).type(torch.FloatTensor)
    Tt = Variable(T).cuda()

    return Tt, Tn

def CLRP(output, max_index = None):
    if max_index == None:
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        print(f'Pred cls : {str(pred)}')
        max_index = pred.squeeze().cpu().numpy()

    tmp = np.ones(output.shape)
    tmp *= 1 / 1000
    tmp[:, max_index] = 0
    Tn = tmp
    with torch.no_grad():
        Tn = torch.from_numpy(Tn).type(torch.FloatTensor)
        Tn = Variable(Tn).cuda()

    tmp = np.zeros(output.shape)
    tmp[:, max_index] = 1
    T = tmp
    with torch.no_grad():
        T = torch.from_numpy(T).type(torch.FloatTensor)
        Tt = Variable(T).cuda()
        
    return Tt,Tn