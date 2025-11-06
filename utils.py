#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" CIS6261TML -- Project Option 1 -- utils.py

# This file contains utility functions
"""

import sys
import os

import time

import matplotlib.pyplot as plt

import numpy as np
import sklearn

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset

import torchvision.transforms as T


# CIFAR-10 stuff
CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
CIFAR10_STD  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3,1,1)

CIFAR10_CHANNEL_STATS = (CIFAR10_MEAN, CIFAR10_STD)
CIFAR10_LABELS = ['airplane','automobile','bird','cat','deer', 'dog','frog','horse','ship','truck']

    
# loading
def make_loader(npz_path, split_x, split_y, batch_size=128, shuffle=True, augment=False, order="NCHW"):
    d = np.load(npz_path)
    x, y = d[split_x], d[split_y]
    channel_stats = CIFAR10_CHANNEL_STATS
    
    assert order in ["NHWC", "NCHW"], f"Invalid order {order}."

    assert (x.shape[0] > 1 and x.shape[3] == 3 and x.shape[1] == x.shape[2]), f"Invalid data shape (not in NHWC) -- found: {x.shape}." # ensure NHWC
                        
    if order == "NCHW": # input is in NHWC
        x = torch.from_numpy(x).permute(0,3,1,2).float()   # NHWC -> NCHW
        
    y = torch.from_numpy(y)
    assert y.ndim == 1, f"Invalid y, expected class indices not one hot (found: {y.shape})"
    if y.ndim > 1:   # if input is one-hot -> class indices
        y = y.argmax(1)
    y = y.long()

    ds = TensorDataset(x, y)

    def collate(samples):
        xs, ys = zip(*samples)
        xb = torch.stack(xs, 0)  # NCHW
        yb = torch.stack(ys, 0)

        # augment
        if augment:
            # random horizontal flip
            mask = torch.rand(xb.size(0)) < 0.5
            xb[mask] = xb[mask].flip(-1)

            # random crop with 4px padding
            pad = 4
            xb = F.pad(xb, (pad,pad,pad,pad), mode='reflect')
            N, C, H, W = xb.shape
            i = torch.randint(0, 2*pad+1, (N,))
            j = torch.randint(0, 2*pad+1, (N,))
            out = torch.empty((N, C, 32, 32), dtype=xb.dtype)
            for n in range(N):
                out[n] = xb[n, :, i[n]:i[n]+32, j[n]:j[n]+32]
            xb = out

        # normalize
        mean, std = channel_stats
        xb = (xb - mean) / std

        return xb, yb

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True, collate_fn=collate)  
       
"""
### Sanity check a loader.
"""    
def check_loader(ldr, verb=0):
    xb, yb = next(iter(ldr))
    if verb > 0:
        print('shape/dtype:', xb.shape, xb.dtype, yb.dtype)
        print('pixel range:', float(xb.min()), float(xb.max()))
        print('labels:', yb[:16].tolist())
        
    # the data should be in channel-first NCHW format
    assert (xb.shape[0] > 1 and xb.shape[1] == 3 and xb.shape[2] == xb.shape[3]), f"Invalid data shape (not in NCHW) -- found: {x.shape}." # ensure NCHW

    # note: it's assumed the loader normalizes data based on channel stats so after normalization we expect:
    # mean: ~0 and std: ~1)
    m = xb.mean(dim=(0,2,3))
    s = xb.std(dim=(0,2,3))
    if verb > 0:
        print('per-channel mean:', [round(v,4) for v in m.tolist()])
        print('per-channel std :', [round(v,4) for v in s.tolist()])

    # assert correct labels and no NaNs or Inf
    assert yb.dtype == torch.long and yb.ndim == 1 and int(yb.min()) >= 0 and int(yb.max()) <= 9
    assert torch.isfinite(xb).all(), "Found Inf or NaN!"
                     
"""
### Grabs the requested number of batches from the loader
""" 
def grab_from_loader(loader, num_batches=1):
    xs, ys = [], []
    i = 0
    for xb, yb in loader:
        if i >= num_batches:
            break
        xs.append(xb)
        ys.append(yb)
        i += 1
    x = torch.cat(xs, 0)
    y = torch.cat(ys, 0)
    return x, y
       
        
"""
### Returns prediction label and confidence of the model on the example x. (note: model expects NCHW)
"""
def pred_label_and_conf(model, x, device="cuda", expected_shape=(1,3,32,32)):
    """
    model: a pytorch model
    x: numpy array or torch tensor (C,H,W) or (1,C,H,W)
    Returns: (pred_label, pred_confidence)
    
    We'll automatically deal with the conversion from torch tensor to numpy and vice versa.
    """
    model.eval()
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if x.ndim == 4:
        assert x.shape[0] == 1, f"Invalid input shape. Expecting a single example, got: {x.shape}."
    if x.ndim == 3:  # single image (C,H,W)
        x = x.unsqueeze(0)
        
    assert x.shape == expected_shape, f"Invalid input shape. Expected: {expected_shape}, got: {x.shape}."

    x = x.to(device).float()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_label = int(np.argmax(probs))
    pred_conf = float(probs[pred_label])
    
    return pred_label, pred_conf


"""
### Returns ResNet18 model for CIFAR10 (not pre-trained).
"""
def get_resnet18_cifar(num_classes=10):
    m = torchvision.models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity() # no downsample
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m



"""
## Plots a set of images (all m x m)
## input is  a square number of images, i.e., np.array with shape (z*z, dim_x, dim_y) for some integer z > 1
"""
def plot_images(im, dim_x=32, dim_y=32, one_row=False, out_fp='out.png', save=False, show=True, cmap='gray', fig_size=(14,14), titles=None, titles_fontsize=12):
    fig = plt.figure(figsize=fig_size)
    if im.shape[-1] != 3:
        im = im.reshape((-1, dim_x, dim_y))

    num = im.shape[0]
    assert num <= 3 or np.sqrt(num)**2 == num or one_row, 'Number of images is too large or not a perfect square!'
    
    if titles is not None:
        assert num == len(titles)
    
    if num <= 3:
        for i in range(0, num):
            plt.subplot(1, num, 1 + i)
            plt.axis('off')
            if type(cmap) == list:
                assert len(cmap) == num
                plt.imshow((im[i]*255).astype(np.uint8), cmap=cmap[i]) # plot raw pixel data
            else:
                plt.imshow((im[i]*255).astype(np.uint8), cmap=cmap) # plot raw pixel data
            if titles is not None:
                plt.title(titles[i], fontsize=titles_fontsize)
    else:
        sq = int(np.sqrt(num))
        for i in range(0, num):
            if one_row:
                plt.subplot(1, num, 1 + i)
            else:
                plt.subplot(sq, sq, 1 + i)
            plt.axis('off')
            if type(cmap) == list:
                assert len(cmap) == num
                plt.imshow(im[i], cmap=cmap[i]) # plot raw pixel data
            else:
                plt.imshow(im[i], cmap=cmap) # plot raw pixel data
            if titles is not None:
                plt.title(titles[i], fontsize=titles_fontsize)

    if save:
        plt.savefig(out_fp)

    if show:
        plt.show()
    else:
        plt.close()
   

    
import hashlib

def memv_filehash(fp):
    hv = hashlib.sha256()
    buf = bytearray(512 * 1024)
    memv = memoryview(buf)
    with open(fp, 'rb', buffering=0) as f:
        for n in iter(lambda : f.readinto(memv), 0):
            hv.update(memv[:n])
    hv = hv.hexdigest()
    return hv, hv[-21:-1].upper()
    
    
"""
## Load model from file
"""        
def load_model(fp, model_fn=None, device="cuda"):
    hv, fg = memv_filehash(fp)

    # create model object
    model = model_fn() if model_fn is not None else get_resnet18_cifar()
    model = model.to(device)

    # load the model weights
    state = torch.load(fp, map_location=device)
    model.load_state_dict(state['model'])

    print(f"Loaded model from {fp} -- hash: {fg}.")
    return model, fg
   
   
"""
### Return accuracy of the model with the given loader.
"""
@torch.no_grad()
def eval_model(model, loader, device="cuda"):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return correct / total
    
    
"""
### Return accuracy of predictions given with the given loader.
### Assumes that predict_fn returns logits (or probas) so that getting argmax returns predicted label
"""
@torch.no_grad()
def eval_wrapper(predict_fn, loader, device="cuda"):
    assert predict_fn is not None
    correct = total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = predict_fn(x, device)
        pred = logits.argmax(1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return correct / total
    

# evaluate accuracy (not efficient to do it sample by sample instead of batch by batch but whatevever)
"""
### Return accuracy of the model on the given samples. 
### This implementation works on one sample at a time so is inefficient for large datasets.
### But it also works when xs and ys are numpy arrays instead of torch tensors.
"""
@torch.no_grad()
def eval_model_samples(model, xs, ys, device="cuda"):
    model.eval()
    correct = 0
    total = 0
    for i in range(xs.shape[0]):
        xsi = torch.from_numpy(xs[i]) if type(xs) == np.ndarray else xs[i]
        x = xsi.unsqueeze(0).float().to(device)
        y = int(np.argmax(ys[i])) if ys[i].ndim > 0 else int(ys[i])
        logits = model(x)
        pred = logits.argmax(1).item()
        correct += int(pred == y)
        total += 1
    return correct/total
