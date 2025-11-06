#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" CIS6261TML -- Project Option 1 -- part1.py

# This file contains the part1 code
"""

import sys
import os

import time

import numpy as np

import sklearn
from sklearn.metrics import confusion_matrix

import torch

import utils # we need this

import diffpure_utils
from diffpure_utils import SDE_Adv_Model, parse_args_and_config, Logger


######### Prediction Fns #########

"""
## Basic prediction function
"""
@torch.no_grad()
def basic_predict(model, x, device="cuda"):
    x = x.to(device)
    logits = model(x)
    return logits


#### TODO: implement your defense(s) as a new prediction function
#### Make sure it is compatible with the rest of the code in this file:
####    - it needs to take model, x, device
####    - it needs to return logits
#### Note: if your predict function operates on probabilities/labels (instead of logits), that is fine provided you adjust the rest of the code.
#### Put your code here

@torch.no_grad()
def diffpure_predict(model, x, device="cuda"):
    
    args, config = parse_args_and_config()
    
    args.classifier = model
    
    log_dir = os.path.join(args.image_folder, args.classifier_name,
                           'seed' + str(args.seed), 'data' + str(args.data_seed))
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    logger = Logger(file_name=f'{log_dir}/log.txt', file_mode="w+", should_flush=True)

    ngpus = torch.cuda.device_count()
    adv_batch_size = args.adv_batch_size * ngpus
    # print(f'ngpus: {ngpus}, adv_batch_size: {adv_batch_size}')

    # load model
    # print('starting the model and loader...')
    new_model = SDE_Adv_Model(args, config, classifier=model)
    if ngpus > 1:
        new_model = torch.nn.DataParallel(new_model)
    new_model = new_model.eval().to(config.device)
    
    
    # --- SDE-based prediction (DiffPure path) ---
    # print('starting the prediction with SDE...')
    x = x.to(config.device if hasattr(config, "device") else device)
    out = new_model(x)  # SDE_Adv_Model does purification + classification
    logits = out[0] if isinstance(out, (tuple, list)) else out
    
    logger.close()
    return logits


######### Membership Inference Attacks (MIAs) #########

"""
## A very simple confidence threshold-based MIA
"""
@torch.no_grad()
def simple_conf_threshold_mia(predict_fn, x, thresh=0.999, device="cuda"):   
    pred_y = predict_fn(x, device).cpu()
    pred_y_probas = torch.softmax(pred_y, dim=1).numpy()
    pred_y_conf = np.max(pred_y_probas, axis=-1)
    return (pred_y_conf > thresh).astype(int)
    
    
"""
## A very simple logit threshold-based MIA
"""
@torch.no_grad()
def simple_logits_threshold_mia(predict_fn, x, thresh=9, device="cuda"):   
    pred_y = predict_fn(x, device).cpu().numpy()
    pred_y_max_logit = np.max(pred_y, axis=-1)
    return (pred_y_max_logit > thresh).astype(int)
    
    
#### TODO [optional] implement new MIA attacks.
#### Put your code here
  
  
######### Adversarial Examples #########

  
#### TODO [optional] implement new adversarial examples attacks.
#### Put your code here  
#### Note: you should have your code save the data to file so it can be loaded and evaluated in Main() (see below).
    
    

def load_and_grab(fp, name, num_batches=4, batch_size=256, shuffle=True):
    loader = utils.make_loader(fp, f"{name}_x", f"{name}_y", batch_size=batch_size, shuffle=shuffle)
    utils.check_loader(loader)
    
    return utils.grab_from_loader(loader, num_batches=num_batches)
    

def load_advex(fp):
    data = np.load(fp)
    return data['adv_x'], data['benign_x'], data['benign_y']
   
######### Main() #########
   
if __name__ == "__main__":

    # Let's check our software versions
    print('### Python version: ' + __import__('sys').version)
    print('### NumPy version: ' + np.__version__)
    print('### Pytorch version: ' + torch.__version__)
    print('------------')

    # deterministic seed for reproducibility
    seed = 42

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Device: {device} ---")
    print("-------------------")
    
    # keep track of time
    st = time.time()

    #### load the data
    print('\n------------ Loading Data & Model ----------')
    
    train_loader = utils.make_loader('./data/train.npz', 'train_x', 'train_y', batch_size=256, shuffle=False)
    utils.check_loader(train_loader)
    
    # val loader
    val_loader = utils.make_loader('./data/valtest.npz', 'val_x', 'val_y', batch_size=512, shuffle=False)
    utils.check_loader(val_loader)
    
    
    # create the model object   
    model_fp = './target_model.pt'
    assert os.path.exists(model_fp) # model must exist
    
    model, fg = utils.load_model(model_fp, device=device)
    assert fg == "0CCE0F932C863D6648E0", f"Modified model file {model_fp}!"
    
    st_after_model = time.time()
        
    ### let's evaluate the raw model on the train and val data
    train_acc = utils.eval_model(model, train_loader, device=device)
    val_acc = utils.eval_model(model, val_loader, device=device)
    print(f"[Raw model] Train accuracy: {train_acc:.4f} ; Val accuracy: {val_acc:.4f}.")
    
    
    ### let's wrap the model prediction function so it could be replaced to implement a defense    
    ### Turn this to True to evaluate your defense (turn it back to False to see the undefended model).
    defense_enabled = True
    
    if defense_enabled:
        predict_fn =  lambda x, dev: diffpure_predict(model, x, device=dev) #
    else:
        # predict_fn points to undefended model
        predict_fn = lambda x, dev: basic_predict(model, x, device=dev)
    
    ### now let's evaluate the model with this prediction function wrapper
    train_acc = utils.eval_wrapper(predict_fn, train_loader, device=device)
    val_acc = utils.eval_wrapper(predict_fn, val_loader, device=device)
    
    print(f"[Model] Train accuracy: {train_acc:.4f} ; Val accuracy: {val_acc:.4f}.")
        
    
    ### evaluating the privacy of the model wrt membership inference
    # load the data
    in_x, in_y = load_and_grab('./data/members.npz', 'members', num_batches=2)
    out_x, out_y = load_and_grab('./data/nonmembers.npz', 'nonmembers', num_batches=2)
    
    mia_eval_x = torch.cat([in_x, out_x], 0)
    mia_eval_y = torch.cat([torch.ones_like(in_y), torch.zeros_like(out_y)], 0)
    mia_eval_y = mia_eval_y.cpu().detach().numpy().reshape((-1,1))
    
    assert mia_eval_x.shape[0] == mia_eval_y.shape[0]
    
    # so we can add new attack functions as needed
    print('\n------------ Privacy Attacks ----------')
    mia_attack_fns = []
    mia_attack_fns.append(('Simple Conf threshold MIA', simple_conf_threshold_mia))
    mia_attack_fns.append(('Simple Logits threshold MIA', simple_logits_threshold_mia))
    # add more lines here to add more attacks
    
    for i, tup in enumerate(mia_attack_fns):
        attack_str, attack_fn = tup
        
        in_out_preds = attack_fn(predict_fn, mia_eval_x, device=device).reshape((-1,1))       
        assert in_out_preds.shape == mia_eval_y.shape, 'Invalid attack output format'
        
        cm = confusion_matrix(mia_eval_y, in_out_preds, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        
        attack_acc = np.trace(cm) / np.sum(np.sum(cm))
        attack_tpr = tp / (tp + fn)
        attack_fpr = fp / (fp + tn)
        attack_adv = attack_tpr - attack_fpr
        attack_precision = tp / (tp + fp)
        attack_recall = tp / (tp + fn)
        attack_f1 = tp / (tp + 0.5*(fp + fn))
        print(f"{attack_str} --- Attack acc: {100*attack_acc:.2f}%; advantage: {attack_adv:.3f}; precision: {attack_precision:.3f}; recall: {attack_recall:.3f}; f1: {attack_f1:.3f}.")
    
    
    ### evaluating the robustness of the model wrt adversarial examples
    print('\n------------ Adversarial Examples ----------')
    advexp_fps = []
    
    advexp_fps.append(('Attack0', 'advexp0.npz', '519D7F5E79C3600B366A'))
    # uncomment/add more lines to add more attacks.
    #advexp_fps.append(('Attack1', 'advexp1.npz', None)) 
    
    for i, tup in enumerate(advexp_fps):
        attack_str, attack_fp, attack_hash = tup
        
        assert os.path.exists(attack_fp), f"Attack file {attack_fp} not found."
        _, fg = utils.memv_filehash(attack_fp)
        if attack_hash is not None:
            assert fg == attack_hash, f"Modified attack file {attack_fp}."
        
        # load the attack data
        adv_x, benign_x, benign_y = load_advex(attack_fp)
        benign_y = benign_y.flatten()
        
        benign_pred_y = predict_fn(torch.from_numpy(benign_x), device).cpu().numpy()
        benign_pred_y = np.argmax(benign_pred_y, axis=-1).astype(int)
        benign_acc = np.mean(benign_y == benign_pred_y)
        
        adv_pred_y = predict_fn(torch.from_numpy(adv_x), device).cpu().numpy()
        adv_pred_y = np.argmax(adv_pred_y, axis=-1).astype(int)
        adv_acc = np.mean(benign_y == adv_pred_y)
        
        print(f"{attack_str} [{fg}] --- Benign acc: {100*benign_acc:.2f}%; adversarial acc: {100*adv_acc:.2f}%")     
    print('------------\n')

    et = time.time()
    total_sec = et - st
    loading_sec = st_after_model - st
    
    print(f"Elapsed time -- total: {total_sec:.1f} seconds (data & model loading: {loading_sec:.1f} seconds).")
    
    sys.exit(0)
