import pickle
import os
import torch
import pandas as pd
from tqdm import tqdm
from src.eval import evaluate
#from src.utils import *
from src.parser import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from transformers import get_linear_schedule_with_warmup
from src.my_plotting import plotter
import matplotlib.pyplot as plt
import numpy as np
import wandb

import torch
#from torch.utils.tensorboard import SummaryWriter
import argparse
from xmlrpc.client import Boolean



#writer = SummaryWriter()

device = 'cuda'

def convert_to_windows(data, n_window):
    windows = list(torch.split(data, n_window))
    for i in range (n_window-windows[-1].shape[0]):
        windows[-1] = torch.cat((windows[-1], windows[-1][-1].unsqueeze(0)))
    return torch.stack(windows)

# parametri: point_global, point_contextual etc
def load_dataset(dataset, part=None):
    loader = [] 
    folder = 'DiffusionAE/processed/' + dataset

    for file in ['train', 'test', 'validation', 'labels', 'labels_validation']:
        if part is None:
            loader.append(np.load(os.path.join(folder, f'{file}.npy')))
        else:
            loader.append(np.load(os.path.join(folder, f'{part}_{file}.npy')))
    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    validation_loader = DataLoader(loader[2], batch_size=loader[2].shape[0])
    return train_loader, test_loader, validation_loader, loader[3], loader[4]

def load_model(training_mode, lr, window_size, p1, p2, dims, batch_size, noise_steps, denoise_steps):
    from models2 import Autoencoder_Diffusion, TransformerBasicBottleneckScaling, TransformerBasicv2Scaling, ConditionalDiffusionTrainingNetwork
    scheduler=None	
    model = None
    diffusion_training_net = ConditionalDiffusionTrainingNetwork(dims, int(window_size), batch_size, noise_steps, denoise_steps).float()
    diffusion_prediction_net = ConditionalDiffusionTrainingNetwork(dims, int(window_size), batch_size, noise_steps, denoise_steps, train=False).float()
    if training_mode == 'both':
        if args.model == 'Autoencoder_Diffusion':
            model = Autoencoder_Diffusion(dims, float(lr), int(window_size), p1, p2).float()
        elif args.model == 'TransformerBasicBottleneckScaling': 
            model = TransformerBasicBottleneckScaling(dims, float(lr), int(window_size), batch_size).float()
        else:
            model = TransformerBasicv2Scaling(dims, float(lr), int(window_size)).float()
        optimizer = torch.optim.Adam(list(model.parameters()) + list(diffusion_training_net.parameters()), lr=model.lr)
        # AE size
        # param_size = 0
        # for param in model.parameters():
        #     param_size += param.nelement() * param.element_size()
        # buffer_size = 0
        # for buffer in model.buffers():
        #     buffer_size += buffer.nelement() * buffer.element_size()
        # size_all_mb = (param_size + buffer_size) / 1024**2
        # print('model size: {:.3f}MB'.format(size_all_mb))
    else:
        optimizer = torch.optim.Adam(diffusion_training_net.parameters(), lr=float(lr))
        # DIFFUSION size
        # param_size = 0
        # for name, param in diffusion_training_net.named_parameters():
        #     param_size += param.nelement() * param.element_size()
        #     print(f'{name} {param.size()}')
        # buffer_size = 0
        # for buffer in diffusion_training_net.buffers():
        #     buffer_size += buffer.nelement() * buffer.element_size()
        # size_all_mb = (param_size + buffer_size) / 1024**2
        # print('diffusion size: {:.3f}MB'.format(size_all_mb))
    return model, diffusion_training_net, diffusion_prediction_net, optimizer, scheduler

CHECKPOINT_FOLDER = '/anomaly-mts/a-mts/checkpoints'
def save_model(model, experiment, diffusion_training_net, optimizer, scheduler, anomaly_score, epoch, diff_loss, ae_loss):
    folder = f'{CHECKPOINT_FOLDER}/{experiment}/'
    os.makedirs(folder, exist_ok=True)
    if model:
        file_path_model = f'{folder}/model.ckpt'
        torch.save({
            'epoch': epoch,
            'ae_loss': ae_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),}, file_path_model)
    file_path_diffusion = f'{folder}/diffusion.ckpt'
    torch.save({
        'epoch': epoch,
        'diffusion_loss': diff_loss,
        'model_state_dict': diffusion_training_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),}, file_path_diffusion)
    print('saved model at ' + folder)

def load_from_checkpoint(training_mode, experiment, model, diffusion_training_net):
    folder = f'{CHECKPOINT_FOLDER}/{experiment}'
    file_path_model = f'{folder}/model.ckpt'
    file_path_diffusion = f'{folder}/diffusion.ckpt'
    # load model
    if training_mode == 'both':
        checkpoint_model = torch.load(file_path_model)
        model.load_state_dict(checkpoint_model['model_state_dict'])
    else: 
        model = None
    # load diffusion
    checkpoint_diffusion = torch.load(file_path_diffusion)
    diffusion_training_net.load_state_dict(checkpoint_diffusion['model_state_dict'])
    return model, diffusion_training_net

def get_diffusion_sample(diffusion_prediction_net, conditioner, k):
    if k <= 1:
        return diffusion_prediction_net(conditioner)
    else:  
        diff_samples = []
        for _ in range(k):
            diff_samples.append(diffusion_prediction_net(conditioner))
        return torch.mean(torch.stack(diff_samples), axis = 0)

def backprop(epoch, model, diffusion_training_net, diffusion_prediction_net, data, diff_lambda, optimizer, scheduler, training_mode, anomaly_score, k, training = True):
    l = nn.MSELoss(reduction = 'none')
    data_x = torch.tensor(data, dtype=torch.float32); dataset = TensorDataset(data_x, data_x)
    bs = diffusion_training_net.batch_size if not model else model.batch
    dataloader = DataLoader(dataset, batch_size = bs)
    w_size = diffusion_training_net.window_size
    l1s, diff_losses, ae_losses = [], [], []
    samples = []
    # cleaned = []
    # original = []
    # all_mins = []
    # all_maxs = []
    if training:
        if training_mode == 'both':
            model.train()
        diffusion_training_net.train()
        for d, _ in dataloader:
            ##### Clean trend datset here 
            """mins = torch.min(d[:, :, 0], dim=1)
            maxs = torch.max(d[:, :, 0], dim=1)
            original.append(d)
            diffs = maxs[0] - mins[0]
            d = d[diffs < 0.04]
            cleaned.append(d)
            all_mins.append(mins)
            all_maxs.append(maxs)"""
            #####
            if args.model == 'Autoencoder_Diffusion':
                local_bs = d.shape[0]
                window = d.view(local_bs, -1)
            else:
                window = d
            window = window.to(device)
            if training_mode == 'both':
                if args.model == 'Autoencoder_Diffusion':
                    ae_reconstruction = model(window)
                else:
                    ae_reconstruction = model(window, window)
                # B x (feats * win)
                ae_loss = l(ae_reconstruction, window)
                ae_reconstruction = ae_reconstruction.reshape(-1, w_size, feats)
                # un tensor cu un element
                diffusion_loss, _ = diffusion_training_net(ae_reconstruction)
                ae_losses.append(torch.mean(ae_loss).item())
                diff_losses.append(torch.mean(diffusion_loss).item())
                if e < 5:
                    loss = torch.mean(ae_loss)
                else:
                    loss = diff_lambda * diffusion_loss + torch.mean(ae_loss)
            else:
                # diff only
                window = window.reshape(-1, w_size, feats)
                loss, _ = diffusion_training_net(window)
            l1s.append(loss.item())
            optimizer.zero_grad()
            loss.backward()                                                                                                     
            optimizer.step()
        tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
        tqdm.write(f'Epoch {epoch},\tAE = {np.mean(ae_losses)}')
        tqdm.write(f'Epoch {epoch},\tDiff = {np.mean(diff_losses)}')
        return np.mean(l1s), np.mean(ae_losses), np.mean(diff_losses)
    else:
        with torch.no_grad():
            if training_mode == 'both':
                model.eval()
            diffusion_prediction_net.load_state_dict(diffusion_training_net.state_dict())
            diffusion_prediction_net.eval()
            diffusion_training_net.eval()
            l1s = [] # scores
            sum_losses = []
            ae_losses = []
            diff_losses = []
            recons = []
            for d, _ in dataloader:
                if args.model == 'Autoencoder_Diffusion':
                    local_bs = d.shape[0]
                    window = d.view(local_bs, -1)
                else:
                    window = d
                window = window.to(device)
                window_reshaped = window.reshape(-1, w_size, feats)
                if training_mode == 'both':
                    if args.model == 'Autoencoder_Diffusion':
                        ae_reconstruction = model(window)
                    else:
                        ae_reconstruction = model(window, window)
                    ae_reconstruction_reshaped = ae_reconstruction.reshape(-1, w_size, feats)
                    recons.append(ae_reconstruction_reshaped)
                    ae_loss = l(ae_reconstruction, window)
                    ae_losses.append(torch.mean(ae_loss).item())
                    _, diff_sample = diffusion_prediction_net(ae_reconstruction_reshaped)
                    diff_sample = torch.squeeze(diff_sample, 1)
                    diffusion_loss = l(diff_sample, window_reshaped)
                    diffusion_loss = torch.mean(diffusion_loss).item()
                    sum_losses.append(torch.mean(ae_loss).item() + diffusion_loss)
                    diff_losses.append(diffusion_loss)
                    samples.append(diff_sample)
                    if anomaly_score == 'both': # 1
                        loss = l(diff_sample, ae_reconstruction_reshaped)
                    elif anomaly_score == 'diffusion': # 3
                        loss = l(diff_sample, window_reshaped)
                    elif anomaly_score == 'autoencoder': # 2
                        loss = l(ae_reconstruction, window)
                    elif anomaly_score == 'sum': # 4 = 2 + 3
                        loss = l(ae_reconstruction, window) + l(window, diff_sample)
                    elif anomaly_score == 'sum2': # 5 = 1 + 2
                        loss = l(diff_sample, ae_reconstruction) + l(ae_reconstruction, window)
                    elif anomaly_score == 'diffusion2': # 6 - 3 conditionat de gt
                        diff_sample = get_diffusion_sample(diffusion_prediction_net, window_reshaped, k)
                        loss = l(diff_sample, window_reshaped)
                else:
                    _, x_recon = diffusion_prediction_net(window_reshaped)
                    x_recon = torch.squeeze(x_recon, 1)
                    samples.append(x_recon)
                    loss = l(x_recon, window_reshaped)
                l1s.append(loss)
        if training_mode == 'both':
            return torch.cat(l1s).detach().cpu().numpy(), np.mean(sum_losses), np.mean(ae_losses), np.mean(diff_losses), torch.cat(samples).detach().cpu().numpy(), torch.cat(recons).detach().cpu().numpy()
        return torch.cat(l1s).detach().cpu().numpy(), np.mean(sum_losses), np.mean(ae_losses), np.mean(diff_losses), torch.cat(samples).detach().cpu().numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
    parser.add_argument('--dataset', 
                        metavar='-d', 
                        type=str, 
                        required=False,
                        default='point_global',
                        help="dataset"),
    parser.add_argument('--file', 
                        metavar='-f', 
                        type=str, 
                        required=False,
                        default=None,
                        help="dataset")
    parser.add_argument('--model', 
                        metavar='-m', 
                        type=str, 
                        required=False,
                        default='Autoencoder_Diffusion',
                        help="model name"),
    parser.add_argument('--training', 
                        metavar='-t', 
                        type=str, 
                        required=False,
                        default='both',
                        help="model to train"),
    parser.add_argument('--anomaly_score', 
                        metavar='-t', 
                        type=str, 
                        required=False,
                        default=None,
                        help="anomaly score"),
    parser.add_argument('--lr', 
                        metavar='-lr', 
                        type=str, 
                        required=False,
                        default='1e-4',
                        help="lerning rate"),
    parser.add_argument('--window_size', 
                        metavar='-ws', 
                        type=str, 
                        required=False,
                        default='10',
                        help="window size"),
    parser.add_argument('--p1', 
                        metavar='-p1', 
                        type=float, 
                        required=False,
                        default='1',
                        help="p1"),
    parser.add_argument('--p2', 
                        metavar='-p2', 
                        type=float, 
                        required=False,
                        default='1',
                        help="p2"),
    parser.add_argument('--k', 
                        metavar='-k', 
                        type=int, 
                        required=False,
                        default='1',
                        help="number of diff samples"),					
    parser.add_argument('--v', 
                        metavar='-v', 
                        type=bool, 
                        required=False,
                        default=False,
                        help="verbose"),
    # parser.add_argument('--test_only', 
    # 					metavar='-t', 
    # 					type=bool, 
    # 					required=False,
    # 					default=False,
    #                     help="test_only"),
    parser.add_argument('--batch_size', 
                        metavar='-t', 
                        type=int, 
                        required=False,
                        default=128,
                        help="batch_size"),
    parser.add_argument('--diff_lambda', 
                        metavar='-t', 
                        type=float, 
                        required=False,
                        default=0.1,
                        help="diff_lambda"),
    parser.add_argument('--noise_steps', 
                        metavar='-t', 
                        type=int, 
                        required=False,
                        default=100,
                        help="noise_steps"),
    parser.add_argument('--denoise_steps', 
                        metavar='-t', 
                        type=int, 
                        required=False,
                        default=10,
                        help="denoise_steps"),
    parser.add_argument('--group', 
                        metavar='-t', 
                        type=str, 
                        required=False,
                        default='search_smd',
                        help="wandb group"),
    parser.add_argument('--test_only', 
                        metavar='-t', 
                        type=bool, 
                        required=False,
                        default=False,
                        help="train new model or not"),
    parser.add_argument('--id', 
                        metavar='-t', 
                        type=int, 
                        required=False,
                        default=0,
                        help="experiment id for multiple runs"),
    args = parser.parse_args()

    config = {
    "dataset": args.dataset,
    "file": args.file,
    "training_mode": args.training, 
    "model": args.model,
    "learning_rate": float(args.lr),
    "window_size": int(args.window_size),
    "lambda": args.diff_lambda,
    "noise_steps":args.noise_steps,
    "batch_size": args.batch_size,
    }

    #anomaly_scores = ['diffusion']
    anomaly_scores = [args.anomaly_score]

    if args.training == 'diffusion':
        experiment = 'diffv4'
    elif args.model == 'Autoencoder_Diffusion':
        experiment = 'autoencoder_both'
    elif args.model == 'TransformerBasicBottleneckScaling':
        experiment = 'tr_bn_diffv4'
    else:
        experiment = 'tr_basic_diffv4'
    
    experiment += f'_{args.dataset}_{args.noise_steps}-{args.denoise_steps}_{args.diff_lambda}_{args.lr}_{args.batch_size}_{args.window_size}'

    if args.training == 'both':
        experiment += f'_{anomaly_scores[0]}_score' 
    #experiment += f'_{args.id}'   

    wandb.init(project="anomaly-mts", entity="yourname", config=config, group=args.group)
    wandb.run.name = experiment
    
    dataset_name = args.dataset
    part = None if not args.file else args.file
    training_mode = 'both' if not args.training else args.training
    anomaly_score = None if not args.anomaly_score else args.anomaly_score
    window_size = int(args.window_size)
    synthetic_datasets = ['point_global', 'point_contextual', 'pattern_shapelet', 'pattern_seasonal', 'pattern_trend', 'all_types', 'pattern_trendv2']
    
    train_loader, test_loader, validation_loader, labels, validation_labels = load_dataset(dataset_name, part)
    model, diffusion_training_net, diffusion_prediction_net, optimizer, scheduler = \
                        load_model(training_mode ,args.lr, args.window_size, args.p1, args.p2, labels.shape[1], args.batch_size, args.noise_steps, args.denoise_steps)
    if model:
        model = model.to(device)
            
    diffusion_training_net = diffusion_training_net.to(device)
    diffusion_prediction_net = diffusion_prediction_net.to(device)

    trainD, testD, validationD = next(iter(train_loader)), next(iter(test_loader)), next(iter(validation_loader))
    trainO, testO, validationO = trainD, testD, validationD
    if args.v:
        print(f'\ntrainD.shape: {trainD.shape}')
        print(f'testD.shape: {testD.shape}')
        print(f'validationD.shape: {validationD.shape}')
        print(f'labels.shape: {labels.shape}')
    
    feats=labels.shape[1]    

    trainD, testD, validationD = convert_to_windows(trainD, window_size), convert_to_windows(testD, window_size), convert_to_windows(validationD, window_size)
    #num_epochs = 500 if args.dataset in synthetic_datasets else 100 
    num_epochs = 500   

    # while labels.shape[0]%window_size!=0:
    #     labels=np.concatenate((labels, np.expand_dims(labels[-1], 0)), axis=0)

    # while validation_labels.shape[0]%window_size!=0:
    #     validation_labels=np.concatenate((labels, np.expand_dims(labels[-1], 0)), axis=0)    
    # print('training')
    epoch = -1

    e = epoch + 1; start = time()
    max_roc_scores = [[0, 0, 0]] * 6
    max_f1_scores = [[0, 0, 0]] * 6
    roc_scores = []
    f1_scores = []
    f1_max = 0
    roc_max = 0
    validation_thresh = 0
    # anomaly_scores = ['diffusion']
    #alpha = 0
    if not args.test_only:
        for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
            train_loss, ae_loss, diff_loss = backprop(e, model, diffusion_training_net, diffusion_prediction_net, trainD, args.diff_lambda, optimizer, scheduler, training_mode, anomaly_score, args.k)
            wandb.log({
                'sum_loss_train': train_loss,
                'ae_loss_train': ae_loss,
                'diff_loss_train': diff_loss,
                'epoch': e
            }, step=e)
            if training_mode == 'both':
                for idx, a_score in enumerate(anomaly_scores):
                    if ae_loss + diff_loss < 0.15:
                        loss0, val_loss, ae_loss_val, diff_loss_val, samples, recons = backprop(e, model, diffusion_training_net, diffusion_prediction_net, validationD, args.diff_lambda, optimizer, scheduler, training_mode, a_score, args.k, training=False)
                        if idx == 0:    
                            wandb.log({
                                'sum_loss_val': val_loss,
                                'ae_loss_val': ae_loss_val,
                                'diff_loss_val': diff_loss_val,
                                'epoch': e
                            }, step=e)
                        loss0 = loss0.reshape(-1,feats)

                        lossFinal = np.mean(np.array(loss0), axis=1)
                        labelsFinal = (np.sum(validation_labels, axis=1) >= 1) + 0

                        result, fprs, tprs = evaluate(lossFinal, labelsFinal)
                        result_roc = result["ROC/AUC"]
                        result_f1 = result["f1"]
                        wandb.log({'roc': result_roc, 'f1': result_f1}, step=e)
                        if result_roc > max_roc_scores[idx][0]:
                            max_roc_scores[idx] = [result_roc, result_f1, e]
                            wandb.run.summary["f1_for_best_roc"] = result_f1
                            wandb.run.summary["best_roc"] = result_roc
                            wandb.run.summary["best_roc_epoch"] = e
                        if result_f1 > max_f1_scores[idx][1]:
                            max_f1_scores[idx] = [result_roc, result_f1, e]
                            save_model(model, experiment, diffusion_training_net, optimizer, None, a_score, e, diff_loss, ae_loss)
                            validation_thresh = result['threshold']
                            wandb.run.summary["best_f1"] = result_f1
                            wandb.run.summary["roc_for_best_f1"] = result_roc
                            wandb.run.summary["best_roc_epoch"] = e
                            wandb.run.summary["best_f1_epoch"] = e
                            wandb.run.summary["f1_pa"] = result['f1_max'] 
                            wandb.run.summary["roc_pa"] = result['roc_max']
                        if e % 100 == 0: 
                            for dim in range(0, feats):
                                fig = plotter(experiment, a_score, validationD.reshape(-1, feats), lossFinal, labelsFinal, result, recons.reshape(-1, feats), samples.reshape(-1, feats), None, dim=dim, plot_test=True, epoch=e)
                        if args.v:
                            print(str(e) + ' ROC: ' + str(result_roc) + ' F1: ' + str(result_f1) + '\n')
                    
            else:
                if train_loss < 0.15:
                    loss0, _, _, val_loss, samples = backprop(e, model, diffusion_training_net, diffusion_prediction_net, validationD, args.diff_lambda, optimizer, scheduler, training_mode, args.anomaly_score, args.k, training=False)
                    wandb.log({'val_loss': loss0.mean(), 'epoch': e}, step=e)
                    loss0 = loss0.reshape(-1,feats)
                    lossFinal = np.mean(np.array(loss0), axis=1)
                    labelsFinal = (np.sum(validation_labels, axis=1) >= 1) + 0
                    result, fprs, tprs = evaluate(lossFinal, labelsFinal)
                    result_roc = result["ROC/AUC"]
                    result_f1 = result["f1"]
                    wandb.log({'roc': result_roc, 'f1': result_f1}, step=e)
                    if result_f1 > f1_max:
                        save_model(None, experiment, diffusion_prediction_net, optimizer, None, -1, e, train_loss, None)
                        f1_max = result_f1
                        validation_thresh = result['threshold']
                        wandb.run.summary["best_f1"] = f1_max
                        wandb.run.summary["roc_for_best_f1"] = result_roc
                        wandb.run.summary["best_f1_epoch"] = e
                        wandb.run.summary["validation_thresh"] = validation_thresh
                    if result_roc > roc_max:
                        roc_max = result_roc 
                        wandb.run.summary["f1_for_best_roc"] = result_f1
                        wandb.run.summary["best_roc"] = roc_max
                        wandb.run.summary["best_roc_epoch"] = e
                    wandb.log({'roc': result_roc, 'f1': result_f1}, step=e)
                    if e % 100 == 0:
                        for dim in range(0, feats):
                            plotter(f'{experiment}_VAL', args.dataset, validationD.reshape(-1, feats), lossFinal, labelsFinal, result, None, samples.reshape(-1, feats), None, dim=dim, plot_test=True, epoch=e)
                    if args.v:
                        print(f"testing loss #{e}: {loss0.mean()}")
                        # print(f"training loss #{e}: {loss1.mean()}")
                        print(f"final ROC #{e}: {result_roc}")
                        print(f"F1 #{e}: {result_f1}")

    # TEST ON TEST SET
    #load model from checkpoint
    model, diffusion_training_net, diffusion_prediction_net, optimizer, scheduler = \
                        load_model(training_mode ,args.lr, args.window_size, args.p1, args.p2, labels.shape[1], args.batch_size, args.noise_steps, args.denoise_steps)
    model, diffusion_training_net = load_from_checkpoint(training_mode, experiment, model, diffusion_training_net)
    if model:
        model = model.to(device)
            
    diffusion_training_net = diffusion_training_net.to(device)
    diffusion_prediction_net = diffusion_prediction_net.to(device)
    # pass test set through the model
    if model:
        if args.test_only:
        #test again on val for double check + get best thresh on validation set to use for test
            loss0, val_loss, ae_loss_val, diff_loss_val, samples, recons = backprop(e, model, diffusion_training_net, diffusion_prediction_net, validationD, args.diff_lambda, optimizer, scheduler, training_mode, args.anomaly_score, args.k, training=False)
            loss0 = loss0.reshape(-1,feats)

            lossFinal = np.mean(np.array(loss0), axis=1)
            # lossFinal = np.max(np.array(loss0), axis=1)
            labelsFinal = (np.sum(validation_labels, axis=1) >= 1) + 0

            result, fprs, tprs = evaluate(lossFinal, labelsFinal)
            validation_thresh = result['threshold']
            result_roc = result["ROC/AUC"]
            result_f1 = result["f1"]
            wandb.run.summary["f1_val"] = result_f1
            wandb.run.summary["roc_val"] = result_roc
            wandb.run.summary["f1_pa_val"] = result['f1_max'] 
            wandb.run.summary["roc_pa_val"] = result['roc_max']
            wandb.run.summary["val_loss"] = val_loss
            wandb.run.summary["ae_loss_val"] = ae_loss_val
            wandb.run.summary["diff_loss_val"] = diff_loss_val

            # for dim in range(0, feats):
            #     fig = plotter(f'{experiment}_VAL', args.anomaly_score, validationD.reshape(-1, feats), lossFinal, labelsFinal, result, recons.reshape(-1, feats), samples.reshape(-1, feats), None, dim=dim, plot_test=True, epoch=e)

        loss0, test_loss, ae_loss_test, diff_loss_test, samples, recons = backprop(e, model, diffusion_training_net, diffusion_prediction_net, testD, args.diff_lambda, optimizer, scheduler, training_mode, args.anomaly_score, args.k, training=False)
        loss0 = loss0.reshape(-1,feats)

        lossFinal = np.mean(np.array(loss0), axis=1)
        np.save(f'../../{args.dataset}_{args.anomaly_score}_score_scores.npy', lossFinal)
        np.save(f'../../{args.dataset}_{args.anomaly_score}_score_recons.npy', samples)
        # np.save('/root/Diff-Anomaly/TranAD/plots_for_paper/shapelet_scores_for_example.npy', lossFinal)
        # lossFinal = np.max(np.array(loss0), axis=1)
        labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
        #validation_thresh = 0.0019
        result = evaluate(lossFinal, labelsFinal, validation_thresh=validation_thresh)
        result_roc = result["ROC/AUC"]
        result_f1 = result["f1"]
        wandb.run.summary["f1_test"] = result_f1
        wandb.run.summary["roc_test"] = result_roc
        wandb.run.summary["f1_pa_test"] = result['f1_max'] 
        #wandb.run.summary["roc_pa_test"] = result['roc_max']
        wandb.run.summary["test_loss"] = test_loss
        wandb.run.summary["ae_loss_test"] = ae_loss_test
        wandb.run.summary["diff_loss_test"] = diff_loss_test
        wandb.run.summary["validation_thresh"] = validation_thresh

        #for dim in range(0, feats):
        #    fig = plotter(f'{experiment}_TEST', args.anomaly_score, testD.reshape(-1, feats), lossFinal, labelsFinal, result, recons.reshape(-1, feats), samples.reshape(-1, feats), None, dim=dim, plot_test=True, epoch=e)
        
    else: 
        if args.test_only:
            loss0, _, _, val_loss, samples = backprop(e, model, diffusion_training_net, diffusion_prediction_net, validationD, args.diff_lambda, optimizer, scheduler, training_mode, args.anomaly_score, args.k, training=False)
            loss0 = loss0.reshape(-1,feats)

            lossFinal = np.mean(np.array(loss0), axis=1)
            labelsFinal = (np.sum(validation_labels, axis=1) >= 1) + 0

            result, fprs, tprs = evaluate(lossFinal, labelsFinal)
            result_roc = result["ROC/AUC"]
            result_f1 = result["f1"]
            validation_thresh = result['threshold']
            wandb.run.summary["f1_val"] = result_f1
            wandb.run.summary["roc_val"] = result_roc
            wandb.run.summary["f1_pa_val"] = result['f1_max'] 
            #wandb.run.summary["roc_pa_val"] = result['roc_max']
            wandb.run.summary["val_loss"] = val_loss
            wandb.run.summary["validation_thresh"] = validation_thresh
            #for dim in range(0, feats):
            #    plotter(f'{experiment}_VAL', args.dataset, validationD.reshape(-1, feats), lossFinal, labelsFinal, result, None, samples.reshape(-1, feats), None, dim=dim, plot_test=True, epoch=e)
        loss0, _, _, test_loss, samples = backprop(e, model, diffusion_training_net, diffusion_prediction_net, testD, args.diff_lambda, optimizer, scheduler, training_mode, args.anomaly_score, args.k, training=False)
        loss0 = loss0.reshape(-1,feats)

        lossFinal = np.mean(np.array(loss0), axis=1)
        np.save(f'../../{args.dataset}_diff_only_scores.npy', lossFinal)
        np.save(f'../../{args.dataset}_diff_only_recons.npy', samples)
        labelsFinal = (np.sum(labels, axis=1) >= 1) + 0

        result = evaluate(lossFinal, labelsFinal, validation_thresh=validation_thresh)
        result_roc = result["ROC/AUC"]
        result_f1 = result["f1"]
        #for dim in range(0, feats):
        #    plotter(f'{experiment}_TEST', args.dataset, testD.reshape(-1, feats), lossFinal, labelsFinal, result, None, samples.reshape(-1, feats), None, dim=dim, plot_test=True, epoch=e)
        wandb.run.summary["f1_test"] = result_f1
        wandb.run.summary["roc_test" ] = result_roc
        wandb.run.summary["f1_pa_test"] = result['f1_max'] 
        wandb.run.summary["roc_pa_test"] = result['roc_max']
        wandb.run.summary["test_loss"] = test_loss
    
    wandb.finish()  
