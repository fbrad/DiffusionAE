import pickle
import os
import torch
import pandas as pd
from tqdm import tqdm
from src.eval import evaluate
from src.utils import *
from src.parser import args
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
from src.my_plotting import plotter
from transformers import get_linear_schedule_with_warmup
import wandb

import matplotlib.pyplot as plt

import torch
#from torch.utils.tensorboard import SummaryWriter

#writer = SummaryWriter()

device = 'cuda'
#feats = 5 


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


CHECKPOINT_FOLDER = '../../../a-mts/checkpoints'
def save_model(model, experiment, optimizer, scheduler, epoch):
    folder = f'{CHECKPOINT_FOLDER}/{experiment}/'
    os.makedirs(folder, exist_ok=True)
    if model:
        file_path_model = f'{folder}/model.ckpt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),}, file_path_model)
    print('saved model at ' + folder)

def load_from_checkpoint(model, experiment):
    folder = f'{CHECKPOINT_FOLDER}/{experiment}'
    file_path_model = f'{folder}/model.ckpt'
    # load model
    checkpoint_model = torch.load(file_path_model)
    model.load_state_dict(checkpoint_model['model_state_dict'])
    return model


def load_model(model_name, lr, window_size, dims, batch_size, p1, p2):
    import models2
    model_class = getattr(models, model_name)
    model = model_class(dims, float(lr), int(window_size), batch_size).double()
    optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
    
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    epoch = -1; accuracy_list = []
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    return model, optimizer,epoch, accuracy_list


def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):
    l = nn.MSELoss(reduction = 'none')
    data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x)
    #bs = model.batch if training else len(data)
    bs = model.batch 
    dataloader = DataLoader(dataset, batch_size = bs)
    n = epoch + 1; w_size = model.n_window
    l1s, l2s = [], []
    if training:
        model.train()
        for d, _ in dataloader:
            d = d.to(device)
            local_bs = d.shape[0]
            # mins = torch.min(d[:, :, 0], dim=1)
            # maxs = torch.max(d[:, :, 0], dim=1)
            # diffs = maxs[0] - mins[0]
            # d = d[diffs < 0.04]
            # if d.shape[0] == 0:
            # 	pass

            #new_d = [d[i] for i in range(0, local_bs) if diffs[i] > 0.03]
            #window = d.view(-1, local_bs)
            # ws x bs x nr_feats
            #print('window shape', window.shape)
            #z = model(window, window).squeeze()
            window = d
            window = window.to(device)
            z = model(window, window)
            #window = d.view(local_bs, -1)
            #z = model(window)
            
            
            #print(type(z))
            #print('z shape', z.shape)
            l1 = l(z, window)
            # l1s.append(torch.mean(l1).item())
            #l1s.append(torch.sum(torch.mean(l1, dim=1)).item())
            # mask = torch.tensor([1, 1, 1, 1, 10], dtype=torch.int64)
            #mask = torch.unsqueeze(torch.unsqueeze(mask, 0), 0)
            # mask = mask.view(1, 1, -1).to(device)
            # loss = torch.mean(l1 * mask)
            #print(l1.shape)
            loss = torch.mean(l1)
            l1s.append(loss.item())
            #loss = torch.sum(torch.mean(l1, dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
        return np.mean(l1s), optimizer.param_groups[0]['lr']
    else:
        with torch.no_grad():
            model.eval()
            l1s = []
            recons = []
            for d, _ in dataloader:
                d = d.to(device)
                local_bs = d.shape[0]
                #window = d.view(-1, local_bs)
                #z = model(window, window).squeeze()

                window = d
                window = window.to(device)
                z = model(window, window)
                recons.append(z)
                #window = d.view(local_bs, -1)
                #z = model(window)
                #recons.append(z)
                loss = l(z, window)
                l1s.append(loss)
                # loss.shape = batch x win_size x nr_feats, append 
                #l1s.append(loss.reshape(-1, loss.shape[2]).detach().cpu())
            #print(loss.shape)
        return torch.cat(l1s).detach().cpu().numpy(), torch.cat(recons).detach().cpu().numpy()


if __name__ == '__main__':
    config = {
    "dataset": args.dataset,
    "file": args.file,
    "model": args.model,
    "learning_rate": float(args.lr),
    "window_size": int(args.window_size),
    "batch_size": args.batch_size,
    }
    
    #experiment = f'tr_bn_only_{args.dataset}_{args.lr}_{args.batch_size}_{args.window_size}_{args.id}'
    experiment = f'tr_bn_only_{args.dataset}_{args.lr}_{args.batch_size}_{args.window_size}'


    wandb.init(project="anomaly-mts", entity="yourname", config=config, group=args.group)
    wandb.run.name = experiment
    dataset_name = args.dataset
    window_size = int(args.window_size)
    part = None if not args.file else args.file


    train_loader, test_loader, validation_loader, labels, validation_labels = load_dataset(dataset_name, part)
    model, optimizer, epoch, accuracy_list = load_model(args.model, args.lr, args.window_size, labels.shape[1], args.batch_size, args.p1, args.p2)
    model = model.to(device)
    ## Prepare data
    trainD, testD, validationD = next(iter(train_loader)), next(iter(test_loader)), next(iter(validation_loader))
    trainO, testO, validationO = trainD, testD, validationD

    trainD, testD, validationD = convert_to_windows(trainD, window_size), convert_to_windows(testD, window_size), convert_to_windows(validationD, window_size)

    num_epochs = 500

    len_dataloder = len(trainD) // model.batch
    if len(trainD) % model.batch:
        len_dataloder += 1
    
    num_training_steps = len_dataloder * num_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, 0.1*num_training_steps, num_training_steps)	
    
    print(testD.shape)
    print(labels.shape)

    # while labels.shape[0]%int(args.window_size)!=0:
    #     labels=np.concatenate((labels, np.expand_dims(labels[-1], 0)), axis=0)
    print('training')

    ### Training phase
    print('training')
    e = epoch + 1; start = time()
    roc_scores = []
    f1_scores = []
    max_f1 = 0
    max_roc = 0
    if not args.test_only:
        for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
            lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
            #accuracy_list.append((lossT, lr))

            loss0, recons = backprop(0, model, validationD, validationO, optimizer, scheduler, training=False)
            wandb.log({
                'sum_loss_train': lossT, 
                'sum_loss_val': loss0.mean(),
                'epoch': e
            }, step=e)
            #loss1, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
            loss0 = loss0.reshape(-1,labels.shape[1])
            #loss1=loss1.reshape(-1,loss1.shape[2])
            lossFinal = np.mean(np.array(loss0), axis=1)
            labelsFinal = (np.sum(validation_labels, axis=1) >= 1) + 0
            result, _, _ = evaluate(lossFinal, labelsFinal)
            result_roc = result["ROC/AUC"]
            result_f1 = result["f1"]
            roc_scores.append(result_roc)
            f1_scores.append(result_f1)
            wandb.log({'roc': result_roc, 'f1': result_f1})
            if result_f1 > max_f1:
                max_f1 = result_f1
                validation_thresh = result['threshold']
                save_model(model, experiment, optimizer, scheduler, e)
                wandb.run.summary["best_f1"] = result_f1
                wandb.run.summary["roc_for_best_f1"] = result_roc
                wandb.run.summary["best_roc_epoch"] = e
                wandb.run.summary["best_f1_epoch"] = e
                wandb.run.summary["f1_pa"] = result['f1_max'] 
                wandb.run.summary["roc_pa"] = result['roc_max']
                wandb.run.summary["validation_thresh"] = validation_thresh
            if result_roc > max_roc:
                max_roc = result_roc
                wandb.run.summary["f1_for_best_roc"] = result_f1
                wandb.run.summary["best_roc"] = result_roc
                wandb.run.summary["best_roc_epoch"] = e

            if e % 100 == 0:
                for dim in range(0, labels.shape[1]):
                    plotter(experiment, None, validationO, lossFinal, labelsFinal, result, recons.reshape(-1, labels.shape[1]), None, None, dim=dim, plot_test=True, epoch=e)
            print(f"final ROC #{e}: {result_roc}")
            print(f"final F1 #{e}: {result_f1}")
    
    # TEST ON TEST SET
    # load from checkpoint
    model, _, _, _ = load_model(args.model, args.lr, args.window_size, labels.shape[1], args.batch_size, args.p1, args.p2)
    model = load_from_checkpoint(model, experiment)
    model = model.to(device)
    if args.test_only:
        loss0, recons = backprop(0, model, validationD, validationO, optimizer, scheduler, training=False)
        loss0 = loss0.reshape(-1,labels.shape[1])
        lossFinal = np.mean(np.array(loss0), axis=1)
        labelsFinal = (np.sum(validation_labels, axis=1) >= 1) + 0
        result, _, _ = evaluate(lossFinal, labelsFinal)
        result_roc = result["ROC/AUC"]
        result_f1 = result["f1"]
        validation_thresh = result['threshold']

        wandb.run.summary["f1_val"] = result_f1
        wandb.run.summary["roc_val"] = result_roc
        wandb.run.summary["f1_pa_val"] = result['f1_max'] 
        wandb.run.summary["roc_pa_val"] = result['roc_max']
        wandb.run.summary["val_loss"] = loss0.mean()    
        wandb.run.summary["validation_thresh"] = validation_thresh
        #for dim in range(0, labels.shape[1]):
        #    plotter(experiment, None, validationO, lossFinal, labelsFinal, result, recons.reshape(-1, labels.shape[1]), None, None, dim=dim, plot_test=True, epoch=e)
    # pass test set through the model
    loss0, recons = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
    loss0 = loss0.reshape(-1,labels.shape[1])
    lossFinal = np.mean(np.array(loss0), axis=1)

    np.save(f'../../{args.dataset}_ae_scores.npy', lossFinal)
    np.save(f'../../{args.dataset}_ae_recons.npy', recons)

    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    result = evaluate(lossFinal, labelsFinal, validation_thresh=validation_thresh)
    result_roc = result["ROC/AUC"]
    result_f1 = result["f1"]

    wandb.run.summary["f1_test"] = result_f1
    wandb.run.summary["roc_test"] = result_roc
    wandb.run.summary["f1_pa_test"] = result['f1_max'] 
    wandb.run.summary["roc_pa_test"] = result['roc_max']
    wandb.run.summary["test_loss"] = loss0.mean()    
    #for dim in range(0, labels.shape[1]):
    #    plotter(experiment, None, testO, lossFinal, labelsFinal, result, recons.reshape(-1, labels.shape[1]), None, None, dim=dim, plot_test=True, epoch=e)
    wandb.finish()
