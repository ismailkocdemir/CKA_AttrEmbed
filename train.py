import os
import h5py
import math
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import itertools
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import utils.io as io
from utils.model import Model
from utils.constants import save_constants
import utils.pytorch_layers as pytorch_layers
from models.resnet import ResnetModel
from models.attribute_embeddings import AttributeEmbeddings
from models.conse import Conse
from dataset import Cifar100Dataset

def train_model(model,dataloaders,exp_const):
    model.attr_embed.load_embeddings(dataloaders['training'].dataset.labels)
    params = model.net.parameters()
    writer = SummaryWriter(exp_const.log_dir)
    
    lr = exp_const.lr
    if exp_const.optimizer == 'SGD':
        opt = optim.SGD(
            params,
            lr=lr,
            momentum=exp_const.momentum,
            weight_decay=1e-4)
    elif exp_const.optimizer == 'Adam':
        opt = optim.Adam(
            params,
            lr=lr,
            weight_decay=1e-4)
    elif exp_const.optimizer == 'Adagrad':
        opt = optim.Adagrad(
            params,
            lr=lr)
    else:
        assert(False), 'optimizer not implemented'

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    train_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=exp_const.milestones, gamma=0.2)
    iter_per_epoch = len(dataloaders['training'])
    warmup_scheduler = pytorch_layers.WarmUpLR(opt, iter_per_epoch * exp_const.warmup_epoch)

    step = 0
    if model.const.model_num is not None:
        step = model.const.model_num

    img_mean = Variable(torch.cuda.FloatTensor(model.img_mean))
    img_std = Variable(torch.cuda.FloatTensor(model.img_std))

    selected_model_results = None
    for epoch in range(exp_const.num_epochs):
        if epoch > exp_const.warmup_epoch:
            train_scheduler.step(epoch)
        for it,data in enumerate(dataloaders['training']):
            if epoch <= exp_const.warmup_epoch:
                warmup_scheduler.step()
            # Set mode
            model.net.train()
            if model.const.sim_loss:
                model.attr_embed.train()

            # Forward pass
            imgs = Variable(data['img'].cuda().float()/255)
            imgs = dataloaders['training'].dataset.normalize(
                imgs,
                img_mean,
                img_std)
            imgs = imgs.permute(0,3,1,2)
            label_idxs = Variable(data['label_idx'].cuda())

            logits, feats = model.net(imgs)

            log_items = {}
            targets = torch.linspace(0.4, 0.8, steps=len(feats)).requires_grad_(False)
            sim_loss_sum = 0.0
            for _idx, feat in enumerate(feats):
                cka, sim_loss = model.attr_embed(feat, label_idxs, targets[_idx])
                sim_loss_sum += sim_loss
                log_items["train/sim_loss_C{}".format(_idx+2)] = sim_loss.item()
                log_items["train/CKA_C{}".format(_idx+2)] = cka.item()

            # Computer loss
            ce_loss_weight = (step/10000) if step<10000 else 1
            loss = ce_loss_weight*criterion(logits,label_idxs) + \
                                    model.const.sim_loss*sim_loss_sum

            # Backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step%exp_const.log_step==0:
                writer.add_scalar('lr', opt.param_groups[0]['lr'], step)

                _,argmax = torch.max(logits,1)
                argmax = argmax.data.cpu().numpy()
                label_idxs_ = label_idxs.data.cpu().numpy()
                train_acc = np.mean(argmax==label_idxs_)*100

                log_items['train/loss'] = loss.item()
                log_items['train/acc'] = train_acc

                log_str = f'Epoch: {epoch} | Iter: {it} | Step: {step} | '
                for name,value in log_items.items():
                    log_str += '{}: {:.4f} | '.format(name,value)
                    writer.add_scalar(name,value,step)
                    
                print(log_str)
            
            if step%(10*exp_const.log_step)==0:
                print(f'Experiment: {exp_const.exp_name}')
                
            if step%exp_const.model_save_step==0:
                save_items = {
                    'net': model.net,
                    'attr_embed': model.attr_embed
                }

                for name,nn_model in save_items.items():
                    model_path = os.path.join(
                        exp_const.model_dir,
                        f'{name}_{step}')
                    torch.save(nn_model.state_dict(),model_path)

            if step%exp_const.val_step==0:
                eval_results = eval_model(
                    model,
                    dataloaders['test'],
                    exp_const,
                    step)
                print(eval_results)
                for name,value in eval_results.items():
                    log_str += '{}: {:.4f} | '.format(name,value)
                    writer.add_scalar(name,value,step)

                if selected_model_results is None:
                    selected_model_results = eval_results
                else:
                    if eval_results['val/acc'] >= \
                            selected_model_results['val/acc']:
                        selected_model_results = eval_results
                
                selected_model_results_json = os.path.join(
                    exp_const.exp_dir,
                    'selected_model_results.json')
                io.dump_json_object(
                    selected_model_results,
                    selected_model_results_json)

            writer.flush()
            step += 1
    writer.close()


def eval_model(model,dataloader,exp_const,step):
    # Set mode
    model.net.eval()
    model.attr_embed.eval()

    img_mean = Variable(torch.cuda.FloatTensor(model.img_mean))
    img_std = Variable(torch.cuda.FloatTensor(model.img_std))

    softmax = nn.Softmax(dim=1)

    correct = 0
    seen_correct_per_class = {l: 0 for l in dataloader.dataset.labels}
    sample_per_class = {l: 0 for l in dataloader.dataset.labels}
    sim_loss_all = [0.0]
    for it,data in enumerate(dataloader): #enumerate(tqdm(dataloader)):
        # Forward pass
        imgs = Variable(data['img'].cuda().float()/255)
        imgs = dataloader.dataset.normalize(
            imgs,
            img_mean,
            img_std)
        imgs = imgs.permute(0,3,1,2)
        
        logits,_ = model.net(imgs)
        '''
        label_idxs = Variable(data['label_idx'].cuda())
        for idx, feat in enumerate(feats):
            if len(sim_loss_all) < idx+1:
                sim_loss_all.insert(idx, 0.0)
            sim = model.attr_embed(feat, label_idxs)
            sim_loss_all[idx] = (sim_loss_all[idx]*it + sim)/(it+1)
        '''
        #label_idxs = label_idxs.data.cpu().numpy()
        gt_labels = data['label']
        prob = softmax(logits)
        prob = prob.data.cpu().numpy()
        prob_zero_seen = np.copy(prob)

        argmax_zero_seen = np.argmax(prob_zero_seen,1)
        for i in range(prob.shape[0]):
            pred_label = dataloader.dataset.labels[argmax_zero_seen[i]]
            gt_label = gt_labels[i]
            sample_per_class[gt_label] += 1
            if gt_label==pred_label:
                seen_correct_per_class[gt_label] += 1
   
    seen_acc = 0
    num_seen_classes = 0
    for l in dataloader.dataset.labels:
        seen_acc += (seen_correct_per_class[l] / sample_per_class[l])
        num_seen_classes += 1

    seen_acc = round(seen_acc*100 / num_seen_classes,4)
    
    eval_results = {
        'val/acc': seen_acc,
    }
    '''
    for _idx, sim_loss in enumerate(sim_loss_all):
        eval_results["val/sim_loss_L{}".format(_idx+1)] = sim_loss
    '''
    return eval_results


def main(exp_const,data_const,model_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    io.mkdir_if_not_exists(exp_const.log_dir)
    io.mkdir_if_not_exists(exp_const.model_dir)
    io.mkdir_if_not_exists(exp_const.vis_dir)
    save_constants(
        {'exp': exp_const,'data': data_const,'model': model_const},
        exp_const.exp_dir)
    
    print('Creating network ...')
    model = Model()
    model.const = model_const
    model.net = ResnetModel(model.const.net)
    model.attr_embed = AttributeEmbeddings(model.const.attr_embed)
    if model.const.model_num is not None:
        model.net.load_state_dict(torch.load(model.const.net_path))
        model.attr_embed.load_state_dict(
            torch.load(model.const.attr_embed_path))
    model.net.cuda()
    model.attr_embed.cuda()
    model.img_mean = np.array([0.5071, 0.4865, 0.4409])
    model.img_std = np.array([0.2673, 0.2564, 0.2762])
    model.to_file(os.path.join(exp_const.exp_dir,'model.txt'))

    print('Creating dataloader ...')
    dataloaders = {}
    for mode, subset in exp_const.subset.items():
        data_const = copy.deepcopy(data_const)
        if subset=='train':
            data_const.train = True
        else:
            data_const.train = False
        dataset = Cifar100Dataset(data_const)
        collate_fn = dataset.get_collate_fn()
        dataloaders[mode] = DataLoader(
            dataset,
            batch_size=exp_const.batch_size,
            shuffle=True,
            num_workers=exp_const.num_workers,
            collate_fn=collate_fn)

    train_model(model,dataloaders,exp_const)    