##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/Sha-Lab/FEAT
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Trainer for meta-train phase. """
import os.path as osp
import os
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler
from models.mtl import MtlLearner
from utils.misc import Averager, count_acc, compute_confidence_interval, ensure_path
from tensorboardX import SummaryWriter
from dataloader.dataset_loader import AdaptTrainer, AdaptValider
import pdb # pdb.set_trace()

class MetaTrainer(object):
    """The class that contains the code for the meta-train phase and meta-eval phase."""
    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        meta_base_dir = osp.join(log_base_dir, 'meta')
        if not osp.exists(meta_base_dir):
            os.mkdir(meta_base_dir)
        save_path = 'shot{}_step{}_gam{}_lr1{}_lr2{}_bs{}_epo{}_inner_lr{}_st{}_si{}'.format(args.shot,args.step_size,args.gamma,args.meta_lr1,args.meta_lr2,\
            args.num_batch,args.max_epoch,args.base_lr,args.update_step,args.step_size)
        args.save_path = '{}/{}'.format(meta_base_dir, save_path)
        ensure_path(args.save_path)

        # Set args to be shareable in the class
        self.args = args

        # Load meta-train set
        self.trainset = AdaptTrainer(self.args)
        self.train_sampler = CategoriesSampler(self.trainset.label, self.args.num_batch, self.args.shot, self.args.train_query)
        self.train_loader = DataLoader(dataset=self.trainset, batch_sampler=self.train_sampler, num_workers=8, pin_memory=True)

        # Load meta-val set
        self.valset = AdaptValider(self.args)
        self.val_sampler = CategoriesSampler(self.valset.label, self.args.num_batch, self.args.shot, self.args.val_query)
        self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=8, pin_memory=True)
        
        self.loc_split = self.trainset.loc_split

        # Build meta-transfer learning model
        self.model = MtlLearner(self.args)

        # Set optimizer 
        self.optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, self.model.encoder.parameters())}, \
            {'params': self.model.base_learner.parameters(), 'lr': self.args.meta_lr2}], lr=self.args.meta_lr1)
        # Set learning rate scheduler 
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma)        
        
        # load pretrained model without FC classifier
        self.model_dict = self.model.state_dict()
        if self.args.init_weights is not None:
            pretrained_dict = torch.load(self.args.init_weights)['params']
        else:
            pre_base_dir = osp.join(log_base_dir, 'pre')
            pre_save_path = 'batchsize' + str(args.pre_batch_size) + '_lr' + str(args.pre_lr) + '_gamma' + str(args.pre_gamma) + '_step' + \
                str(args.pre_step_size) + '_maxepoch' + str(args.pre_max_epoch)
            pre_save_path = pre_base_dir + '/' + pre_save_path
            pretrained_dict = torch.load(osp.join(pre_save_path, self.args.pre_stage+'.pth'))['params']
        pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.model_dict}
        # print(pretrained_dict.keys())
        self.model_dict.update(pretrained_dict)
        self.model.load_state_dict(self.model_dict)    

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()
        
    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """  
        torch.save(dict(params=self.model.state_dict()), osp.join(self.args.save_path, name + '.pth'))           

    def train(self):
        """The function for the meta-train phase."""

        # Set the meta-train log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0

        # Set global count to zero
        global_count = 0
        # Set tensorboardX
        writer = SummaryWriter(comment=self.args.save_path)
        
        # Start meta-train
        for epoch in range(1, self.args.max_epoch + 1):
            # Update learning rate
            self.lr_scheduler.step()
            # Set the model to train mode
            self.model.train()
            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc_averager = Averager()

            # Using tqdm to read samples from train loader
            tqdm_gen = tqdm.tqdm(self.train_loader)
            for i, batch in enumerate(tqdm_gen, 1):

                if torch.cuda.is_available():
                    data = batch[0].cuda()
                else:
                    data = batch[0]
                data_seg = []
                for idx,zone_key in enumerate(self.loc_split):
                    pos = self.loc_split[zone_key]
                    data_seg.append(F.upsample(data[:,:,:,int(pos['y']*640):int(pos['y']*640)+int(pos['h']*640)],size=[128,200],mode='bilinear'))

                label = batch[1]
                label_seg = []
                if torch.cuda.is_available():
                    for the_seg in label:
                        label_seg.append(the_seg.type(torch.cuda.LongTensor))
                else:
                    for the_seg in label:
                        label_seg.append(the_seg.type(torch.LongTensor))

                p = self.args.shot * len(self.train_sampler.opt_diag)
                data_shot, data_query, label_shot, label_query = [],[],[],[]
                for seg_idx in range(len(self.loc_split)):
                    data_shot.append(data_seg[seg_idx][:p])
                    data_query.append(data_seg[seg_idx][p:])
                    label_shot.append(label_seg[seg_idx][:p])
                    label_query.append(label_seg[seg_idx][p:])
                data_shot = torch.cat(data_shot,dim=0)
                data_query = torch.cat(data_query,dim=0)
                label_shot = torch.cat(label_shot,dim=0)
                label_query = torch.cat(label_query,dim=0)
                # pdb.set_trace()

                # Output logits for model
                logits = self.model((data_shot, label_shot, data_query))
                # Calculate meta-train loss
                loss = F.cross_entropy(logits, label_query)
                # Calculate meta-train accuracy
                acc = count_acc(logits, label_query)
                # Write the tensorboardX records
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)
                # Print loss and accuracy for this step
                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f}'.format(epoch, loss.item(), acc))

                # Add loss and accuracy for the averagers
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)

                # Loss backwards and optimizer updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Update the averagers
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()

            # Start validation for this epoch, set model to eval mode
            self.model.eval()

            # Set averager classes to record validation losses and accuracies
            val_loss_averager = Averager()
            val_acc_averager = Averager()
                
            # Print previous information
            if epoch % 10 == 0:
                print('Best Epoch {}, Best Val Acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
            # Run meta-validation
            for i, batch in enumerate(self.val_loader, 1):

                if torch.cuda.is_available():
                    data = batch[0].cuda()
                else:
                    data = batch[0]
                data_seg = []
                for idx,zone_key in enumerate(self.loc_split):
                    pos = self.loc_split[zone_key]
                    data_seg.append(F.upsample(data[:,:,:,int(pos['y']*640):int(pos['y']*640)+int(pos['h']*640)],size=[128,200],mode='bilinear'))

                label = batch[1]
                label_seg = []
                if torch.cuda.is_available():
                    for the_seg in label:
                        label_seg.append(the_seg.type(torch.cuda.LongTensor))
                else:
                    for the_seg in label:
                        label_seg.append(the_seg.type(torch.LongTensor))

                p = self.args.shot * len(self.val_sampler.opt_diag)
                data_shot, data_query, label_shot, label_query = [],[],[],[]
                for seg_idx in range(len(self.loc_split)):
                    data_shot.append(data_seg[seg_idx][:p])
                    data_query.append(data_seg[seg_idx][p:])
                    label_shot.append(label_seg[seg_idx][:p])
                    label_query.append(label_seg[seg_idx][p:])
                data_shot = torch.cat(data_shot,dim=0)
                data_query = torch.cat(data_query,dim=0)
                label_shot = torch.cat(label_shot,dim=0)
                label_query = torch.cat(label_query,dim=0)

                logits = self.model((data_shot, label_shot, data_query))
                loss = F.cross_entropy(logits, label_query)
                acc = count_acc(logits, label_query)
                # Add loss and accuracy for the averagers
                val_loss_averager.add(loss.item())
                val_acc_averager.add(acc)

            # Update validation averagers
            val_loss_averager = val_loss_averager.item()
            val_acc_averager = val_acc_averager.item()
            # Write the tensorboardX records
            writer.add_scalar('data/val_loss', float(val_loss_averager), epoch)
            writer.add_scalar('data/val_acc', float(val_acc_averager), epoch)       
            # Print loss and accuracy for this epoch
            print('Epoch {}, Val, Loss={:.4f} Acc={:.4f}'.format(epoch, val_loss_averager, val_acc_averager))

            # Update best saved model
            if val_acc_averager > trlog['max_acc']:
                trlog['max_acc'] = val_acc_averager
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc')
            # Save model every 10 epochs
            if epoch % 10 == 0:
                self.save_model('epoch'+str(epoch))

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)
            trlog['val_loss'].append(val_loss_averager)
            trlog['val_acc'].append(val_acc_averager)

            # Save log
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))
            
        writer.close()

    def eval(self):
        """The function for the meta-eval phase."""
        # Load the logs
        trlog = torch.load(osp.join(self.args.save_path, 'trlog'))
        self.args.setname = 'maxim' #maxim

        # Load meta-test set
        num_run = 100
        test_set = Dataset('test', self.args)
        sampler = CategoriesSampler(test_set.label, num_run, self.args.shot, self.args.val_query)
        loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
        loc_split = test_set.loc_split  # self.trainset.loc_split
        # del loc_split['zone1']
        # del loc_split['zone2']
        # del loc_split['zone3']
        print(loc_split)

        # Set test accuracy recorder
        test_acc_record = np.zeros((num_run,))

        # Load model for meta-test phase
        if self.args.eval_weights is not None:
            print('model load way 1')
            self.model.load_state_dict(torch.load(self.args.eval_weights)['params'])
        else:
            print('model load way 2')
            self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc' + '.pth'))['params'])
        # Set model to eval mode
        self.model.eval()

        # Set accuracy averager
        ave_acc = Averager()
        result_load = {}
            
        # Start meta-test
        for i, batch in enumerate(loader, 1):

            if torch.cuda.is_available():
                data = batch[0].cuda()
            else:
                data = batch[0]
            data_seg = []
            for idx,zone_key in enumerate(loc_split):
                pos = loc_split[zone_key]
                data_seg.append(F.upsample(data[:,:,:,int(pos['y']*640):int(pos['y']*640)+int(pos['h']*640)],size=[128,200],mode='bilinear'))

            label = batch[1]
            label_seg = []
            if torch.cuda.is_available():
                for the_seg in label:
                    label_seg.append(the_seg.type(torch.cuda.LongTensor))
            else:
                for the_seg in label:
                    label_seg.append(the_seg.type(torch.LongTensor))

            p = self.args.shot * len(sampler.opt_diag)
            if i%5 == 1:                
                data_shot, data_query, label_shot, label_query = [],[],[],[]
                for seg_idx in range(len(loc_split)):
                    data_shot.append(data_seg[seg_idx][:p])
                    data_query.append(data_seg[seg_idx][p:])
                    label_shot.append(label_seg[seg_idx][:p])
                    label_query.append(label_seg[seg_idx][p:])
                data_shot = torch.cat(data_shot,dim=0)
                data_query = torch.cat(data_query,dim=0)
                label_shot = torch.cat(label_shot,dim=0)
                label_query = torch.cat(label_query,dim=0)
            else:
                data_query,label_query = [],[]
                for seg_idx in range(len(loc_split)):
                    data_query.append(data_seg[seg_idx])
                    label_query.append(label_seg[seg_idx])
                data_query = torch.cat(data_query,dim=0)
                label_query = torch.cat(label_query,dim=0)

            logits = self.model((data_shot, label_shot, data_query))
            loss = F.cross_entropy(logits, label_query)
            acc = count_acc(logits, label_query)
            ave_acc.add(acc)
            test_acc_record[i-1] = acc

            pred = F.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()
            result_load['batch_%d'%(i)] = {'acc':acc, 'pred':pred, 'label':label_query.cpu().numpy()}
            
            if i % 10 == 0:
                print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

        adaptation_dir = './adaptation/'
        if not osp.exists(adaptation_dir):
            os.mkdir(adaptation_dir)
        set_adaptation = adaptation_dir + self.args.setname + '/'
        if not osp.exists(set_adaptation):
            os.mkdir(set_adaptation)

        result_load['acc'] = ave_acc.item() * 100
        model_path = 'shot_'+ str(self.args.shot) + '_' + self.args.pre_stage
        np.save(set_adaptation+model_path+'.npy', result_load)
            
        # Calculate the confidence interval, update the logs
        m, pm = compute_confidence_interval(test_acc_record)
        print('Val Best Epoch {}, Acc {:.4f}, Test Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'], ave_acc.item()))
        print('Test Acc {:.4f} + {:.4f}'.format(m, pm))
        print(test_acc_record)
        