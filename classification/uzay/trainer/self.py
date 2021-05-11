##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Trainer for pretrain phase. """
import os.path as osp
import os
import tqdm
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from models.NetworkPre import FeatureNet
from utils.misc import Averager, count_acc, ensure_path
from tensorboardX import SummaryWriter
from dataloader.selflearn_loader import SelfLoader as Dataset
import pdb
import numpy as np
import torch.nn as nn
import torchvision.models as models

sobel_x = torch.from_numpy(np.array([[1,0,-1],[2,0,-2],[1,0,-1]])).float().unsqueeze(0).unsqueeze(0)
sobel_y = torch.from_numpy(np.array([[1,2,1],[0,0,0],[-1,-2,-1]])).float().unsqueeze(0).unsqueeze(0)
if torch.cuda.is_available():
    sobel_x = sobel_x.cuda()
    sobel_y = sobel_y.cuda()
    
class SelfTrainer(object):
    """The class that contains the code for the pretrain phase."""
    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        pre_base_dir = osp.join(log_base_dir, 'self')
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)

        save_path = '{}_bs{}_lr{}_gam{}_epo{}_step{}_cls{}'.format(args.model_type,args.pre_batch_size,args.pre_lr,args.pre_gamma,args.pre_max_epoch,args.pre_step_size,args.cls_wgt)
        save_path = '{}_init'.format(save_path) if args.init == 1 else save_path
        args.save_path = '{}/{}'.format(pre_base_dir,save_path)
        ensure_path(args.save_path)

        # Set args to be shareable in the class
        self.args = args

        # Load pretrain set
        self.trainset = Dataset('train', self.args, train_aug=True)
        self.train_loader = DataLoader(dataset=self.trainset, batch_size=args.pre_batch_size, shuffle=True, num_workers=8, pin_memory=True)
        # Load meta-val set
        self.valset = Dataset('val', self.args, train_aug=False)
        self.val_loader = DataLoader(dataset=self.valset, batch_size=args.pre_batch_size, shuffle=True, num_workers=8, pin_memory=True)
        self.loc_split = self.trainset.loc_split
        
        # Build pretrain model
        self.model = FeatureNet(self.args, mode='self', n_class=2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.pre_lr, betas=(0.9, 0.995), weight_decay=0.0005)
        # Set learning rate scheduler 
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.pre_step_size, gamma=self.args.pre_gamma)

        if args.init == 1 and args.model_type == 'ResNet18ORI':
            pretrained_dict = models.resnet18(pretrained=True)
            pretrained_dict = pretrained_dict.state_dict()
            del pretrained_dict['fc.weight'],pretrained_dict['fc.bias']
            pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
            print('Loading the model pretrained on ImageNet1K')
            model_dict = self.model.state_dict()
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            self.model.eval()
        else:
            print('Training From Scratch')

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()
        
    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """  
        torch.save(dict(params=self.model.encoder.state_dict()), osp.join(self.args.save_path, name + '.pth'))

    def save_model_full(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """  
        torch.save(dict(params=self.model.state_dict()), osp.join(self.args.save_path, name + '_full.pth'))
        
    def train(self):
        """The function for the pre-train phase."""

        # Set the pretrain log
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
        writer = SummaryWriter(self.args.save_path)
        
        # Start pretrain
        for epoch in range(1, self.args.pre_max_epoch + 1):
            # Update learning rate
            self.lr_scheduler.step()
            # Set the model to train mode
            self.model.train()
            self.model.mode = 'self'
            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc_averager = Averager()
                
            # Using tqdm to read samples from train loader
            tqdm_gen = tqdm.tqdm(self.train_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                # Update global count number 
                global_count = global_count + 1
                if torch.cuda.is_available():
                    data = batch[0].cuda()
                    gray = batch[2].cuda()
                else:
                    data = batch[0]
                    gray = batch[2]

                label_seg = torch.cat(batch[1])
                if torch.cuda.is_available():
                    label_seg = label_seg.type(torch.cuda.LongTensor)
                else:
                    label_seg = label_seg.type(torch.LongTensor)

                data_seg,gray_seg = [],[]
                for idx,zone_key in enumerate(self.loc_split):
                    pos = self.loc_split[zone_key]
                    data_seg.append(F.upsample(data[:,:,:,int(pos['y']*320):int(pos['y']*320)+int(pos['h']*320)],size=[64,100],mode='bilinear'))
                    gray_seg.append(F.upsample(gray[:,:,:,int(pos['y']*320):int(pos['y']*320)+int(pos['h']*320)],size=[64,100],mode='bilinear'))
                
                # Output logits for model
                x = torch.cat(data_seg)
                x_180 = x.flip(2).flip(3)
                x_augmentation = torch.cat((x, x_180),0)
                y_augmentation = label_seg.repeat(2)
                
                z = torch.cat(gray_seg)
                z_180 = z.flip(2).flip(3)
                z_augmentation = torch.cat((z, z_180),0)
                recon,logits = self.model(x_augmentation)
                # Calculate train loss
                loss_cls = F.cross_entropy(logits, y_augmentation)
                loss_recon = self.Loss_MSE_Recon(recon,z_augmentation,y_augmentation)
                loss = self.args.cls_wgt * loss_cls + loss_recon
                # Calculate train accuracy
                acc = 1.0-loss_recon.item()
                # Write the tensorboardX records
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)
                writer.add_scalar('data/epoch', float(epoch), global_count)
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
            self.model.mode = 'self'

            # Set averager classes to record validation losses and accuracies
            val_loss_averager = Averager()
            val_acc_averager = Averager()
              
            # Print previous information  
            if epoch % 10 == 0:
                print('Best Epoch {}, Best Val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
            # Run meta-validation

            tqdm_gen = tqdm.tqdm(self.val_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                if torch.cuda.is_available():
                    data = batch[0].cuda()
                    gray = batch[2].cuda()
                else:
                    data = batch[0]
                    gray = batch[2]

                label_seg = torch.cat(batch[1])
                if torch.cuda.is_available():
                    label_seg = label_seg.type(torch.cuda.LongTensor)
                else:
                    label_seg = label_seg.type(torch.LongTensor)

                data_seg,gray_seg = [],[]
                for idx,zone_key in enumerate(self.loc_split):
                    pos = self.loc_split[zone_key]
                    data_seg.append(F.upsample(data[:,:,:,int(pos['y']*320):int(pos['y']*320)+int(pos['h']*320)],size=[64,100],mode='bilinear'))
                    gray_seg.append(F.upsample(gray[:,:,:,int(pos['y']*320):int(pos['y']*320)+int(pos['h']*320)],size=[64,100],mode='bilinear'))

                # Output logits for model
                # Output logits for model
                recon,_ = self.model(torch.cat(data_seg))
                # Calculate train loss
                loss = self.Loss_MSE_Recon(recon,torch.cat(gray_seg),label_seg)
                # loss = F.cross_entropy(logits, label_seg)
                # Calculate train accuracy
                # acc = count_acc(logits, label_seg)
                acc = 1.0-loss.item()

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
                self.save_model_full('max_acc')
            # Save model every 10 epochs

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)
            trlog['val_loss'].append(val_loss_averager)
            trlog['val_acc'].append(val_acc_averager)

            # Save log
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))
        writer.close()
    
    def Reg_Sobel(self, img,label):
        grad_x = F.conv2d(img,sobel_x,padding=1)
        grad_y = F.conv2d(img,sobel_y,padding=1)
        grad = torch.sqrt(grad_y*grad_y+grad_x*grad_x)
        scale_max = torch.max(grad.view(grad.size()[0],-1), 1, keepdim=False, out=None).values.view(grad.size()[0],1,1,1)
        scale_min = torch.min(grad.view(grad.size()[0],-1), 1, keepdim=False, out=None).values.view(grad.size()[0],1,1,1)
        noised_gt = (grad-scale_min)/(scale_max-scale_min)
        noised_gt[label==0] = 0
        return noised_gt

    def Loss_MSE_Recon(self, pred,truth,label):
        criterion = nn.MSELoss()
        grad = self.Reg_Sobel(truth,label)
        # mse_loss = torch.sqrt(criterion(pred,grad))
        mse_loss = criterion(pred,grad)
        return mse_loss