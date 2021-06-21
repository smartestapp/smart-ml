import os.path as osp
import os
import tqdm
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from models.NetworkPre import FeatureNet
from misc import Averager, count_acc, ensure_path
from tensorboardX import SummaryWriter
from dataloader.SelfLoader import SelfLoader as Dataset
import pdb
import numpy as np
import torch.nn as nn
import torchvision.models as models

sobel_x = torch.from_numpy(np.array([[1,0,-1],[2,0,-2],[1,0,-1]])).float().unsqueeze(0).unsqueeze(0)
sobel_y = torch.from_numpy(np.array([[1,2,1],[0,0,0],[-1,-2,-1]])).float().unsqueeze(0).unsqueeze(0)
if torch.cuda.is_available():
    sobel_x = sobel_x.cuda()
    sobel_y = sobel_y.cuda()

def folder_name(args):
    save_path = '{}_lr{}_gam{}_step{}'.format(args.model_type,args.lr,args.gamma,args.step_size)
    save_path = '{}_ce{}_xonly{}_syn{}_std{}'.format(save_path,args.cls_wgt,args.xonly,args.add_syn,args.noise_std)
    save_path = '{}_run{}'.format(save_path,args.run)
    return save_path
    
class SelfTrainer(object):
    """The class that contains the code for the pretrain phase."""
    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        folder = folder_name(args)
        args.save_path = '{}/{}'.format(osp.join(args.log_root,args.mode),folder)
        ensure_path(args.save_path)
        self.args = args

        # Load pretrain set
        self.trainset = Dataset('train', self.args, train_aug=True)
        self.train_loader = DataLoader(dataset=self.trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker, pin_memory=True)
        self.valset = Dataset('val', self.args, train_aug=False)
        self.val_loader = DataLoader(dataset=self.valset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker, pin_memory=True)
        self.loc_split = self.trainset.loc_split
        
        # Build pretrain model
        self.model = FeatureNet(self.args, mode='self', n_class=args.n_class)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.995), weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

        if args.self_init == 1 and args.model_type == 'ResNet18ORI':
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
        torch.save(dict(params=self.model.encoder.state_dict()), osp.join(self.args.save_path, name + '.pth'))

    def save_model_full(self, name): 
        torch.save(dict(params=self.model.state_dict()), osp.join(self.args.save_path, name + '_full.pth'))
    
    def train(self):
        """The function for the pre-train phase."""

        # Set the pretrain log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['train_criterion'] = []
        trlog['eval_criterion'] = []
        trlog['best_criterion'] = None
        trlog['best_epoch'] = 0

        # Set tensorboardX
        writer = SummaryWriter(self.args.save_path)
        
        # Start pretrain
        for epoch in range(1, self.args.max_epoch + 1):
            
            train_msg, train_loss,train_criterion = self.train_epoch(epoch)
            eval_msg, eval_criterion = self.eval_epoch()
            full_msg = '{} | {}'.format(train_msg,eval_msg)

            # Write the tensorboardX records
            writer.add_scalar('train/loss', float(train_loss), epoch)
            writer.add_scalar('train/criterion', float(train_criterion), epoch)
            writer.add_scalar('test/criterion', float(eval_criterion), epoch)
            
            # Update best saved model
            if trlog['best_criterion'] is None or eval_criterion < trlog['best_criterion']:
                trlog['best_criterion'] = eval_criterion
                trlog['best_criterion'] = epoch
                self.save_model('max_acc')
                self.save_model_full('max_acc')
                full_msg = '{} | Saving the Best Model until now'.format(full_msg)

            # Update the logs
            trlog['train_loss'].append(train_loss)
            trlog['train_criterion'].append(train_criterion)
            trlog['eval_criterion'].append(eval_criterion)
            
            # Save log
            print(full_msg)
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))
        writer.close()
    
    def train_epoch(self, epoch):
        loss_averager = Averager()
        criterion_averager = Averager()

        # Update learning rate
        self.lr_scheduler.step()
        # Set the model to train mode
        self.model.train()
        self.model.mode = 'self'

        # Using tqdm to read samples from train loader
        tqdm_gen = tqdm.tqdm(self.train_loader)
        for i, batch in enumerate(tqdm_gen, 1):

            bs,nzone,_,height,width = batch[0].size()
            x = batch[0].view(-1,3,height,width).cuda()
            bs,nzone,_,height,width = batch[2].size()
            z = batch[2].view(-1,1,height,width).cuda()
            label_seg = batch[1].view(-1).type(torch.cuda.LongTensor)

            # Output logits for model
            x_180 = x.flip(2).flip(3)
            x_augmentation = torch.cat((x, x_180),0)
            y_augmentation = label_seg.repeat(2)

            z_180 = z.flip(2).flip(3)
            z_augmentation = torch.cat((z, z_180),0)
            recon,logit = self.model(x_augmentation)

            # Calculate train loss
            loss_cls = F.cross_entropy(logit, y_augmentation)
            loss_recon = self.Loss_MSE_Recon(recon,z_augmentation,y_augmentation)
            loss = loss_recon + self.args.cls_wgt*loss_cls

            # Print loss and accuracy for this step
            tqdm_gen.set_description('Epoch {}, Loss={:.4f} Criterion{:.4f}'.format(epoch, loss.item(), loss_recon.item()))

            # Add loss and accuracy for the averagers
            loss_averager.add(loss.item())
            criterion_averager.add(loss_recon.item())

            # Loss backwards and optimizer updates
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Update the averagers
        loss_averager = loss_averager.item()
        criterion_averager = criterion_averager.item()
        msg = 'Epoch {} Loss={:.4f} Criterion{:.4f}'.format(epoch, loss_averager, criterion_averager)
        return msg,loss_averager,criterion_averager
    
    def eval_epoch(self):
        criterion_averager = Averager()
        # Set the model to Eval mode
        self.model.eval()
        self.model.mode = 'self'

        tqdm_gen = tqdm.tqdm(self.val_loader)
        for i, batch in enumerate(tqdm_gen, 1):
            bs,nzone,_,height,width = batch[0].size()
            x = batch[0].view(-1,3,height,width).cuda()
            bs,nzone,_,height,width = batch[2].size()
            z = batch[2].view(-1,1,height,width).cuda()
            label_seg = batch[1].view(-1).type(torch.cuda.LongTensor)

            recon,_ = self.model(x)
            loss_recon = self.Loss_MSE_Recon(recon,z,label_seg)
            criterion_averager.add(loss_recon.item())

        criterion_averager = criterion_averager.item()
        msg = 'Test Criterion{:.4f}'.format(criterion_averager)
        return msg,criterion_averager
    
    def Reg_Sobel(self, img,label):
        grad_x = F.conv2d(img,sobel_x,padding=1)
        grad_y = F.conv2d(img,sobel_y,padding=1)
        if self.args.xonly == 1:
            grad = torch.sqrt(grad_x*grad_x)
        else:
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