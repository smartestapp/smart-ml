""" Trainer for meta-train phase. """
import os.path as osp
import os,json
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.NetworkPre import FeatureNet
from misc import Averager, count_acc, count_mem_acc, ensure_path
from tensorboardX import SummaryWriter
from dataloader.PreLoader import AdaptTrainer as TrainDataset
from dataloader.PreLoader import AdaptValider as ValidDataset
import torchvision.models as models
import pdb
import pandas as pd
import pickle


ResNet18_DIR = [
    None
    # 'ResNet18_bs150_lr0.0001_gam0.2_epo150_step30_cls0.0_xonly0_std0.05_init',
    # 'ResNet18_bs150_lr0.0001_gam0.2_epo150_step30_cls0.0_xonly0_std0.05_init'
]
ResNet18ORI_DIR = [
    'ResNet18ORI_lr0.0005_gam0.2_step30_ce0_xonly0_syn1_std0_run0'
    # 'ResNet18ORI_bs150_lr0.0001_gam0.2_epo150_step30_cls0.0_xonly0_std0.05_init',
    # 'ResNet18ORI_bs150_lr0.0001_gam0.2_epo150_step30_cls0.0_xonly0_std0.1_init',
    # 'ResNet18ORI_bs150_lr0.0005_gam0.2_epo150_step30_cls0.0_xonly1_std0.05_init'
]

def folder_name(args):
    # Simplified Version
    # save_path = '{}_{}_shot{}_label{}'.format(args.setname, args.model_type, args.shot, args.label_mode)
    # save_path = '{}_init{}_lr{}'.format(save_path,args.init,args.lr)
    # save_path = '{}_color{}_ctrl{}'.format(save_path,args.color,args.ctrl_full)
    # save_path = '{}_run{}'.format(save_path,args.run)

    save_path = '{}_{}_shot{}_label{}_ct{}'.format(args.setname, args.model_type, args.shot, args.label_mode, args.ct_wgt)
    save_path = '{}_init{}_lr{}_gam{}_step{}'.format(save_path,args.adapt_init,args.lr,args.gamma,args.step_size)
    save_path = '{}_syn{}_all{}_std{}_color{}_ctrl{}'.format(save_path,args.add_syn,args.train_all,args.noise_std,args.color,args.ctrl_full)
    save_path = '{}_run{}'.format(save_path,args.run)
    return save_path

specific_zone = [0,1,2]
class AdaptTrainer(object):
    """The class that contains the code for the meta-train phase and meta-eval phase."""
    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        folder = folder_name(args)
        args.save_path = '{}/{}'.format(osp.join(args.log_root,args.mode),folder)
        ensure_path(args.save_path)
        self.args = args
        
        train_shot = None if args.shot >= 100 else args.shot
        trainmode = 'trainall' if self.args.train_all == 1 else 'train'
        
        self.trainset = TrainDataset(trainmode, self.args, train_shot, args.color)
        self.train_loader = DataLoader(dataset=self.trainset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
        self.valset = ValidDataset(self.args,self.trainset.datasets_path)
        self.val_loader = DataLoader(dataset=self.valset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
        self.loc_split = self.trainset.loc_split

        # Build meta-transfer learning model
        self.model = FeatureNet(self.args, mode='adapt', n_class=args.n_class, flag_meta=True)
        if args.optim == 'Adam':
            self.optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, self.model.encoder.parameters())}, \
                {'params': self.model.pre_fc.parameters(), 'lr': self.args.lr}], lr=self.args.lr)
        elif args.optim == 'SGD':
            pass # self.optimizer = torch.optim.SGD()
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        
        # load pretrained model without FC classifier
        if args.adapt_init == 0:
            assert args.model_type == 'ResNet18ORI'
            pretrained_dict = models.resnet18(pretrained=True)
            pretrained_dict = pretrained_dict.state_dict()
            del pretrained_dict['fc.weight'],pretrained_dict['fc.bias']
            pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
            print('Loading the model pretrained on ImageNet1K')
        else:
            if args.model_type == 'ResNet18ORI':
                the_dir = ResNet18ORI_DIR
            elif args.model_type == 'ResNet18':
                the_dir = ResNet18_DIR
            print('Loading Pre-Trained Model under Self-Supervision')
            model = osp.join(args.log_root, 'self', the_dir[args.adapt_init-1], 'max_acc.pth')
            pretrained_dict = torch.load(model)['params']
            pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
        model_dict = self.model.state_dict()
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        self.model.eval()
        
        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()
        
    def save_model(self, name, max_acc, max_acc_epoch):
        checkpoints = {'params':self.model.state_dict(),'max_acc':max_acc, 'max_acc_epoch':max_acc_epoch}
        torch.save(checkpoints, osp.join(self.args.save_path, name + '.pth'))
        print('save_model {}.pth'.format(name))
    
    def train(self):
        """The function for the pre-train phase."""

        # Set the pretrain log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc_zone'] = []
        trlog['val_acc_memb'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0

        writer = SummaryWriter(self.args.save_path)
        # self.prototype_eval()
        
        # Start pretrain
        for epoch in range(1, self.args.max_epoch+1):
            self.lr_scheduler.step()
            train_loss_averager,train_acc_averager,train_msg = self.train_epoch(epoch)
            val_acc_averager,val_mem_averager,test_msg = self.eval_epoch(epoch)
            print('{} | {}'.format(train_msg,test_msg))

            writer.add_scalar('train/loss', float(train_loss_averager), epoch)
            writer.add_scalar('train/acc', float(train_acc_averager), epoch)
            writer.add_scalar('eval/zone', float(val_acc_averager), epoch)
            writer.add_scalar('eval/memb', float(val_mem_averager), epoch)

            # Update best saved model
            if (val_acc_averager >= trlog['max_acc'] and epoch < 30) or (val_acc_averager > trlog['max_acc']):
                trlog['max_acc'] = val_acc_averager
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc',trlog['max_acc'],trlog['max_acc_epoch'])

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)
            trlog['val_acc_zone'].append(val_acc_averager)
            trlog['val_acc_memb'].append(val_mem_averager)

            # Save log
            if epoch % 10 == 0 or epoch == self.args.max_epoch:
                torch.save(trlog, osp.join(self.args.save_path, 'trlog'))
                print('Logging ... The best accuracy {} in epoch {}'.format(trlog['max_acc'],trlog['max_acc_epoch']))
        writer.close()
    
    def train_epoch(self,epoch):
        self.model.train()              # Set the model to train mode
        self.model.mode = 'adapt'    # self.model.encoder.eval()
        train_loss_averager = Averager()
        train_acc_averager = Averager()

        tqdm_gen = tqdm.tqdm(self.train_loader,leave=False)
        for i, batch in enumerate(tqdm_gen, 1):
            left_zdata = batch[0][:,0,].cuda()
            right_zdata = batch[0][:,1,].cuda()
            left_zlabel = batch[1][:,0].type(torch.cuda.LongTensor)
            right_zlabel = batch[1][:,1].type(torch.cuda.LongTensor)

            left_aug = torch.cat((left_zdata, left_zdata.flip(2).flip(3)),0)
            right_aug = torch.cat((right_zdata, right_zdata.flip(2).flip(3)),0)
            x_augmentation = torch.cat((left_aug, right_aug),0)
            y_augmentation = torch.cat((left_zlabel.repeat(2), right_zlabel.repeat(2)),0)
            bsz = left_aug.size(0)
            
            # Output logits, loss and accuracy for model
            logits,features = self.model(x_augmentation)
            features = F.normalize(features, p=2, dim=-1).unsqueeze(1)
            confeat = torch.cat([features[:bsz], features[bsz:]], dim=1)
            loss = F.cross_entropy(logits, y_augmentation) + self.args.ct_wgt * SupConLoss(confeat,y_augmentation[:bsz])
            acc = count_acc(logits, y_augmentation)
            
            # Print Logging
            tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f}'.format(epoch, loss.item(), acc))
            train_loss_averager.add(loss.item())
            train_acc_averager.add(acc)

            # Loss backwards and optimizer updates
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update the averagers
        train_loss_averager = train_loss_averager.item()
        train_acc_averager = train_acc_averager.item()
        train_msg = 'Epoch {} Loss {} Acc {}'.format(epoch,train_loss_averager,train_acc_averager)
        return train_loss_averager,train_acc_averager,train_msg

    def eval_epoch(self,epoch=0,path=None):
        """The function for the pre-train phase."""
        if path is not None:
            try:
                pretrained_dict = torch.load(osp.join(path, 'max_acc.pth'))['params']
                pretrained_dict = {k:v for k, v in pretrained_dict.items() if 'encoder' in k or 'pre_fc' in k}
                self.model.load_state_dict(pretrained_dict)
            except:
                return 1,2,'temp model '+path
        self.model.eval()
        self.model.mode = 'adapt'

        if self.val_loader is not None:
            test_loader = self.val_loader
        else:
            testset = AdaptValider(self.args)
            test_loader = DataLoader(dataset=testset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
        loc_split = test_loader.dataset.loc_split

        result_gather = {'pred':[],'label':[],'zone_match':[],'memb_match':[],'conf':[],'file':[]}
        val_acc_averager = Averager()
        val_mem_averager = Averager()
        tqdm_gen = tqdm.tqdm(test_loader,leave=False)

        for i, batch in enumerate(tqdm_gen, 1):
            data = batch[0].cuda()
            label_seg = torch.cat(batch[1])
            label_seg = label_seg.type(torch.cuda.LongTensor)

            data_seg = []
            for idx,zone_key in enumerate(loc_split):
                pos = loc_split[zone_key]
                data_seg.append(F.interpolate(data[:,:,:,int(pos['y']*480):int(pos['y']*480)+int(pos['h']*480)],size=[160,160],mode='bilinear'))
            
            # Output logits for model
            logits,_ = self.model(torch.cat(data_seg))
            prob_zone,pred_zone = torch.max(F.softmax(logits, dim=1),dim=-1)
            prob_memb = prob_zone.detach().cpu().view(len(loc_split),-1).permute(1,0)
            pred_memb = pred_zone.detach().cpu().view(len(loc_split),-1).permute(1,0)
            label_memb = label_seg.contiguous().cpu().view(len(loc_split),-1).permute(1,0)
            acc = count_acc(logits, label_seg)
            mem_acc = count_mem_acc(pred_memb, label_memb)
            
            val_acc_averager.add(acc,data.size(0)*len(loc_split))
            val_mem_averager.add(mem_acc,data.size(0))

            result_gather['pred'].append(pred_memb.numpy())
            result_gather['conf'].append(prob_memb.numpy())
            result_gather['label'].append(label_memb.numpy())
            result_gather['zone_match'].append(np.where(np.equal(result_gather['pred'][-1],result_gather['label'][-1]),1,0))
            result_gather['memb_match'].append(np.where(np.equal(result_gather['zone_match'][-1].mean(axis=-1),1),1,0))
            result_gather['file'].extend(batch[2])

        # Update validation averagers
        val_acc_averager = val_acc_averager.item()
        val_mem_averager = val_mem_averager.item()
        msg = 'Test Acc Zone={:.4f} Mem={:.4f}'.format(val_acc_averager,val_mem_averager)

        if epoch == 0:
            records = []
            for the_key in ['pred','label','zone_match','memb_match','conf']:
                result_gather[the_key] = np.concatenate(result_gather[the_key],axis=0)
            for the_pred,the_conf,the_label,the_zonematch,the_membmatch,the_file in zip(result_gather['pred'],result_gather['conf'],result_gather['label'],result_gather['zone_match'],result_gather['memb_match'],result_gather['file']):
                records.append((the_pred.tolist(),the_conf.tolist(),the_label.tolist(),the_zonematch.tolist(),the_membmatch.tolist(),the_file))

            log_root = './logs/json_log/'+path.split('/')[-1]
            ensure_path(log_root)
            # Save the results in two different formats for the convenience of processing
            df_records = pd.DataFrame.from_dict(records)
            df_records.to_excel(log_root+'/data.xlsx')
            with open(log_root+'/result.json', 'w') as json_file:
                json.dump(records,json_file)
            
        return val_acc_averager,val_mem_averager,msg

def SupConLoss(features, labels=None, mask=None, temperature=0.5, contrast_mode='all',base_temperature=0.07):

    device = (torch.device('cuda')
                if features.is_cuda
                else torch.device('cpu'))

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                            'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss
