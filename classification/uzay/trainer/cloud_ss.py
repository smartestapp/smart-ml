""" Trainer for meta-train phase. """
import os.path as osp
import os,json
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler
from models.NetworkPre import FeatureNet
from utils.misc import Averager, count_acc, count_mem_acc, ensure_path
from tensorboardX import SummaryWriter
from dataloader.dataset_loader import AdaptTrainer, AdaptValider
import torchvision.models as models
import pdb

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

specific_zone = [0,1,2]
class ContainerSS(object):
    """The class that contains the code for the meta-train phase and meta-eval phase."""
    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        meta_base_dir = osp.join(log_base_dir, 'meta')
        if not osp.exists(meta_base_dir):
            os.mkdir(meta_base_dir)

        save_path = '{}_{}_shot{}_step{}_gam{}_lr1{}_lr2{}_epo{}_color{}'.format(args.setname, args.model_type,args.shot,args.step_size,args.gamma,args.meta_lr1,args.meta_lr2,args.max_epoch,args.color)
        save_path = '{}_all'.format(save_path) if args.train_all == 1 else save_path
        args.save_path = '{}/{}'.format(meta_base_dir, save_path)
        ensure_path(args.save_path)

        # Set args to be shareable in the class
        self.args = args
        train_shot = None if args.shot > 100 else args.shot

        trainmode = 'trainall' if self.args.train_all == 1 else 'train'
        self.trainset = AdaptTrainer(trainmode, self.args, train_shot, args.color)
        self.train_loader = DataLoader(dataset=self.trainset, batch_size=args.pre_batch_size, shuffle=True, num_workers=8, pin_memory=True)
        self.valset = AdaptValider(self.args,self.trainset.datasets_path)
        self.val_loader = DataLoader(dataset=self.valset, batch_size=100, shuffle=False, num_workers=8, pin_memory=True)
        self.loc_split = self.trainset.loc_split

        # Build meta-transfer learning model
        self.model = FeatureNet(self.args, mode='cloud_ss',flag_meta=True)
        self.optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, self.model.encoder.parameters())}, \
            {'params': self.model.pre_fc.parameters(), 'lr': self.args.meta_lr2}], lr=self.args.meta_lr1)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        
        # load pretrained model without FC classifier
        if args.model_type == 'ResNet18':
            print('Loading Pre-Trained Model')
            pre_base_dir = osp.join(log_base_dir, 'pre')
            pre_save_path = 'ResNet18_bs100_lr0.01_gamma0.2_epo100_step50_color_init'
            pretrained_dict = torch.load(osp.join(pre_base_dir, pre_save_path, 'max_acc.pth'))['params']
            pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
        elif args.model_type == 'ResNet18ORI':
            pretrained_dict = models.resnet18(pretrained=True)
            pretrained_dict = pretrained_dict.state_dict()
            del pretrained_dict['fc.weight'],pretrained_dict['fc.bias']
            pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
            print('Loading the model pretrained on ImageNet1K')
        model_dict = self.model.state_dict()
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        self.model.eval()
        
        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()
        
    def save_model(self, name, max_acc, max_acc_epoch):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """ 
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
            writer.add_scalar('acc/zone', float(val_acc_averager), epoch)
            writer.add_scalar('acc/memb', float(val_mem_averager), epoch)

            # Update best saved model
            if val_acc_averager > trlog['max_acc']:
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
        self.model.mode = 'cloud_ss'    # self.model.encoder.eval()
        train_loss_averager = Averager()
        train_acc_averager = Averager()

        tqdm_gen = tqdm.tqdm(self.train_loader,leave=False)
        for i, batch in enumerate(tqdm_gen, 1):
            data_zone = batch[0].view(-1,3,64,100).cuda()
            label_zone = batch[1].view(-1).type(torch.cuda.LongTensor)

            x = data_zone
            x_180 = x.flip(2).flip(3)
            x_augmentation = torch.cat((x, x_180),0)
            y_augmentation = label_zone.repeat(2)

            # Output logits, loss and accuracy for model
            logits,_ = self.model(x_augmentation)
            loss = F.cross_entropy(logits, y_augmentation)
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
                # print('load',path)
            except:
                return 1,2,'temp model '+path
        self.model.eval()
        self.model.mode = 'cloud_ss'

        if self.val_loader is not None:
            test_loader = self.val_loader
        else:
            testset = AdaptValider(self.args)
            test_loader = DataLoader(dataset=testset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True) #self.args.pre_batch_size
        loc_split = test_loader.dataset.loc_split

        # result_load = {}
        # result_load['batch_%d'%(i)] = {'acc':acc, 'pred':pred_memb.numpy(), 'label':label_memb.numpy()}
        result_gather = {'pred':[],'label':[],'zone_match':[],'memb_match':[],'conf':[]}
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
                data_seg.append(F.upsample(data[:,:,:,int(pos['y']*320):int(pos['y']*320)+int(pos['h']*320)],size=[64,100],mode='bilinear'))
            
            # Output logits for model
            logits,_ = self.model(torch.cat(data_seg))
            prob_zone,pred_zone = torch.max(F.softmax(logits, dim=1),dim=-1)
            prob_memb = prob_zone.detach().cpu().view(len(loc_split),-1).permute(1,0)
            pred_memb = pred_zone.detach().cpu().view(len(loc_split),-1).permute(1,0)
            # pdb.set_trace()
            # pred_memb = prob_zone.argmax(dim=1).cpu().view(len(loc_split),-1).permute(1,0)
            label_memb = label_seg.contiguous().cpu().view(len(loc_split),-1).permute(1,0)
            acc = count_acc(logits, label_seg)
            mem_acc = count_mem_acc(pred_memb, label_memb)
            
            val_acc_averager.add(acc)
            val_mem_averager.add(mem_acc)

            result_gather['pred'].append(pred_memb.numpy())
            result_gather['conf'].append(prob_memb.numpy())
            result_gather['label'].append(label_memb.numpy())
            result_gather['zone_match'].append(np.where(np.equal(result_gather['pred'][-1],result_gather['label'][-1]),1,0))
            result_gather['memb_match'].append(np.where(np.equal(result_gather['zone_match'][-1].mean(axis=-1),1),1,0))
        
        # Update validation averagers
        val_acc_averager = val_acc_averager.item()
        val_mem_averager = val_mem_averager.item()
        msg = 'Test Acc Zone={:.4f} Mem={:.4f}'.format(val_acc_averager,val_mem_averager)

        if epoch == 0:
            result_gather['pred'] = np.concatenate(result_gather['pred'],axis=0)
            result_gather['conf'] = np.concatenate(result_gather['conf'],axis=0)
            result_gather['label'] = np.concatenate(result_gather['label'],axis=0)
            result_gather['zone_match'] = np.concatenate(result_gather['zone_match'],axis=0)
            result_gather['memb_match'] = np.concatenate(result_gather['memb_match'],axis=0)
            for i in range(result_gather['pred'].shape[0]):
                print('pred:',result_gather['pred'][i],'conf:',result_gather['conf'][i],'label:',result_gather['label'][i],'zone_match:',result_gather['zone_match'][i],'memb_match:',result_gather['memb_match'][i])
        
        return val_acc_averager,val_mem_averager,msg

    def prototype_eval(self):
        self.model.eval()
        features_np,labels_np = [],[]
        tqdm_gen = tqdm.tqdm(self.train_loader,leave=False)
        for i, batch in enumerate(tqdm_gen, 1):
            data_zone = batch[0].view(-1,3,64,100).cuda()
            label_zone = batch[1].view(-1)

            x = data_zone
            x_180 = x.flip(2).flip(3)
            x_augmentation = torch.cat((x, x_180),0)
            y_augmentation = label_zone.repeat(2)

            # Output logits, loss and accuracy for model
            _,features = self.model(x_augmentation)
            features = F.normalize(features,p=2,dim=-1).detach().cpu().numpy()
            features_np.append(features)
            labels_np.append(y_augmentation.numpy())
            tqdm_gen.set_description('Preparing')

        supp_feat_np = np.concatenate(features_np,axis=0)
        supp_label_np = np.concatenate(labels_np,axis=0)
        msg_list = []

        clf = LinearSVC()
        pdb.set_trace()
        # clf.fit(supp_feat_np, supp_label_np)
        # clf_attr = {'coef':clf.coef_,'intercept':clf.intercept_,'n_iter':clf.n_iter_}
        model_file = '{}.npy'.format(self.args.setname)
        the_clf = np.load(model_file,allow_pickle=True).item()
        clf.coef_ = the_clf['coef']
        clf.intercept_ = the_clf['intercept']
        clf.n_iter_ = the_clf['n_iter']
        clf.classes_ = np.array([0,1]).astype(np.int)
        
        if self.val_loader is not None:
            test_loader = self.val_loader
        else:
            testset = AdaptValider(self.args)
            test_loader = DataLoader(dataset=testset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True) #self.args.pre_batch_size
        tqdm_gen,loc_split = tqdm.tqdm(test_loader,leave=False),test_loader.dataset.loc_split
        result_gather = {'pred':[],'label':[],'zone_match':[],'memb_match':[]}

        for i, batch in enumerate(tqdm_gen, 1):
            data = batch[0].cuda()
            label_seg = torch.cat(batch[1])
            label_seg = label_seg.type(torch.cuda.LongTensor)

            data_seg = []
            for idx,zone_key in enumerate(loc_split):
                pos = loc_split[zone_key]
                data_seg.append(F.upsample(data[:,:,:,int(pos['y']*320):int(pos['y']*320)+int(pos['h']*320)],size=[64,100],mode='bilinear'))
            
            # Output logits for model
            _,features = self.model(torch.cat(data_seg))
            features = F.normalize(features,p=2,dim=-1).detach().cpu().numpy()
            pred_memb = np.transpose(clf.predict(features).reshape(len(loc_split),-1),(1,0))
            label_memb = label_seg.contiguous().cpu().view(len(loc_split),-1).permute(1,0)

            result_gather['pred'].append(pred_memb)
            result_gather['label'].append(label_memb.numpy())
            result_gather['zone_match'].append(np.where(np.equal(result_gather['pred'][-1],result_gather['label'][-1]),1,0))
            result_gather['memb_match'].append(np.where(np.equal(result_gather['zone_match'][-1].mean(axis=-1),1),1,0))
        
        result_gather['pred'] = np.concatenate(result_gather['pred'],axis=0)
        result_gather['label'] = np.concatenate(result_gather['label'],axis=0)
        result_gather['zone_match'] = np.concatenate(result_gather['zone_match'],axis=0)
        result_gather['memb_match'] = np.concatenate(result_gather['memb_match'],axis=0)
        # for i in range(result_gather['pred'].shape[0]):
        #     print(result_gather['pred'][i],result_gather['label'][i],result_gather['zone_match'][i],result_gather['memb_match'][i])
        the_msg = 'zone acc {} memb acc {}'.format(np.mean(result_gather['zone_match']),np.mean(result_gather['memb_match']))
        print(the_msg)