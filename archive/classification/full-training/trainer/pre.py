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
from misc import Averager, count_acc, ensure_path, count_mem_acc
from tensorboardX import SummaryWriter
from dataloader.PreLoader import AdaptTrainer as TrainDataset
from dataloader.PreLoader import AdaptValider as ValidDataset
import torchvision.models as models
import pdb
import numpy as np

def folder_name(args):
    save_path = '{}_lr{}_gam{}_step{}'.format(args.model_type,args.lr,args.gamma,args.step_size)
    save_path = '{}_syn{}_all{}_std{}'.format(save_path,args.add_syn,args.train_all,args.noise_std)
    save_path = '{}_run{}'.format(save_path,args.run)
    return save_path

class PreTrainer(object):
    """The class that contains the code for the pretrain phase."""
    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        folder = folder_name(args)
        args.save_path = '{}/{}'.format(osp.join(args.log_root,args.mode),folder)
        ensure_path(args.save_path)
        self.args = args

        # Build dataset and data loader
        if 'train' in self.args.phase:
            self.trainset = TrainDataset('train', self.args, None, train_aug=True)
            self.train_loader = DataLoader(dataset=self.trainset, batch_size=args.pre_batch_size, shuffle=True, num_workers=8, pin_memory=True)
            self.valset = ValidDataset('val', self.args, None, train_aug=False)
            self.val_loader = DataLoader(dataset=self.valset, batch_size=int(args.pre_batch_size/2), shuffle=True, num_workers=8, pin_memory=True)
            self.loc_split = self.trainset.loc_split
        
        # Build pretrain model
        self.model = FeatureNet(self.args, mode='pre', n_class=args.n_class)

        # load Image-Net Pretrained network when self_init is 1
        if args.self_init == 1:
            pretrained_dict = models.resnet18(pretrained=True)
            pretrained_dict = pretrained_dict.state_dict()
            del pretrained_dict['fc.weight'],pretrained_dict['fc.bias']
            pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
            print('Loading the model pretrained on ImageNet1K')
            model_dict = self.model.state_dict()
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            self.model.eval()

        # Set optimizer
        if args.optim == 'Adam':
            self.optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, self.model.encoder.parameters())}, \
                {'params': self.model.pre_fc.parameters(), 'lr': self.args.meta_lr1}], lr=self.args.meta_lr1)
        elif args.optim == 'SGD':
            # self.optimizer = torch.optim.SGD()
            pass
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

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
            # Update learning rate and Set model mode
            self.lr_scheduler.step()
            self.model.train()
            self.model.mode = 'pre'
            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc_averager = Averager()
                
            # Using tqdm to read samples from train loader
            tqdm_gen = tqdm.tqdm(self.train_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                # Update global count number 
                global_count = global_count + 1
                data = batch[0].cuda()

                label_seg = torch.cat(batch[1]).type(torch.cuda.LongTensor)

                data_seg = []
                for idx,zone_key in enumerate(self.loc_split):
                    pos = self.loc_split[zone_key]
                    data_seg.append(F.upsample(data[:,:,:,int(pos['y']*320):int(pos['y']*320)+int(pos['h']*320)],size=[64,100],mode='bilinear'))

                # index = specific_zone
                # label_seg = batch[1][index].type(torch.cuda.LongTensor)
                # pos = self.loc_split['zone'+str(index+1)]
                # data_seg = [F.upsample(data[:,:,:,int(pos['y']*320):int(pos['y']*320)+int(pos['h']*320)],size=[64,100],mode='bilinear')]
                
                x = torch.cat(data_seg)
                x_180 = x.flip(2).flip(3)
                x_augmentation = torch.cat((x, x_180),0)
                y_augmentation = label_seg.repeat(2)
                # Output logits, loss, and accuracy
                logits = self.model(x_augmentation)
                loss = F.cross_entropy(logits, y_augmentation)
                acc = count_acc(logits, y_augmentation)
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
            self.model.mode = 'pre'

            # Set averager classes to record validation losses and accuracies
            val_loss_averager = Averager()
            val_acc_averager = Averager()
              
            # Print previous information  
            if epoch % 10 == 0:
                print('Best Epoch {}, Best Val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
            # Run meta-validation

            tqdm_gen = tqdm.tqdm(self.val_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                data = batch[0].cuda()

                # index = specific_zone
                # label_seg = batch[1][index].type(torch.cuda.LongTensor)
                # pos = self.loc_split['zone'+str(index+1)]
                # data_seg = [F.upsample(data[:,:,:,int(pos['y']*320):int(pos['y']*320)+int(pos['h']*320)],size=[64,100],mode='bilinear')]
                
                label_seg = torch.cat(batch[1]).type(torch.cuda.LongTensor)
                data_seg = []
                for idx,zone_key in enumerate(self.loc_split):
                    pos = self.loc_split[zone_key]
                    data_seg.append(F.upsample(data[:,:,:,int(pos['y']*320):int(pos['y']*320)+int(pos['h']*320)],size=[64,100],mode='bilinear'))

                # Output logits for model
                logits = self.model(torch.cat(data_seg))
                # print('logits',logits.size())
                # Calculate train loss
                loss = F.cross_entropy(logits, label_seg)
                # Calculate train accuracy
                acc = count_acc(logits, label_seg)

                val_loss_averager.add(loss.item(),data.size(0))
                val_acc_averager.add(acc,data.size(0))

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

    def eval(self,param_mode='full'):
        """The function for the pre-train phase."""
        
        pre_save_path = './logs/pre/'
        model_path = 'batchsize128_lr0.1_gamma0.2_step30_maxepoch150'

        if param_mode == 'full':
            pretrained_dict = torch.load(osp.join(pre_save_path, 'confirmed', model_path, 'max_acc_full.pth'))['params']
            # pretrained_dict = torch.load(osp.join(self.args.save_path, 'max_acc_full.pth'))['params']
            self.model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.model_dict}
            self.model_dict.update(pretrained_dict)
            self.model.load_state_dict(self.model_dict)
        elif param_mode == 'feature':
            pretrained_dict = torch.load(osp.join(pre_save_path+model_path, 'max_acc.pth'))['params']
            pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
            self.model_dict = self.model.state_dict()
            self.model_dict.update(pretrained_dict)
            self.model.load_state_dict(self.model_dict)
        else:
            raise ValueError('Wrong params mode name.') 

        testset = Dataset('val', self.args, None, train_aug=False)
        test_loader = DataLoader(dataset=testset, batch_size=200, shuffle=True, num_workers=8, pin_memory=True) #self.args.pre_batch_size
        loc_split = testset.loc_split

        result_load = {}

        # Start validation for this epoch, set model to eval mode
        self.model.eval()
        self.model.mode = 'pre'

        # Set averager classes to record validation losses and accuracies
        val_acc_averager = Averager()
        val_mem_averager = Averager()
        tqdm_gen = tqdm.tqdm(test_loader)
        label_list = []
        pred_list = []
        for i, batch in enumerate(tqdm_gen, 1):

            data = batch[0].cuda()

            label_seg = torch.cat(batch[1]).type(torch.cuda.LongTensor)
            data_seg = []
            for idx,zone_key in enumerate(loc_split):
                pos = loc_split[zone_key]
                data_seg.append(F.upsample(data[:,:,:,int(pos['y']*320):int(pos['y']*320)+int(pos['h']*320)],size=[64,100],mode='bilinear'))
            
            # index = specific_zone
            # label_seg = batch[1][index].type(torch.cuda.LongTensor)
            # pos = self.loc_split['zone'+str(index+1)]
            # data_seg = [F.upsample(data[:,:,:,int(pos['y']*320):int(pos['y']*320)+int(pos['h']*320)],size=[64,100],mode='bilinear')]

            # Output logits for model
            logits = self.model(torch.cat(data_seg))
            acc = count_acc(logits, label_seg)
            mem_acc = count_mem_acc(logits, label_seg)

            pred = F.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()
            pred_list.append(pred)
            label_list.append(label_seg.cpu().numpy())

            val_acc_averager.add(acc)
            val_mem_averager.add(mem_acc)
            result_load['batch_%d'%(i)] = {'acc':acc, 'pred':pred, 'label':label_seg.cpu().numpy()}

        print(label_seg)
        print(pred)
        # Update validation averagers
        val_acc_averager = val_acc_averager.item()  
        val_mem_averager = val_mem_averager.item()
        # Print loss and accuracy for this epoch
        print('Test, Acc={:.4f} MemAcc={:.4f}'.format(val_acc_averager,val_mem_averager))
        baseline_dir = './baseline/'
        if not osp.exists(baseline_dir):
            os.mkdir(baseline_dir)
        set_baseline = baseline_dir + self.args.setname + '/'
        if not osp.exists(set_baseline):
            os.mkdir(set_baseline)

        result_load['acc'] = val_acc_averager
        # np.save(set_baseline+model_path+'_'+param_mode+'.npy', result_load)
        # with open(set_baseline + model_path + '.json','w') as json_file:
        #     json.dump(result_load,json_file, indent=2)
