""" Dataloader for all datasets. """
import os,json,pdb,argparse
import os.path as osp
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from .augmentation_pool import RandAugmentMC,RandAugmentPC
from .processing_tool import CommonLoad,AddGaussianNoise,image_convert,read_from_path

ratios = {'ACON_Ab':[(0.83,0.3),(0.95,0.7)],
          'DeepBlue_Ag':[(0.1,0.3),(0.15,0.7)],
          'Paramount_Ag':[(0.1,0.3),(0.15,0.7)],
          'BTNx':[(0.83,0.3),(0.95,0.7)],
          'ACON_Ag':[(0.92,0.3),(0.97,0.7)],
          'RapidConnect_Ab':[(0.95,0.3),(0.97,0.7)],
          'Quidel_Ag':[(0.6,0.3),(0.9,0.7)],
          'AccessBio_Ag':[(0.05,0.3),(0.2,0.7)]
         }
         
class DatasetInitializer(Dataset):
    """The class to load the dataset"""
    def __init__(self, setmode, args, num_shot=None):
        # Set the path according to train, val and test   
        self.setmode = setmode
        self.white = args.white
        self.num_shot = num_shot
        self.datasets_path,self.datasets_label,self.json_list = {},{},{}
        self.target = args.setname
        self.mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
        self.std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
        self.setname = 'trainall' if args.train_all == 1 and self.setmode=='train' else args.setname

        with open(args.dataset_dir + 'kit_data_v7.json') as json_file:  
            json_data = json.load(json_file)
        
        all_list = ['BTNx','ACON_Ab','ACON_Ag'] # ['Quidel_Ag','BTNx','ACON_Ab','ACON_Ag','Paramount_Ag','RapidConnect_Ab','DeepBlue_Ag','Abbot_Binax']
        if args.setname not in all_list:
            all_list.append(args.setname)

        if self.setname == 'trainall':
            for set_assist in all_list:
                self.datasets_path[set_assist],self.datasets_label[set_assist] = self.dataset_initialization(args, set_assist)
                self.json_list[set_assist] =  json_data[set_assist]['dimensions']['zones'].copy()
        else:
            self.datasets_path[self.target],self.datasets_label[self.target] = self.dataset_initialization(args, self.target, True)
        
        self.loc_split = json_data[self.target]['dimensions']['zones']
        self.num_split = self.loc_split['n']
        del self.loc_split['n']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.transform(self.data[i]), self.label[i]
    
    def dataset_initialization(self, args, dataset, target=False):
        # Variable Initialization
        label_id_dict,data,label = {},[],[]
        label_file = pd.read_excel(osp.join(args.dataset_dir, dataset, 'labels.xlsx'))
        # Mode Assignment
        # You can assign group_mode as ['Train','Test'] to include all samples if you need
        mode = 'train' if 'train' in self.setmode else 'test'
        group_mode = ['Train'] if mode == 'train' else ['Test']

        if dataset == 'BTNx':
            if mode == 'test':
                ## Eval Mode 1
                THE_PATH = osp.join(args.dataset_dir, 'BTNx_mayo','V102')
                label_file = pd.read_excel(osp.join(THE_PATH, 'labels_Mayo_test_BTNX_105_images.xlsx'))
                
                for the_id, zone1, zone2, zone3 in zip(label_file['cu_key (S)'].to_list(),label_file['final_consensus_control'].to_list(),label_file['final_consensus_igg'].to_list(),label_file['final_consensus_igm'].to_list()):
                    if zone1 == 1:
                        label_id_dict[the_id] = [zone1,zone2,zone3]
                for the_key in label_id_dict:
                    data.append(osp.join(THE_PATH, the_key))
                    label.append(label_id_dict[the_key])
                ## Eval Mode 2
                # THE_PATH = osp.join(args.dataset_dir, 'BTNx_Eval', 'paper', 'test')
                # label_file = pd.read_excel(osp.join(args.dataset_dir, 'BTNx_Eval', 'labels.xlsx'))
                # label_id_dict = {}
                # for the_id, zone1, zone2, zone3 in zip(label_file['Sample ID'].to_list(),label_file['Zone 1'].to_list(),label_file['Zone 2'].to_list(),label_file['Zone 3'].to_list()):
                #     label_id_dict[the_id] = [zone1,zone2,zone3]
                # sub_folder = osp.join(THE_PATH, 'membranes_original')
                # for image_path in os.listdir(sub_folder):
                #     if '._' not in image_path and 'DS_Store' not in image_path:
                #         data.append(osp.join(sub_folder, image_path))
                #         label.append(label_id_dict[image_path[:-4]])
            else:
                THE_PATH = osp.join(args.dataset_dir, 'BTNx_Eval', 'paper', 'train')
                label_file = pd.read_excel(osp.join(args.dataset_dir, 'BTNx_Eval', 'labels.xlsx'))
                for the_id, zone1, zone2, zone3 in zip(label_file['Sample ID'].to_list(),label_file['Zone 1'].to_list(),label_file['Zone 2'].to_list(),label_file['Zone 3'].to_list()):
                    label_id_dict[the_id] = [zone1,zone2,zone3]
                sub_folder = osp.join(THE_PATH, 'membranes_original')
                for image_path in os.listdir(sub_folder):
                    if '._' not in image_path and 'DS_Store' not in image_path:
                        data.append(osp.join(sub_folder, image_path))
                        label.append(label_id_dict[image_path[:-4]])

        elif dataset == 'ACON_Ab':
            for the_id, zone1, zone2, zone3, the_grp in zip(label_file['Sample ID'].to_list(),label_file['Zone 1'].to_list(),label_file['Zone 2'].to_list(),label_file['Zone 3'].to_list(),label_file['Dataset'].to_list()):
                if zone1 in [0,1] and zone2 in [0,1] and zone3 in [0,1] and the_grp in group_mode: # Necessary Filtering cauased by the structure of label file
                    label_id_dict[the_id] = [zone1,zone2,zone3]
            data,label = CommonLoad(args, dataset, label_id_dict)
        
        elif dataset == 'ACON_Ag':
            for the_id, zone1, zone2, the_grp in zip(label_file['Sample ID'].to_list(),label_file['Zone 1'].to_list(),label_file['Zone 2'].to_list(),label_file['Dataset'].to_list()):
                if the_grp in group_mode:
                    label_id_dict[the_id] = [zone1,zone2]
            data,label = CommonLoad(args, dataset, label_id_dict)

        elif dataset == 'Paramount_Ag':
            for the_id, zone1, zone2, the_grp in zip(label_file['Sample ID'].to_list(),label_file['Zone 1'].to_list(),label_file['Zone 2'].to_list(),label_file['Dataset'].to_list()):
                if zone1 in [0,1] and zone2 in [0,1] and the_grp in group_mode:
                    label_id_dict[the_id] = [zone1,zone2]
            data,label = CommonLoad(args, dataset, label_id_dict)

        elif dataset == 'DeepBlue_Ag':
            for the_id, zone1, zone2, the_grp in zip(label_file['Sample ID'].to_list(),label_file['Zone 1'].to_list(),label_file['Zone 2'].to_list(),label_file['Dataset'].to_list()):
                if zone1 in [0,1] and zone2 in [0,1] and the_grp in group_mode:
                    label_id_dict[the_id] = [zone1,zone2]
            data,label = CommonLoad(args, dataset, label_id_dict)

        elif dataset == 'RapidConnect_Ab':
            for the_id, zone1, zone2, zone3 in zip(label_file['Sample ID'].to_list(),label_file['Zone 1'].to_list(),label_file['Zone 2'].to_list(),label_file['Zone 3'].to_list()):
                if the_grp in group_mode:
                    label_id_dict[the_id] = [zone1,zone2,zone3]
            data,label = CommonLoad(args, dataset, label_id_dict)

        elif 'Quidel_Ag' in dataset:
            label_name,memb_name = 'label_full_visual.xlsx','membrane_fullV3'
            THE_PATH = osp.join(args.dataset_dir, 'Quidel_Ag')
            label_file = pd.read_excel(osp.join(THE_PATH, label_name))

            for sub_folder, the_id, zone1, zone2, the_grp in zip(label_file['the_folder'].to_list(),label_file['Sample ID'].to_list(),label_file['Zone 1'].to_list(),label_file['Zone 2'].to_list(),label_file['Dataset'].to_list()):
                if the_grp in group_mode:
                    label_id_dict['{}+{}'.format(sub_folder,the_id)] = [zone1,zone2]
            this_folder = osp.join(THE_PATH, memb_name)
            this_folder_subfolders = os.listdir(this_folder)
            this_folder_subfolders.remove('.DS_Store')
            for this_sub_folder in this_folder_subfolders:
                thie_sub_path = os.path.join(this_folder,this_sub_folder)
                this_folder_images = os.listdir(thie_sub_path)
                for image_path in this_folder_images:
                    if '._' not in image_path and 'DS_Store' not in image_path:
                        if '{}+{}'.format(this_sub_folder,image_path[:-4]) in label_id_dict and image_path.split('.')[-1]=='npy':
                            data.append(osp.join(thie_sub_path, image_path))
                            label.append(label_id_dict['{}+{}'.format(this_sub_folder,image_path[:-4])])
            if 'Train' in group_mode:
                THE_PATH = osp.join(args.dataset_dir, 'Quidel_Ag', 'Quidel_Failure_02_275')
                label_file = pd.read_excel(osp.join(THE_PATH, 'labels.xlsx'))
                Fdata,Flabel = failure_test(osp.join(THE_PATH,'membranes'),label_file,visual=self.visual_judge,lossy=False)
                data.extend(Fdata)
                label.extend(Flabel)

        elif 'Abbot_Binax' in dataset:
            for the_id, zone1, zone2, the_grp in zip(label_file['Sample ID'].to_list(),label_file['Zone 1'].to_list(),label_file['Zone 2'].to_list(),label_file['Dataset'].to_list()):
                if the_grp in group_mode:
                    label_id_dict[the_id] = [zone1,zone2]
            data,label = CommonLoad(args, dataset, label_id_dict)

        print('There are {} images in the {} set for {} mode'.format(len(data),dataset,mode))
        return data,label

class AdaptTrainer(DatasetInitializer):
    def __init__(self, setmode, args, num_shot=None, color_mode=1):
        super(AdaptTrainer,self).__init__(setmode, args, num_shot)
        self.supp_aug = args.supp_aug
        print(color_mode)
        
        memb2zone = transforms.Compose([
            transforms.Resize((160,500)),
            transforms.RandomRotation(10),
            transforms.Resize((160,480)),
            transforms.ToTensor()
        ])
        zone_tensor2pil = transforms.ToPILImage()

        zones = []
        labels = []
        self.label = []
        for the_set in self.datasets_path:
            data_path = self.datasets_path[the_set]
            data_label = self.datasets_label[the_set]
            data_split = self.json_list[the_set] if the_set in self.json_list else self.loc_split.copy() 
            if 'n' in data_split:
                del data_split['n']
            remove_ctrl = args.ctrl_full if the_set == 'BTNx' and len(self.datasets_path) > 1 else 1
            index_memb = [i for i in range(len(data_path))]
            if remove_ctrl > 0 and remove_ctrl < 1:
                keep_index = np.random.choice(index_memb, int(len(data_path)*remove_ctrl), False)
            else:
                keep_index = index_memb if remove_ctrl == 1 else []

            if the_set == self.target and self.num_shot < 50:
                # 50 is too large for few-shot, 
                # when self.num_shot is 50, use all samples
                if self.num_shot == 0:
                    # DO NOT use any available training data
                    continue
                zone = {0:[],1:[]}
                for the_index,(the_path,the_label) in enumerate(zip(data_path,data_label)):
                    the_item = read_from_path(the_path)
                    the_item = memb2zone(the_item).unsqueeze(0)
                    for idx,zone_key in enumerate(data_split):
                        pos = data_split[zone_key]
                        zone[the_label[idx]].append(F.interpolate(the_item[:,:,:,int(pos['y']*480):int(pos['y']*480)+int(pos['h']*480)],size=[160,160],mode='bilinear'))
                neg_index = np.random.choice(len(zone[0]),self.num_shot,replace=False).tolist() if self.num_shot < len(zone[0]) else [i for i in range(len(zone[0]))]
                pos_index = np.random.choice(len(zone[1]),self.num_shot,replace=False).tolist() if self.num_shot < len(zone[1]) else [i for i in range(len(zone[1]))]
                zones.extend([zone[0][int(i)] for i in neg_index]+[zone[1][int(i)] for i in pos_index])
                labels.extend([0]*len(neg_index)+[1]*len(pos_index))
            else:
                # The few-shot sampling is only for target kit, use all samples of assisting kits
                for the_index,(the_path,the_label) in enumerate(zip(data_path,data_label)):
                    zone = []
                    the_item = read_from_path(the_path)
                    the_item = memb2zone(the_item).unsqueeze(0)
                    for zone_key in data_split:
                        if zone_key == 'zone1' and the_index not in keep_index:
                            continue
                        pos = data_split[zone_key]
                        zone.append(F.interpolate(the_item[:,:,:,int(pos['y']*480):int(pos['y']*480)+int(pos['h']*480)],size=[160,160],mode='bilinear'))
                    zones.extend(zone)
                    labels.extend(the_label if the_index in keep_index else the_label[1:])

        self.data = zones
        self.label = labels
        del zones
        self.data = [zone_tensor2pil(the_zone[0]) for the_zone in self.data]

        if args.add_syn == 1:
            neg_root = os.path.join(args.dataset_dir,'VAE_Synthetic','NegativeV2')
            pos_root = os.path.join(args.dataset_dir,'VAE_Synthetic','FaintPositiveV3')
            pos_zone,neg_zone = [],[]

            for the_file in os.listdir(pos_root):
                try:
                    the_img = Image.open(os.path.join(pos_root,the_file)).convert('RGB')
                    the_img = the_img.rotate(90,expand=True).resize((160,160))
                    pos_zone.append(the_img)
                except:
                    pass
            for the_file in os.listdir(neg_root):
                try:
                    the_img = Image.open(os.path.join(neg_root,the_file)).convert('RGB')
                    the_img = the_img.rotate(90,expand=True).resize((160,160))
                    neg_zone.append(the_img)
                except:
                    pass
            pos_label = [1]*len(pos_zone)
            neg_label = [0]*len(neg_zone)
            syn_zone = pos_zone + neg_zone
            syn_label = pos_label + neg_label
            self.data.extend(syn_zone)
            self.label.extend(syn_label)
        assert len(self.data) == len(self.label)
        print('There are {} images for ALL train mode'.format(len(self.data)))

        if color_mode == 0:
            self.transform = transforms.Compose([
                transforms.RandomCrop([160,160],padding=(20,0)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array(self.mean), np.array(self.std))
            ])
        elif color_mode == 1:
            self.transform = transforms.Compose([
                transforms.RandomCrop([160,160],padding=(20,4)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(np.array(self.mean), np.array(self.std))
            ])
        elif color_mode == 2:
            self.transform = transforms.Compose([
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop([160,160],padding=(20,0),padding_mode='reflect'),
                # transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.5)], p=0.8),
                # transforms.RandomGrayscale(p=0.2),
                RandAugmentPC(n=2, m=10),
                transforms.ToTensor(),
                # transforms.RandomApply([AddGaussianNoise(0.1, args.noise_std)], p=0.5),
                transforms.Normalize(np.array(self.mean), np.array(self.std))
            ])      
                                              
        else:
            raise ValueError('Not Implemented Error')

    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        imgs = torch.stack([self.transform(self.data[i]) for _ in range(self.supp_aug)])
        labels = np.array([self.label[i]]*self.supp_aug)
        return imgs,labels

class AdaptValider(DatasetInitializer):
    def __init__(self, args,trained_path=None):
        super(AdaptValider,self).__init__('val', args)
        self.transform = transforms.Compose([
            transforms.Resize((160,480)),
            transforms.ToTensor(),
            transforms.Normalize(np.array(self.mean), np.array(self.std))
        ])
        assert len(self.json_list) == 0 and self.target in self.datasets_path

        self.data,self.label,self.path = [],[],[]
        for the_key in self.datasets_path:
            data_path = self.datasets_path[the_key]
            data_label = self.datasets_label[the_key]
            for the_path,the_label in zip(data_path,data_label):
                try:
                    the_img = read_from_path(the_path)
                except:
                    continue
                self.data.append(image_convert(the_img,ratios[the_key]) if self.white==1 and args.phase=='adapt_test' else the_img)
                self.label.append(the_label)
                self.path.append(the_path)

        print('There are {} images for {} test mode'.format(len(self.data),self.target))
        print(self.loc_split)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.transform(self.data[i]), self.label[i], self.path[i]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--setname', type=str, default='BTNx', choices=['oraquick','biomedomics','maxim','aytu','sd_igg','BTNx'])
    parser.add_argument('--dataset_dir', type=str, default='/home/jiawei/DATA/COVID/Eval_Data_classify/') # Dataset folder
    args = parser.parse_args()