""" Dataloader for all datasets. """
import os.path as osp
import os,json
from PIL import Image
import numpy as np
import pdb
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from .augmentation_pool import RandAugmentMC,RandAugmentPC
from .processing_tool import CommonLoad,AddGaussianNoise

class DatasetInitializer(Dataset):
    """The class to load the dataset"""
    def __init__(self, setmode, args, num_shot=None):
        # Set the path according to train, val and test   
        self.setmode = setmode
        self.num_shot = num_shot
        self.datasets_path,self.datasets_label,self.loc_split,self.num_split = {},{},{},{}
        self.mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
        self.std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
        self.setname = 'trainall' if args.train_all == 1 and self.setmode=='train' else args.setname

        with open(args.dataset_dir + 'kit_data_v7.json') as json_file:  
            json_data = json.load(json_file)
        
        all_list = ['BTNx','ACON_Ab','ACON_Ag'] # 'AccessBio_Ag', 'Paramount_Ag','RapidConnect_Ab','DeepBlue_Ag','Abbot_Binax'
        if args.setname not in all_list:
            all_list.append(args.setname)
        
        if self.setname == 'trainall':
            for set_assist in all_list:
                self.datasets_path[set_assist],self.datasets_label[set_assist] = self.dataset_initialization(args, set_assist) # 
                temp_json = json_data[set_assist]['dimensions']['zones'].copy()
                self.num_split[set_assist] = temp_json['n']
                del temp_json['n']
                self.loc_split[set_assist] =  temp_json
        else:
            self.datasets_path[self.setname],self.datasets_label[self.setname] = self.dataset_initialization(args, self.setname)
            temp_json = json_data[self.setname]['dimensions']['zones'].copy()
            self.num_split[self.setname] = temp_json['n']
            del temp_json['n']
            self.loc_split[self.setname] =  temp_json
        
        self.data = []
        self.label = []
        self.kitid = []
        for the_key in self.datasets_path:
            data_path = self.datasets_path[the_key]
            for the_path in data_path:
                the_img = Image.open(the_path).convert('RGB')
                the_item = the_img.rotate(90,expand=True) if the_img.size[0] < the_img.size[1] else the_img.rotate(180,expand=True)
                self.data.append(the_item)
            self.label.extend(self.datasets_label[the_key])
            self.kitid.extend([the_key for _ in range(len(data_path))])
            
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
        group_mode = 'Train' if mode == 'train' else 'Test' 
        
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

        elif 'Abbot_Binax' in dataset:
            for the_id, zone1, zone2, the_grp in zip(label_file['Sample ID'].to_list(),label_file['Zone 1'].to_list(),label_file['Zone 2'].to_list(),label_file['Dataset'].to_list()):
                if the_grp in group_mode:
                    label_id_dict[the_id] = [zone1,zone2]
            data,label = CommonLoad(args, dataset, label_id_dict)

        print('There are {} images in the {} set for {} mode'.format(len(data),dataset,mode))
        return data,label

class SelfLoader(DatasetInitializer):
    def __init__(self, setmode, args, train_aug=False):
        super(SelfLoader,self).__init__(setmode, args)

        if train_aug:
            self.transform = transforms.Compose([
                transforms.Resize((160,480)),
                transforms.RandomVerticalFlip()
            ])
            self.transform_zone = transforms.Compose([
                transforms.Resize((160,160)),
                transforms.RandomVerticalFlip()
            ])
            self.rgb = transforms.Compose([
                transforms.RandomApply([transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4, hue=0.5)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.RandomApply([AddGaussianNoise(0.1, args.noise_std)], p=0.8),
                transforms.Normalize(np.array(self.mean), np.array(self.std))
            ])
        else:
            self.transform = transforms.Compose([transforms.Resize((160,480))])
            self.transform_zone = transforms.Compose([transforms.Resize((160,160))])
            self.rgb = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(np.array(self.mean), np.array(self.std))
            ])
        self.gray = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])


        if args.add_syn == 1 and self.setmode == 'train':
            neg_root = os.path.join(args.dataset_dir,'VAE_Synthetic','NegativeV2')
            pos_root = os.path.join(args.dataset_dir,'VAE_Synthetic','FaintPositiveV3')
            pos_zone,neg_zone = [],[]

            for the_file in os.listdir(pos_root):
                try:
                    the_img = Image.open(os.path.join(pos_root,the_file)).convert('RGB')
                    the_img = the_img.rotate(90,expand=True)
                    pos_zone.append(the_img)
                except:
                    pass
            for the_file in os.listdir(neg_root):
                try:
                    the_img = Image.open(os.path.join(neg_root,the_file)).convert('RGB')
                    the_img = the_img.rotate(90,expand=True)
                    neg_zone.append(the_img)
                except:
                    pass
            pos_label = [1]*len(pos_zone)
            neg_label = [0]*len(neg_zone)
            self.syn_zone = pos_zone + neg_zone
            self.syn_label = pos_label + neg_label
        else:
            self.syn_zone = []
            self.syn_label = []
            
        print(len(self.syn_label))

    def __len__(self):
        return len(self.data) + int(len(self.syn_label))

    def __getitem__(self, i):
        if i < len(self.data):
            image = self.transform(self.data[i])
            rgb,gray = self.rgb(image).unsqueeze(0),self.gray(image).unsqueeze(0)
            the_kit,the_label = self.kitid[i],self.label[i]
            loc_split,num_split = self.loc_split[the_kit],self.num_split[the_kit]
            data_seg,gray_seg = [],[]
            for idx,zone_key in enumerate(loc_split):
                pos = loc_split[zone_key]
                data_seg.append(F.interpolate(rgb[:,:,:,int(pos['y']*480):int(pos['y']*480)+int(pos['h']*480)],size=[160,160],mode='bilinear')[0])
                gray_seg.append(F.interpolate(gray[:,:,:,int(pos['y']*480):int(pos['y']*480)+int(pos['h']*480)],size=[160,160],mode='bilinear')[0])
            if num_split == 2:
                data_seg.append(data_seg[-1])
                gray_seg.append(gray_seg[-1])
                the_label.append(the_label[-1])
            data_seg,gray_seg = torch.stack(data_seg),torch.stack(gray_seg)
            return data_seg, np.array(the_label), gray_seg
        else:
            np.random.seed(i)
            the_index = np.random.choice(len(self.syn_label), 3, False).tolist()
            images = [self.transform_zone(self.syn_zone[i]) for i in the_index]
            label = [self.syn_label[i] for i in the_index]
            rgbs = torch.stack([self.rgb(the_image) for the_image in images])
            grays = torch.stack([F.interpolate(self.gray(the_image).unsqueeze(0),size=[160,160],mode='bilinear')[0] for the_image in images])
            label = np.array(label)
            return rgbs, label, grays