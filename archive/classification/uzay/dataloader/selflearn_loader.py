""" Dataloader for all datasets. """
import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import json
import pdb



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

system_test_pool = ['ACON_Ag']

class DatasetInitializer(Dataset):
    """The class to load the dataset"""
    def __init__(self, setmode, args):
        # Set the path according to train, val and test   
        self.setmode = setmode

        with open(args.dataset_dir + 'kit_data_v5.json') as json_file:  
            json_data = json.load(json_file)

        self.datasets_path,self.datasets_label,self.json_list = {},{},{}
        the_setname = 'BTNx' if self.setmode in ['val','test'] and args.setname in system_test_pool else args.setname
        self.the_setname = the_setname
        data_path,self.label = self.dataset_initialization(args, the_setname)

        self.loc_split = json_data[the_setname]['dimensions']['zones']
        self.num_split = self.loc_split['n']
        del self.loc_split['n']

        self.data = []
        for the_path in data_path:
            the_img = Image.open(the_path).convert('RGB')
            the_item = the_img.rotate(90,expand=True) if the_img.size[0] < the_img.size[1] else the_img.rotate(180,expand=True)
            self.data.append(the_item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.transform(self.data[i]), self.label[i]
    
    def dataset_initialization(self, args, dataset):
        data,label = [],[]
        mode = 'train' if 'train' in self.setmode else 'test'
        if dataset == 'BTNx':
            if mode == 'test':
                THE_PATH = osp.join(args.dataset_dir, 'BTNx_mayo')
                label_file = pd.read_excel(osp.join(THE_PATH, 'labels_Mayo_test_BTNX_105_images.xlsx'))
                label_id_dict = {}
                for the_id, zone1, zone2, zone3 in zip(label_file['cu_key (S)'].to_list(),label_file['Control '].to_list(),label_file['Zone 1'].to_list(),label_file['Zone 2'].to_list()):
                    if zone1 == 1:
                        label_id_dict[the_id] = [zone1,zone2,zone3]
                
                for the_key in label_id_dict:
                    data.append(osp.join(THE_PATH, the_key))
                    label.append(label_id_dict[the_key])
            else:
                THE_PATH = osp.join(args.dataset_dir, 'BTNx_Eval', 'paper', mode)
                label_file = pd.read_excel(osp.join(args.dataset_dir, 'BTNx_Eval', 'labels.xlsx'))
                label_id_dict = {}
                for the_id, zone1, zone2, zone3 in zip(label_file['Sample ID'].to_list(),label_file['Zone 1'].to_list(),label_file['Zone 2'].to_list(),label_file['Zone 3'].to_list()):
                    label_id_dict[the_id] = [zone1,zone2,zone3]

                folders = [osp.join(THE_PATH, 'membranes_original'),osp.join(THE_PATH, 'membranes_cloud')] #if mode == 'test' else [osp.join(THE_PATH, 'membranes_original')]

                for this_folder in folders:
                    this_folder_images = os.listdir(this_folder)
                    for image_path in this_folder_images:
                        if '._' not in image_path and 'DS_Store' not in image_path:
                            data.append(osp.join(this_folder, image_path))
                            label.append(label_id_dict[image_path[:-4]])

        elif dataset == ['ACON_Ab','RapidConnect_Ab']:
            THE_PATH = osp.join(args.dataset_dir, 'COVID-APP', dataset)
            label_file = pd.read_excel(osp.join(THE_PATH, 'labels.xlsx'))
            label_id_dict = {}
            for the_id, zone1, zone2, zone3 in zip(label_file['Sample ID'].to_list(),label_file['Zone 1'].to_list(),label_file['Zone 2'].to_list(),label_file['Zone 3'].to_list()):
                if zone1 in [0,1] and zone2 in [0,1] and zone3 in [0,1]:
                    label_id_dict[the_id] = [zone1,zone2,zone3]
            this_folder = osp.join(THE_PATH, 'membranes')
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                if '._' not in image_path and 'DS_Store' not in image_path and image_path[:-4] in label_id_dict:
                    data.append(osp.join(this_folder, image_path))
                    label.append(label_id_dict[image_path[:-4]])
        
        elif dataset in ['ACON_Ag','DeepBlue_Ag']:
            THE_PATH = osp.join(args.dataset_dir, 'COVID-APP', dataset)
            label_file = pd.read_excel(osp.join(THE_PATH, 'labels.xlsx'))
            label_id_dict = {}
            for the_id, zone1, zone2 in zip(label_file['Sample ID'].to_list(),label_file['Zone 1'].to_list(),label_file['Zone 2'].to_list()):
                label_id_dict[the_id] = [zone1,zone2]
            this_folder = osp.join(THE_PATH, 'membranes')
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                if '._' not in image_path and 'DS_Store' not in image_path and image_path[:-4] in label_id_dict:
                    data.append(osp.join(this_folder, image_path))
                    label.append(label_id_dict[image_path[:-4]])   

        elif dataset == 'oraquick':
            THE_PATH = osp.join(args.dataset_dir, dataset, mode)
            label_list = os.listdir(THE_PATH)

            folders = [osp.join(THE_PATH, the_label) for the_label in label_list if os.path.isdir(osp.join(THE_PATH, the_label))]
            for this_folder in folders:
                this_folder_images = os.listdir(this_folder)
                for image_path in this_folder_images:
                    if '._' not in image_path and 'DS_Store' not in image_path:
                        data.append(osp.join(this_folder, image_path))
                        label.append(self.label_trans(this_folder.split('/')[-1],args.setname))
            weight_sum,filtered_data,filtered_label = {},[],[]
            for the_idx,the_label in enumerate(label):
                the_sum = the_label[0]*1+the_label[1]*2+the_label[2]*4
                if the_sum not in weight_sum:
                    weight_sum[the_sum] = {'detail':the_label,'idx':[]}
                weight_sum[the_sum]['idx'].append(the_idx)
            if self.num_shot is not None:
                weight_sum,filtered_data,filtered_label = {},[],[]
                for the_idx,the_label in enumerate(label):
                    the_sum = the_label[0]*1+the_label[1]*2+the_label[2]*4
                    if the_sum not in weight_sum:
                        weight_sum[the_sum] = {'detail':the_label,'idx':[]}
                    weight_sum[the_sum]['idx'].append(the_idx)
                for the_sum in weight_sum:
                    the_idxes = np.random.permutation(len(weight_sum[the_sum]['idx']))[:self.num_shot*2]
                    for the_idx in the_idxes:
                        filtered_data.append(data[weight_sum[the_sum]['idx'][the_idx]])
                        filtered_label.append(weight_sum[the_sum]['detail'])
                data = filtered_data
                label = filtered_label
                del filtered_data,filtered_label
            else:
                if 'train' in self.setmode:
                    print('No Filtering')
            if 'train' in self.setmode:
                for the_path,the_label in zip(data,label):
                    print(the_label,the_path)
        
        elif dataset == 'abbott':
            mode = self.setmode if 'train' in self.setmode else 'test'
            THE_PATH = osp.join(args.dataset_dir, 'abbott')
            label_file = pd.read_excel(osp.join(args.dataset_dir, 'abbott', 'labels.xlsx'))
            label_id_dict = {}
            for the_id, zone1, zone2 in zip(label_file['Sample ID'].to_list(),label_file['Zone 1'].to_list(),label_file['Zone 2'].to_list()):
                label_id_dict[the_id] = [zone1,zone2]

            folders = [osp.join(THE_PATH, 'membranes_original')]
            for this_folder in folders:
                this_folder_images = os.listdir(this_folder)
                for image_path in this_folder_images:
                    if '._' not in image_path and 'DS_Store' not in image_path:
                        data.append(osp.join(this_folder, image_path))
                        label.append(label_id_dict[image_path[:-4]])

        # if 'train' in self.setmode:
        #     for the_path,the_label in zip(data,label):
        #         print(the_label,the_path)
        print('There are {} images in the {} set for {} mode'.format(len(data),dataset,mode))
        return data,label

    def label_trans(self, label, name):
        if name == 'biomedomics':
            binary = []
            if 'C' in label:
                binary.append(1)
            else:
                binary.append(0)
            if 'G' in label:
                binary.append(1)
            else:
                binary.append(0)
            if 'M' in label:
                binary.append(1)
            else:
                binary.append(0)
            return binary
        elif name == 'oraquick':
            if label == 'positive':
                return [1,1]
            elif label == 'negative':
                return [1,0]
            elif label == 'invalid':
                return [0,0]
            else:
                raise ValueError('Invalid label name.')
        elif name == 'sd_igg':
            if label == '10':
                return [1,0]
            elif label == '11':
                return [1,1]
            else:
                raise ValueError('Invalid label name')
        elif name in ['maxim','aytu','BTNx']:
            if label == '000':
                return [0,0,0]
            elif label == '100':
                return [1,0,0]
            elif label == '101':
                return [1,0,1]
            elif label == '110':
                return [1,1,0]
            elif label == '111':
                return [1,1,1]
            else:
                raise ValueError('Invalid label name')
        else:
            raise ValueError('Invalid set name.')

class SelfLoader(DatasetInitializer):
    def __init__(self, setmode, args, train_aug=False):
        super(SelfLoader,self).__init__(setmode, args)

        # Transformation
        if train_aug:
            self.transform = transforms.Compose([
                transforms.Resize((65,325)),
                transforms.RandomRotation(5),
                transforms.Resize((64,320)),
                transforms.RandomVerticalFlip()])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((64,320))])
        self.rgb = transforms.Compose([
            transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                 np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        self.gray = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image = self.transform(self.data[i])
        rgb,gray = self.rgb(image),self.gray(image)
        return rgb, self.label[i], gray