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

system_test_pool = []#['ACON_Ag']

class DatasetInitializer(Dataset):
    """The class to load the dataset"""
    def __init__(self, setmode, args, num_shot=None):
        # Set the path according to train, val and test   
        self.setmode = setmode
        self.num_shot = num_shot
        self.datasets_path,self.datasets_label,self.json_list = {},{},{}
        the_setname = 'BTNx' if self.setmode in ['val','test'] and args.setname in system_test_pool else args.setname
        self.the_setname = the_setname

        with open(args.dataset_dir + 'kit_data_v5.json') as json_file:  
            json_data = json.load(json_file)

        if setmode == 'trainall':
            for set_assist in ['BTNx','ACON_Ab','DeepBlue_Ag']:
                self.datasets_path[set_assist],self.datasets_label[set_assist] = self.dataset_initialization(args, set_assist)
                self.json_list[set_assist] =  json_data[set_assist]['dimensions']['zones'].copy()
        else:
            self.datasets_path[the_setname],self.datasets_label[the_setname] = self.dataset_initialization(args, the_setname)
            
        self.loc_split = json_data[the_setname]['dimensions']['zones']
        self.num_split = self.loc_split['n']
        del self.loc_split['n']

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
                
                weight_sum,filtered_data,filtered_label = {},[],[]
                for the_idx,the_label in enumerate(label):
                    the_sum = the_label[0]*1+the_label[1]*2+the_label[2]*4
                    if the_sum not in weight_sum:
                        weight_sum[the_sum] = {'detail':the_label,'idx':[]}
                    weight_sum[the_sum]['idx'].append(the_idx)

                if self.num_shot is not None and  self.setmode == 'train':
                    weight_sum,filtered_data,filtered_label = {},[],[]
                    for the_idx,the_label in enumerate(label):
                        the_sum = the_label[0]*1+the_label[1]*2+the_label[2]*4
                        if the_sum not in weight_sum:
                            weight_sum[the_sum] = {'detail':the_label,'idx':[]}
                        weight_sum[the_sum]['idx'].append(the_idx)

                    for the_sum in weight_sum:
                        the_idxes = np.random.permutation(len(weight_sum[the_sum]['idx']))[:self.num_shot]
                        for the_idx in the_idxes:
                            filtered_data.append(data[weight_sum[the_sum]['idx'][the_idx]])
                            filtered_label.append(weight_sum[the_sum]['detail'])
                    data =  filtered_data # sample_great if self.num_shot == 3 else filtered_data
                    label = filtered_label
                    del filtered_data,filtered_label
                else:
                    if 'train' in self.setmode:
                        print('No Filtering')

        elif dataset == 'ACON_Ab':
            THE_PATH = osp.join(args.dataset_dir, 'COVID-APP', 'ACON_Ab')
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
                    the_idxes = np.random.permutation(len(weight_sum[the_sum]['idx']))[:self.num_shot]
                    for the_idx in the_idxes:
                        filtered_data.append(data[weight_sum[the_sum]['idx'][the_idx]])
                        filtered_label.append(weight_sum[the_sum]['detail'])
                data = filtered_data
                label = filtered_label
                del filtered_data,filtered_label
            else:
                if 'train' in self.setmode:
                    print('No Filtering')
        
        elif dataset == 'RapidConnect_Ab':

            THE_PATH = osp.join(args.dataset_dir, 'COVID-APP', 'RapidConnect_Ab')
            label_file = pd.read_excel(osp.join(THE_PATH, 'labels.xlsx'))
            label_id_dict = {}
            for the_id, zone1, zone2, zone3 in zip(label_file['Sample ID'].to_list(),label_file['Zone 1'].to_list(),label_file['Zone 2'].to_list(),label_file['Zone 3'].to_list()):
                label_id_dict[the_id] = [zone1,zone2,zone3]
            this_folder = osp.join(THE_PATH, 'membranes')
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                if '._' not in image_path and 'DS_Store' not in image_path and image_path[:-4] in label_id_dict:
                    data.append(osp.join(this_folder, image_path))
                    label.append(label_id_dict[image_path[:-4]])
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
        
        elif dataset == 'ACON_Ag':
            THE_PATH = osp.join(args.dataset_dir, 'COVID-APP', 'ACON_Ag')
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
        
        elif dataset == 'DeepBlue_Ag':
            THE_PATH = osp.join(args.dataset_dir, 'COVID-APP', 'DeepBlue_Ag')
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
            weight_sum,filtered_data,filtered_label = {},[],[]
            for the_idx,the_label in enumerate(label):
                the_sum = the_label[0]*1+the_label[1]*2
                if the_sum not in weight_sum:
                    weight_sum[the_sum] = {'detail':the_label,'idx':[]}
                weight_sum[the_sum]['idx'].append(the_idx)
            if self.num_shot is not None:
                weight_sum,filtered_data,filtered_label = {},[],[]
                for the_idx,the_label in enumerate(label):
                    the_sum = the_label[0]*1+the_label[1]*2
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
        # for the_path,the_label in zip(data,label):
        #     print(the_label,the_path)
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

class AdaptTrainer(DatasetInitializer):
    def __init__(self, setmode, args, num_shot=None, color_mode=1):
        super(AdaptTrainer,self).__init__(setmode, args, num_shot)
        self.supp_aug = args.supp_aug
        print(color_mode)
        
        memb2zone = transforms.Compose([
            transforms.Resize((65,325)),
            transforms.RandomRotation(5),
            transforms.Resize((64,320)),
            transforms.ToTensor()
        ])
        zone_tensor2pil = transforms.ToPILImage()

        zones = []
        self.label = []
        for the_set in self.datasets_path:
            data_path = self.datasets_path[the_set]
            data_label = self.datasets_label[the_set]
            data_split = self.json_list[the_set] if the_set in self.json_list else self.loc_split.copy() 
            if 'n' in data_split:
                del data_split['n']
            for the_path in data_path:
                zone = []
                the_img = Image.open(the_path).convert('RGB')
                the_item = the_img.rotate(90,expand=True) if the_img.size[0] < the_img.size[1] else the_img.rotate(180,expand=True)
                the_item = memb2zone(the_item).unsqueeze(0)
                for zone_key in data_split:
                    pos = data_split[zone_key]
                    zone.append(F.upsample(the_item[:,:,:,int(pos['y']*320):int(pos['y']*320)+int(pos['h']*320)],size=[64,100],mode='bilinear'))
                zones.extend(zone)
            self.label.extend(np.array(data_label).reshape(-1).tolist())
        self.data = zones
        del zones
        self.data = [zone_tensor2pil(the_zone[0]) for the_zone in self.data]

        if color_mode == 0:
            self.transform = transforms.Compose([
                transforms.RandomCrop([64,100],padding=(20,0)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), 
                                    np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif color_mode == 1:
            self.transform = transforms.Compose([
                transforms.RandomCrop([64,100],padding=(20,4)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), 
                                    np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif color_mode == 2:
            self.transform = transforms.Compose([
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop([64,100],padding=(20,0),padding_mode='reflect'),
                RandAugmentPC(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), 
                                    np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        else:
            raise ValueError('Not Implemented Error')

        # for idx,the_data in enumerate(self.data):
        #     the_data.save('./temp/orig{}.jpg'.format(idx))
        #     the_new = zone_tensor2pil(self.transform(the_data))
        #     the_new.save('./temp/new{}.jpg'.format(idx))
        # pdb.set_trace()

    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        imgs = torch.stack([self.transform(self.data[i]) for _ in range(self.supp_aug)]) # list(map(lambda x: self.transform(x), support_xs))
        labels = np.array([self.label[i]]*self.supp_aug)
        return imgs,labels

class AdaptValider(DatasetInitializer):
    def __init__(self, args,trained_path=None):
        super(AdaptValider,self).__init__('val', args)
        self.transform = transforms.Compose([
            transforms.Resize((64,320)),
            transforms.ToTensor(),
            transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        assert len(self.json_list) == 0 and self.the_setname in self.datasets_path

        trained_path = trained_path[self.the_setname] if self.the_setname in ['ACON_Ab','DeepBlue_Ag','RapidConnect_Ab'] else None
        self.data,self.label = [],[]
        for the_key in self.datasets_path:
            data_path = self.datasets_path[the_key]
            data_label = self.datasets_label[the_key]
            for the_path,the_label in zip(data_path,data_label):
                # if trained_path is not None:
                #     if the_path in trained_path:
                #         continue
                the_img = Image.open(the_path).convert('RGB')
                the_img = the_img.rotate(90,expand=True) if the_img.size[0] < the_img.size[1] else the_img.rotate(180,expand=True)
                self.data.append(the_img)
                self.label.append(the_label)
        print('There are {} images for {} test mode'.format(len(self.data),self.the_setname))

    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.transform(self.data[i]), self.label[i]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--setname', type=str, default='BTNx', choices=['oraquick','biomedomics','maxim','aytu','sd_igg','BTNx'])
    parser.add_argument('--dataset_dir', type=str, default='/home/jiawei/DATA/COVID/Eval_Data_classify/') # Dataset folder
    args = parser.parse_args()


# sample_justsoso = [
#     '/home/jiawei/DATA/COVID/Eval_Data_classify/BTNx_Eval/paper/train/membranes_cloud/IMG_6153.jpg',
#     '/home/jiawei/DATA/COVID/Eval_Data_classify/BTNx_Eval/paper/train/membranes_original/IMG_6170.jpg',
#     '/home/jiawei/DATA/COVID/Eval_Data_classify/BTNx_Eval/paper/train/membranes_cloud/5.6_btnxc_batch_4_AB49.jpg',
#     '/home/jiawei/DATA/COVID/Eval_Data_classify/BTNx_Eval/paper/train/membranes_original/5.6_btnxc_batch_10_AB116.jpg',
#     '/home/jiawei/DATA/COVID/Eval_Data_classify/BTNx_Eval/paper/train/membranes_cloud/IMG_6159.jpg',
#     '/home/jiawei/DATA/COVID/Eval_Data_classify/BTNx_Eval/paper/train/membranes_cloud/5.5_btnxc_batch_8_S459.jpg',
#     '/home/jiawei/DATA/COVID/Eval_Data_classify/BTNx_Eval/paper/train/membranes_original/5.5_btnxc_batch_16_AB317.jpg',
#     '/home/jiawei/DATA/COVID/Eval_Data_classify/BTNx_Eval/paper/train/membranes_cloud/IMG_5468-weak pos IgM.jpg',
#     '/home/jiawei/DATA/COVID/Eval_Data_classify/BTNx_Eval/paper/train/membranes_cloud/5.6_btnxc_batch_1_AB10.jpg',
#     '/home/jiawei/DATA/COVID/Eval_Data_classify/BTNx_Eval/paper/train/membranes_original/5.6_btnxc_batch_7_AB83.jpg',
#     '/home/jiawei/DATA/COVID/Eval_Data_classify/BTNx_Eval/paper/train/membranes_cloud/5.6_btnxc_batch_7_AB83.jpg',
#     '/home/jiawei/DATA/COVID/Eval_Data_classify/BTNx_Eval/paper/train/membranes_cloud/5.6_btnxc_batch_6.AB78.jpg']

# sample_great = [
#     '/home/jiawei/DATA/COVID/Eval_Data_classify/BTNx_Eval/paper/train/membranes_cloud/5.5_btnxc_batch_1_AB282.jpg',
#     '/home/jiawei/DATA/COVID/Eval_Data_classify/BTNx_Eval/paper/train/membranes_original/5.6_btnxc_batch_3_AB35.jpg',
#     '/home/jiawei/DATA/COVID/Eval_Data_classify/BTNx_Eval/paper/train/membranes_cloud/5.6_btnxc_batch_17_AB209.jpg',
#     '/home/jiawei/DATA/COVID/Eval_Data_classify/BTNx_Eval/paper/train/membranes_original/5.5_btnxc_batch_5_S632.jpg',
#     '/home/jiawei/DATA/COVID/Eval_Data_classify/BTNx_Eval/paper/train/membranes_original/5.6_btnxc_batch_8_AB97.jpg',
#     '/home/jiawei/DATA/COVID/Eval_Data_classify/BTNx_Eval/paper/train/membranes_original/5.5_btnxc_batch_13_SAV13.jpg',
#     '/home/jiawei/DATA/COVID/Eval_Data_classify/BTNx_Eval/paper/train/membranes_cloud/5.6_btnxc_batch_19_SAV50.jpg',
#     '/home/jiawei/DATA/COVID/Eval_Data_classify/BTNx_Eval/paper/train/membranes_original/AB317_13.jpg',
#     '/home/jiawei/DATA/COVID/Eval_Data_classify/BTNx_Eval/paper/train/membranes_cloud/5.6_btnxc_batch_20_S614.jpg',
#     '/home/jiawei/DATA/COVID/Eval_Data_classify/BTNx_Eval/paper/train/membranes_original/5.6_btnxc_batch_7_AB92.jpg',
#     '/home/jiawei/DATA/COVID/Eval_Data_classify/BTNx_Eval/paper/train/membranes_cloud/5.6_btnxc_batch_7_AB92.jpg',
#     '/home/jiawei/DATA/COVID/Eval_Data_classify/BTNx_Eval/paper/train/membranes_cloud/5.6_btnxc_batch_16_AB200.jpg'
#     ]
