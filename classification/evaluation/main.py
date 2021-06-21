##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Main function for this repo. """
import argparse
import torch
import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import json
import pdb
import pprint
import pandas as pd
from classifier_ss import ZoneClassifier

dataset_pool = ['Quidel_Ag','BTNx','ACON_Ab','ACON_Ag','DeepBlue_Ag','RapidConnect_Ab','abbott','oraquick','biomedomics','maxim','aytu','sd_igg']

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def set_gpu(cuda_device):
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    print('Using gpu:', cuda_device)

def read_image(path):
    if path.split('.')[-1] == 'npy':
        the_img = np.load(path).astype(dtype=np.uint8)
        the_img = np.flip(the_img, -1)
        the_img = Image.fromarray(the_img).convert('RGB')
    elif the_path.split('.')[-1] == 'jpg':
        the_img = Image.open(the_path).convert('RGB')
    return the_img

def postprocessing(label,results):
    # For each membrane, you can use this function for any needed processing
    # zone_pred = results['zone_classification'].tolist()
    # zone_score = results['detection score'].tolist()
    zone_pred = [0,0]
    zone_pred[0] = 1 if results['detection score'][0] > 0.95 else 0
    zone_pred[1] = 1 if results['detection score'][1] > 0.0147 else 0
    zone_match = np.where(np.equal(label,zone_pred),1,0)
    memb_match = np.where(np.mean(zone_match)==1,1,0)
    return zone_match,memb_match


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic parameters
    parser.add_argument('--gpu', default='0') # GPU id
    parser.add_argument('--kit_id', type=str, default='Quidel_Ag', choices=dataset_pool)
    parser.add_argument('--thredhold', type=float, default=0.5)
    parser.add_argument('--resolution',type=str, default='low',choices=['high','low'])

    args = parser.parse_args()
    
    # Set the GPU id
    if torch.cuda.is_available():
        set_gpu(args.gpu)
        torch.backends.cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.Resize((160,480) if args.resolution=='high' else (64,320)),
        transforms.ToTensor(),
        transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    classifier = ZoneClassifier(args, transform)
    classifier.id_specification(args.kit_id)

    #### =======================================================================================================================================
    # records = {'label':[],'zone_match':[],'memb_match':[],'detection':[],'zone_pred':[],'diagnosis':[],'file':[]}
    # root_kit = '/home/jiawei/DATA/COVID/Eval_Data_classify/COVID-APP/Quidel_Ag/'
    # label_file = pd.read_excel(osp.join(root_kit,'label_full.xlsx'))
    # label_id_dict = {}

    # for the_folder, the_id, zone1, zone2 in zip(label_file['the_folder'].to_list(), label_file['Sample ID'].to_list(),label_file['Zone 1'].to_list(),label_file['Zone 2'].to_list()):

    #     label_id_dict[the_id] = [zone1,zone2]
    #     img_path = os.path.join(root_kit,'membrane_fullV3',the_folder,'{}.npy'.format(the_id))
    #     try:
    #         img_memb = read_image(img_path)
    #         img_memb = img_memb.rotate(90,expand=True)
    #     except:
    #         print(img_path)
    #         continue
        
    #     img_memb = transform(img_memb)
    #     img_memb = img_memb.cuda() if torch.cuda.is_available() else img_memb
    #     results = classifier.eval(img_memb,args.thredhold)
    #     zone_match,memb_match = postprocessing(label_id_dict[the_id],results)

    #     records['detection'].append(results['detection score'].tolist())
    #     records['zone_pred'].append(results['zone_classification'].tolist())
    #     records['diagnosis'].append(results['diagnosis'].tolist())
    #     records['label'].append(label_id_dict[the_id])
    #     records['zone_match'].append(zone_match.tolist())
    #     records['memb_match'].append(memb_match.tolist())
    #     records['file'].append(img_path)
    
    # log_root = './'
    # df_records = pd.DataFrame.from_dict(records)
    # df_records.to_excel(log_root+'/prediction.xlsx')
    #### =======================================================================================================================================
    records = {'label':[],'zone_match':[],'memb_match':[],'detection':[],'zone_pred':[],'diagnosis':[],'file':[]}
    root_kit = '/home/jiawei/DATA/COVID/Eval_Data_classify/COVID-APP/Quidel_Failure_04_64/'
    label_file = pd.read_excel(osp.join(root_kit,'labels.xlsx'),sheet_name='to_test')
    label_id_dict = {}

    for the_id, zones in zip(label_file['sample_id'].to_list(),label_file['Blue label'].to_list()):
        zone1,zone2 = zones[1:-1].split(',')
        label_id_dict[the_id] = [float(zone1),float(zone2)]
        img_path = os.path.join(root_kit,'membranes','{}-mem.npy'.format(the_id))
        try:
            img_memb = read_image(img_path)
            img_memb = img_memb.rotate(90,expand=True) if img_memb.size[0] < img_memb.size[1] else img_memb
        except:
            print(img_path)
            continue
        
        raw_transform = transforms.Compose([
            transforms.Resize((160,480) if args.resolution=='high' else (64,320)),
            transforms.ToTensor()])
        raw_memb = raw_transform(img_memb)
        img_memb = transform(img_memb)
        img_memb = img_memb.cuda() if torch.cuda.is_available() else img_memb
        classifier.ActMap(img_memb,raw_memb,the_id)
        break
    #     results = classifier.eval(img_memb,args.thredhold)
    #     zone_match,memb_match = postprocessing(label_id_dict[the_id],results)

    #     records['detection'].append(results['detection score'].tolist())
    #     records['zone_pred'].append(results['zone_classification'].tolist())
    #     records['diagnosis'].append(results['diagnosis'].tolist())
    #     records['label'].append(label_id_dict[the_id])
    #     records['zone_match'].append(zone_match.tolist())
    #     records['memb_match'].append(memb_match.tolist())
    #     records['file'].append(img_path)
    
    # log_root = './'
    # df_records = pd.DataFrame.from_dict(records)
    # df_records.to_excel(log_root+'/prediction.xlsx')