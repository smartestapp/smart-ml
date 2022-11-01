import os.path as osp
import os
import tqdm
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from models.Network import FeatureNet
import torchvision.models as models
import pdb
import random
import cv2

src = './'

class ZoneClassifier(object):
    def __init__(self, args, transform):
        # Set args to be shareable in the class
        self.args = args
        self.transform = transform
        self.model = FeatureNet(self.args)
        self.pre_model_load()

        # Get the images' paths and labels
        with open(src + 'kit_data.json') as json_file:  
            self.meta = json.load(json_file)

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()

    def eval(self, img_memb, thre=0.5):
        '''
        Process image membranes for evaluation.
        1. crop the membranes into small zones where each zone cover the region of displaying lines.
        2. make prediction for each zone.
        3. agreegate the prediction of zones and map them to the final diagnosis.
        '''
        height = 480 if self.args.resolution=='high' else 320
        zone_size = [160,160] if self.args.resolution=='high' else [64,100]
        img_zone = []
        img_memb = img_memb.unsqueeze(0)
        for idx,zone_key in enumerate(self.loc_split):
            pos = self.loc_split[zone_key]
            img_zone.append(F.upsample(img_memb[:,:,:,int(pos['y']*height):int(pos['y']*height)+int(pos['h']*height)],size=zone_size,mode='bilinear'))

        img_zone = torch.cat(img_zone,dim=0)
        with torch.no_grad():
            zone_logits = self.model(img_zone)[0]
        zone_prob = F.softmax(zone_logits, dim=1).cpu()
        zone_detction = zone_prob[:,1].numpy()
        zone_pred = np.where(zone_detction >= thre, 1, 0)
        diagonsis = self.diagonsis[tuple(zone_pred)]
        return {'zone_classification':zone_pred, 'detection score':zone_detction, 'diagnosis':diagonsis}
    
    def ActMap(self, img_memb, raw_memb, img_id=None):
        '''
        One customized function for activaty map visualization. not used in evaluation.
        '''
        height = 480 if self.args.resolution=='high' else 320
        zone_size = [160,160] if self.args.resolution=='high' else [64,100]
        map_size = (160,160) if self.args.resolution=='high' else (100,64)
        img_zone,raw_zone = [],[]
        img_memb = img_memb.unsqueeze(0)
        raw_memb = raw_memb.unsqueeze(0)
        for idx,zone_key in enumerate(self.loc_split):
            pos = self.loc_split[zone_key]
            img_zone.append(F.upsample(img_memb[:,:,:,int(pos['y']*height):int(pos['y']*height)+int(pos['h']*height)],size=zone_size,mode='bilinear'))
            raw_zone.append(F.upsample(raw_memb[:,:,:,int(pos['y']*height):int(pos['y']*height)+int(pos['h']*height)],size=zone_size,mode='bilinear'))


        img_zone = torch.cat(img_zone,dim=0)
        raw_zone = torch.cat(raw_zone,dim=0)
        with torch.no_grad():
            zone_maps = self.model(img_zone)[1]
        mask_cur = zone_maps.sum(dim=1).cpu()

        for i in range(mask_cur.size(0)):
            the_mask = (mask_cur[i] - mask_cur[i].min())#/(mask_cur.max() - mask_cur.min())
            the_mask = the_mask.numpy()
            heat_img = cv2.resize(the_mask, dsize=map_size)
            heat_img = heat_img * 255/1000
            heat_img = heat_img.astype(np.uint8)
            heat_img = cv2.applyColorMap(heat_img, cv2.COLORMAP_JET)
            zone_img = raw_zone[i].permute(1,2,0).numpy() * 255
            zone_img = Image.fromarray(zone_img.astype(np.uint8))
            if img_id == None:
                cv2.imwrite('{}_activation.png'.format(i), heat_img)
                zone_img.save('{}_membrance.png'.format(i))
            else:
                cv2.imwrite('{}_{}_activation.png'.format(img_id,i), heat_img)
                zone_img.save('{}_{}_membrane.png'.format(img_id,i))
    
    def pre_model_load(self):
        '''
        Load the model pretrained on ImageNet 
        (Just for initialization purpose or for customized evaluation, 
        the model will be reloaded later.)
        '''
        pretrained_dict = models.resnet18(pretrained=True)
        pretrained_dict = pretrained_dict.state_dict()
        del pretrained_dict['fc.weight']
        del pretrained_dict['fc.bias']
        pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
        model_dict = self.model.state_dict()
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def id_specification(self, kit_id,path=None):
        self.id_meta_load(kit_id)
        self.id_model_load(kit_id,path)

    def id_meta_load(self, kit_id='Quidel_Ag'):
        '''
        Load the meta data for zone split & diagonise lookup table
        self.loc_split: the position to split the membrane
        self.num_split; te number of zones in one membrane
        self.diag_map: how to make the final diagnosis according to 
            the zone classificaiton results
        '''
        self.loc_split = self.meta[kit_id]['dimensions']['zones']
        self.num_split = self.loc_split['n']
        del self.loc_split['n']
        self.diag_map = self.meta[kit_id]['diagnosis_mapping']
        for idx,key in enumerate(self.diag_map):
            zones_str = key[1:-1].split(', ')
            zones_cls = [int(zone_cls) for zone_cls in zones_str]
            if idx == 0:
                self.diagonsis = np.zeros([2]*len(zones_cls),dtype=np.int8)
            self.diagonsis[tuple(zones_cls)] = self.diag_map[key]

    def id_model_load(self, kit_id='Quidel_Ag', adapt_params_path=None):
        '''
        Load the checkpoint for the specificed test kit
        '''
        if adapt_params_path is None:
            adapt_params_path = osp.join(src, 'logs', kit_id, 'model_final.pth')
        adapt_params_dict = torch.load(adapt_params_path)['params']
        print('Loading model for {} from {}'.format(kit_id,adapt_params_path))
        self.model_dict = self.model.state_dict()
        self.model_dict.update(adapt_params_dict)
        self.model.load_state_dict(self.model_dict)
        self.model.eval()