""" Processing Tools for all datasets. """
import os,json,pdb,argparse
import os.path as osp
import numpy as np
from PIL import Image
import torch

def CommonLoad(args, dataset, label_id_dict):
    data,label = [],[]
    THE_PATH = osp.join(args.dataset_dir, dataset)
    this_folder = osp.join(THE_PATH, 'membranes')
    this_folder_images = os.listdir(this_folder)
    for image_path in this_folder_images:
        if '._' not in image_path and 'DS_Store' not in image_path and image_path[:-4] in label_id_dict:
            data.append(osp.join(this_folder, image_path))
            label.append(label_id_dict[image_path[:-4]])
    return data,label

def read_from_path(the_path):
    if the_path.split('.')[-1] == 'jpg':
        the_img = Image.open(the_path).convert('RGB')
    else:
        the_img = np.load(the_path).astype(dtype=np.uint8)
        the_img = np.flip(the_img, -1)
        the_img = Image.fromarray(the_img).convert('RGB')
    the_img = the_img.rotate(90,expand=True) if the_img.size[0] < the_img.size[1] else the_img.rotate(180,expand=True)
    return the_img


# https://androidkt.com/pytorch-image-augmentation-using-transforms/
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return ycbcr 

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

def image_convert(images,ratio):
    data = np.asarray(images)
    refpos = (int(data.shape[0]*ratio[0][1]),int(data.shape[1]*ratio[0][0])) # top left corner
    refend = (int(data.shape[0]*ratio[1][1]),int(data.shape[1]*ratio[1][0]))
    sub = data[refpos[0]:refend[0],refpos[1]:refend[1]]

    ycbcr = rgb2ycbcr(data)
    ysub = rgb2ycbcr(sub)
    yc = list(np.mean(ysub[:,:,i]) for i in range(3))
    for i in range(1,3):
        ycbcr[:,:,i] = np.clip(ycbcr[:,:,i] + (128-yc[i]), 0, 255)
    rgb = ycbcr2rgb(ycbcr)
    rgb = Image.fromarray(rgb)
    return rgb

# def shot_filtering(data,label,num_shot,num_zone=3): 
#     if num_shot > 50:
#         return data,label
        
#     weight_sum,filtered_data,filtered_label = {},[],[]
#     for the_idx,the_label in enumerate(label):
#         if num_zone == 3:
#             the_sum = the_label[0]*1+the_label[1]*2+the_label[2]*4
#         elif num_zone == 2:
#             the_sum = the_label[0]*1+the_label[1]*2
#         else:
#             raise ValueError('No Implementation')
#         if the_sum not in weight_sum:
#             weight_sum[the_sum] = {'detail':the_label,'idx':[]}
#         weight_sum[the_sum]['idx'].append(the_idx)

#     for the_sum in weight_sum:
#         the_idxes = np.random.permutation(len(weight_sum[the_sum]['idx']))[:num_shot]
#         for the_idx in the_idxes:
#             filtered_data.append(data[weight_sum[the_sum]['idx'][the_idx]])
#             filtered_label.append(weight_sum[the_sum]['detail'])
#     return filtered_data, filtered_label

# def single_test(path,zone_cnt):
#     data,label = [],[]
#     for image_path in os.listdir(path):
#         if '._' not in image_path and 'DS_Store' not in image_path:
#             data.append(osp.join(path, image_path))
#             label.append([0]*zone_cnt)
#     return data,label

# def failure_test(path,label_file,visual=1,lossy=False):
#     data,label = [],[]
#     label_id_dict = {}
#     prefix = 'Label_visual_numerical_zone' if visual>=1 else 'Label_conc_numerical_zone'
#     for filename,zone1,zone2 in zip(label_file['File name'].to_list(),label_file[prefix+'1'].to_list(),label_file[prefix+'2'].to_list()):
#         label_id_dict[filename[:-4]] = [zone1,zone2]
#     the_files = os.listdir(path)

#     the_format = 'jpg' if lossy else 'npy'
#     for the_file in the_files:
#         the_id,the_suffix = the_file.split('.')
#         if the_suffix == the_format:
#             data.append(osp.join(path, the_file))
#             label.append(label_id_dict[the_id])
#     return data,label

# def chart_test(path,label_file,visual=1,lossy=False):
#     data,label = [],[]
#     label_id_dict = {}
#     for filename,zone1,zone2 in zip(label_file['filename'].to_list(),label_file['zone1'].to_list(),label_file['zone2'].to_list()):
#         label_id_dict[filename[:-4]] = [zone1,zone2]
#     the_files = os.listdir(path)

#     the_format = 'jpg' if lossy else 'npy'
#     for the_file in the_files:
#         the_id,the_suffix = the_file.split('.')
#         if the_suffix == the_format:
#             try:
#                 label.append(label_id_dict[the_id.split('-')[0]])
#                 data.append(osp.join(path, the_file))
#             except:
#                 pass
#     return data,label

# def json_generation(kit,num_split,data,label,num_shot):
#     for json_id in range(1,num_split+1):
#         weight_sum,filtered_data,filtered_label = {},[],[]
#         for the_idx,the_label in enumerate(label):
#             if kit in ['ACON_Ag','DeepBlue_Ag']:
#                 the_sum = the_label[0]*1+the_label[1]*2
#             else:
#                 the_sum = the_label[0]*1+the_label[1]*2+the_label[2]*4
#             if the_sum not in weight_sum:
#                 weight_sum[the_sum] = {'detail':the_label,'idx':[]}
#             weight_sum[the_sum]['idx'].append(the_idx)
#         for the_sum in weight_sum:
#             the_idxes = np.random.permutation(len(weight_sum[the_sum]['idx']))[:num_shot]
#             for the_idx in the_idxes:
#                 filtered_data.append(data[weight_sum[the_sum]['idx'][the_idx]])
#                 filtered_label.append(weight_sum[the_sum]['detail'])
#         json_content = [[the_data,the_label] for (the_data,the_label) in zip(filtered_data,filtered_label)]

#         with open('/home/jiawei/DATA/COVID/Eval_Data_classify/COVID-APP/{}/{}_split{}.json'.format(kit,kit,json_id),'w') as json_file:
#             json.dump(json_content, json_file)
