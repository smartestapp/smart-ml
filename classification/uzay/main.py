"""
Main script for running inference on trained classification models.

Sample Calls:

python main.py --kit_id=BTNx --membranes_dir=../../data/btnx_field_membranes/ --output_name=btnx_mayo_eval
python main.py --kit_id=ACON_Ab --membranes_dir=../../data/aconab_membranes/ --output_name=aconab_eval
python main.py --kit_id=DeepBlue_Ag --membranes_dir=../../data/deepblue_membranes/ --output_name=deepblue_eval

python main.py --kit_id=RapidConnect_Ab --membranes_dir=../../data/rapidconnect_membranes/ --output_name=rapidconnectab_eval
python main.py --kit_id=ACON_Ag --membranes_dir=../../data/aconag_membranes/ --output_name=aconag_eval

python main.py --kit_id=Paramount_Ag --membranes_dir=../../data/paramountag_membranes/ --output_name=paramountag_eval

python main.py --kit_id=ACON_Ag --membranes_dir=../../data/study-images-aconag-membranes --output_name=study-images-aconag_eval
python main.py --kit_id=ACON_Ag --membranes_dir=../../data/study-images-aconag-membranes-2 --output_name=study-images-aconag_eval_2

python main.py --kit_id=ACON_Ag --membranes_dir=../../data/study-images-aconag-membranes --output_name=study-images-aconag_tight_eval --json_path=kit_data_v5_tight.json

python main.py --kit_id=Quidel_Ag --membranes_dir=../../data/quidelagrsv_membranes --output_name=quidelagrsv_eval
python main.py --kit_id=Quidel_Ag --membranes_dir=../../data/quidelagsars_membranes --output_name=quidelagsars_eval
python main.py --kit_id=Quidel_Ag --membranes_dir=../../data/quidelag_membranes --output_name=quidelag_eval

python main.py --kit_id=AccessBio_Ag --membranes_dir=../../data/accessbio_membranes --output_name=accessbio_eval --white_balance --threshold_rejection
python main.py --kit_id=Quidel_Ag --membranes_dir=../../data/sialab_quidelag_membranes_V2 --output_name=sialab_quidelag_eval

python main.py --kit_id=Quidel_Ag --model_version=v5 --membranes_dir=../../data/Dilutionstudy_sialab_quidelag_membranes_V2 --gradcam --output_name=Dilutionstudy_sialab_quidelag_membranes_V2
python main.py --kit_id=Quidel_Ag --model_version=v5 --membranes_dir=../../data/first_quidel_set_membranes --gradcam --output_name=first_quidel_set_membranes
python main.py --kit_id=Quidel_Ag --model_version=v5 --membranes_dir=../../data/nih_quidelag_membranes --gradcam --output_name=nih_quidelag_membranes_V2
python main.py --kit_id= Quidel_Ag --model_version=v6 --membranes_dir=quidel_failure_membranes --output_name=CHECK_NOW
"""

import torch
import argparse
from models.NetworkPre import FeatureNet
from torchvision import transforms
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import os
from PIL import Image
import torch.nn.functional as F
import cv2
import torchvision.models as models
import torchvision
from utils import visualize_cam, Normalize
from gradcam import GradCAM, GradCAMpp
from torchvision.utils import make_grid, save_image
from torch.autograd import Variable
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='ResNet18ORI', choices=['ResNet18', 'ResNet18ORI'])
parser.add_argument('--kit_id', type=str, required=True, choices=['BTNx', 'ACON_Ab', 'ACON_Ag', 'DeepBlue_Ag', 'RapidConnect_Ab', 'Paramount_Ag', 'Quidel_Ag', 'AccessBio_Ag'])
parser.add_argument('--model_version', type=str, default='v6')
parser.add_argument('--json_path', type=str, default='kit_data_v5.json')
parser.add_argument('--membranes_dir', type=str, required=True)
parser.add_argument('--output_name', type=str, required=True)
parser.add_argument('--input_is_zones', default=False, action='store_true')
parser.add_argument('--gradcam', default=False, action='store_true')
parser.add_argument('--white_balance', default=False, action='store_true')
parser.add_argument('--threshold_rejection', default=False, action='store_true')
args = parser.parse_args()


def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        # plt.imshow(image.astype(np.uint8), cmap=cmap, norm=norm, interpolation=interpolation)
        plt.imshow(image, cmap=cmap, norm=norm, interpolation=interpolation)
        i += 1
    plt.show()

if not os.path.exists(args.membranes_dir):
    raise ValueError('The specified path for membranes_dir=%s does not exist!' % args.membranes_dir)

transform_no_normalization = transforms.Compose([transforms.Resize((64,320)), transforms.ToTensor()])

transform = transforms.Compose([transforms.Resize((64,320)), 
                                transforms.ToTensor(), 
                                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), 
                                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])


######## PART WE USE TO GET THE MODEL .PT FILE #############
model = FeatureNet(args=args, mode='cloud_ss', n_class=2, flag_meta=True)

load_path = 'logs/%s/%s/max_acc.pth' % (args.model_version, args.kit_id)
print('Loading Model from Path: ', load_path)
if not os.path.exists(load_path):
    print('Path not found; loading BTNx model!')
    load_path = 'logs/%s/BTNx/max_acc.pth' % args.model_version

print('Loaded Model has Keys: ', torch.load(load_path, map_location=torch.device('cpu')).keys())

if args.threshold_rejection:
    print('Threshold: ', torch.load(load_path, map_location=torch.device('cpu'))['thre'])   # threshold
    thresholds = torch.load(load_path, map_location=torch.device('cpu'))['thre']
    # print('Ratio: ', torch.load(load_path, map_location=torch.device('cpu'))['white_thre'])  # ratio

model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu'))['params'])
model.eval()
#############################################################

############## PART FOR WHITE COLOR BALANCE #################
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

def image_convert(images, ratio):
    data = np.asarray(images)
    refpos = (int(data.shape[0]*eval(ratio[0])[1]), int(data.shape[1]*eval(ratio[0])[0])) # top left corner
    refend = (int(data.shape[0]*eval(ratio[1])[1]), int(data.shape[1]*eval(ratio[1])[0]))
    sub = data[refpos[0]:refend[0],refpos[1]:refend[1]]

    ycbcr = rgb2ycbcr(data)
    ysub = rgb2ycbcr(sub)
    yc = list(np.mean(ysub[:,:,i]) for i in range(3))
    for i in range(1,3):
        ycbcr[:,:,i] = np.clip(ycbcr[:,:,i] + (128-yc[i]), 0, 255)
    rgb = ycbcr2rgb(ycbcr)
    rgb = Image.fromarray(rgb)
    return rgb
#############################################################

if args.gradcam:
    ## GRADCAM SETUP ##
    cam_dict = dict()

    model_dict = dict(type='resnet', arch=model, layer_name='layer1', input_size=(64, 100))
    gradcam = GradCAM(model_dict, True)
    gradcampp = GradCAMpp(model_dict, True)
    cam_dict['resnet_layer1'] = [gradcam, gradcampp]

    model_dict = dict(type='resnet', arch=model, layer_name='layer2', input_size=(64, 100))
    gradcam = GradCAM(model_dict, True)
    gradcampp = GradCAMpp(model_dict, True)
    cam_dict['resnet_layer2'] = [gradcam, gradcampp]

    model_dict = dict(type='resnet', arch=model, layer_name='layer3', input_size=(64, 100))
    gradcam = GradCAM(model_dict, True)
    gradcampp = GradCAMpp(model_dict, True)
    cam_dict['resnet_layer3'] = [gradcam, gradcampp]

    model_dict = dict(type='resnet', arch=model, layer_name='layer4', input_size=(64, 100))
    gradcam = GradCAM(model_dict, True)
    gradcampp = GradCAMpp(model_dict, True)
    cam_dict['resnet_layer4'] = [gradcam, gradcampp]

KIT_DATA = json.load(open(args.json_path, 'r'))[args.kit_id]

NUM_ZONES = KIT_DATA['dimensions']['zones']['n']

if args.white_balance:
    print('Got Ratio: ', (eval(KIT_DATA['ratio'][0]), eval(KIT_DATA['ratio'][1])))

print('GOT NUM_ZONES=%d' % NUM_ZONES)
columns = ['Sample ID']
if not args.input_is_zones:
    for i in range(NUM_ZONES): 
        columns.append('Pred_%d' % (i + 1))
        columns.append('Pred_%d_Confidence' % (i + 1))
else:
    columns.append('Pred')
    columns.append('Pred_Confidence')

DF = pd.DataFrame({}, columns=columns)
current_idx = 0

for root, dirs, filenames in os.walk(args.membranes_dir):
    for filename in tqdm(filenames, desc='Iterating Images'):
        if filename.startswith('.'):
            continue

        membrane_original = Image.open(os.path.join(root, filename)).convert('RGB').rotate(90, expand=True)

        if args.white_balance:
            membrane_original = image_convert(membrane_original, KIT_DATA['ratio'])

        membrane = transform(membrane_original).unsqueeze(0)

        ############## GRADCAM ANALYSIS ##############
        if args.gradcam:
            visualization_membrane = transform_no_normalization(membrane_original).unsqueeze(0)

            for i in range(NUM_ZONES):
                pos = KIT_DATA['dimensions']['zones']['zone%d' % (i + 1)]
                zone_model = F.upsample(membrane[:, :, :, int(pos['y'] * 320): int(pos['y'] * 320) + int(pos['h'] * 320)], size=[64, 100], mode='bilinear')
                zone_viz = F.upsample(visualization_membrane[:, :, :, int(pos['y'] * 320): int(pos['y'] * 320) + int(pos['h'] * 320)], size=[64, 100], mode='bilinear')

                visualization_images = []
                for gradcam, gradcam_pp in cam_dict.values():
                    mask, _ = gradcam(zone_model)
                    # print('mask: ', mask.shape)
                    heatmap, result = visualize_cam(mask, zone_viz)
                    # print('heatmap: ', heatmap.shape)

                    mask_pp, _ = gradcam_pp(zone_model)
                    heatmap_pp, result_pp = visualize_cam(mask_pp, zone_viz)
                    visualization_images.append(torch.stack([zone_viz.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0))

                images = make_grid(torch.cat(visualization_images, 0), nrow=5)
                if not os.path.exists(os.path.join('%s_gradcam_outputs' % args.output_name, filename)):
                    os.makedirs(os.path.join('%s_gradcam_outputs' % args.output_name, filename))

                prediction = F.softmax(model(zone_model)[0], dim=1).argmax(dim=1).cpu().detach().numpy().tolist()[0]
                save_image(images, os.path.join('%s_gradcam_outputs' % args.output_name, filename, 'zone_no=%d_prediction=%d.jpg' % (i + 1, prediction)))
            ##########################################
            
        zones = []
        if not args.input_is_zones:
            for i in range(NUM_ZONES):
                pos = KIT_DATA['dimensions']['zones']['zone%d' % (i + 1)]

                ######## SAVING ZONE IMAGES SENT AS INPUT TO THE MODEL FOR INSPECTION #############
                base_zone = membrane[:, :, :, int(pos['y'] * 320): int(pos['y'] * 320) + int(pos['h'] * 320)][0].permute(1, 2, 0).numpy()
                zone = F.upsample(membrane[:, :, :, int(pos['y'] * 320): int(pos['y'] * 320) + int(pos['h'] * 320)], size=[64, 100], mode='bilinear')[0].permute(1, 2, 0).numpy()
                cv2.imwrite('%d_base.jpg' % i, np.array(base_zone * (255 / np.max(base_zone)), dtype=int))
                cv2.imwrite('%d_model.jpg' % i, np.array(zone * (255 / np.max(zone)), dtype=int))
                ###################################################################################

                zones.append(F.upsample(membrane[:, :, :, int(pos['y'] * 320): int(pos['y'] * 320) + int(pos['h']*320)], size=[64, 100], mode='bilinear'))

            zones = torch.cat(zones, dim=0)
            logits, _ = model(zones)

            confidence_scores = F.softmax(logits).cpu().detach().numpy().tolist()
            predictions = F.softmax(logits, dim=1).argmax(dim=1).cpu().detach().numpy().tolist()

            if args.threshold_rejection:
                positive_confidence_scores = [p[1] for p in confidence_scores]
                for k, P_pos in enumerate(positive_confidence_scores):
                    if thresholds[0] <= P_pos <= thresholds[1]:
                        predictions[k] = -1

        else:
            logits, _ = model(membrane)
            confidence_scores = F.softmax(logits).cpu().detach().numpy().tolist()
            predictions = F.softmax(logits, dim=1).argmax(dim=1).cpu().detach().numpy().tolist()

        entry = [filename]
        for prediction, confidence_score in list(zip(predictions, confidence_scores)):
            entry += [prediction, confidence_score]

        DF.loc[current_idx] = entry
        current_idx += 1

        # print(filename, predictions, confidence_scores)


print(DF.head())
DF.to_csv('eval/%s.csv' % args.output_name)
