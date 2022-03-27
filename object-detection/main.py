"""
Main script for training, evaluation, and running inference on object detection models.
"""

import os
import numpy as np
import cv2
import math
import json
import argparse
import traceback

from tqdm import tqdm

import torch
from torch.utils.data import ConcatDataset, DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR

from engine import train_one_epoch, evaluate
import utils 

from custom_utils.data_utils import get_lfa_dataset
from custom_utils.model_utils import get_instance_segmentation_model

from sklearn.metrics import jaccard_score

# (0) Configure device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('DEVICE: ', DEVICE)

# (1) Set hyperparameters
# TODO: The hyperparameters below can be converted to command-line arguments if desired. The current design choice was 
#       adopted due to the large number of arguments.
# (1.1) Set hyperparameters for training
###################################################################################################
NUM_CLASSES = 3  # ('background', 'kit', 'membrane')
BATCH_SIZE = 2  # Feel free to increase this to the limit that your GPU handles
HIDDEN_SIZE = 256  # Hidden layer size for our Mask-RCNN model
LEARNING_RATE = 5E-6  # NOTE: Pretraining is done with 5E-5 and finetuning is done with 5E-6, 1E-6, or 5E-7
SEED = 42  # Set seeds for reproducibility
RESIZE = True  # This can be kept as is; we usually need to resize images so they fit into the GPU for training
SHOTS = None  # Set this to a integer if you only want to train with a few images (e.g. SHOTS=5 => only 5 images are used)
NUM_REPEATS = 1  # How many times you want to repeat the experiment (e.g. NUM_REPEATS=2 => model is trained two times for NUM_EPOCHS to get a more consistent performance metric)
NUM_EPOCHS = 100 # Usuaully we use 100 here, but NOTE that you should set this to 0 if you only want to run inference on a new set of images (i.e. skip training)
ONLY_EVAL = False  # Set this to True if you want to evaluate the 'test' or 'val' set without training
###################################################################################################

# (1.2) Set hyperparameters for inference
###################################################################################################
DATA_FOLDERNAME = 'CovidImages'  # Must be inside 'data' folder
OUTPUT_FOLDERNAME = 'CovidTest_membranes'  # Will be created inside 'output' folder
SPECIFIC_FILENAME = None  # To investigate a single image, give a image filename here
SHOW_IMGS = False  # Set True to flash intermediate outputs (e.g. masks, bounding boxes, etc.) to screen
RESIZE_TO_800 = True  # Set True to cap images by max. 800 pixels high while inputting them to Mask R-CNN
USE_ORIGINAL_RESOLUTION = True  # Set True to test original / high res. output for higher quality input to classification (Jiawei's model)
SAVE = True  # Set True to to save the resultant membranes
OVERWRITE = False  # Set True to overwrite membrane files inside `OUTPUT_FOLDERNAME` if there is a name-match
TEST_ID = 'aconag'  # ID of the test kit, must match an entry in `kit_data.json`
ANGLE_CALCULATION_METHOD = 'membrane_mask' # Choose one from 'membrane_mask' and 'kit_mask'; which mask to base angle calculation on
# NOTE: Check which method we choose for each `TEST_ID` from AWS Lambda Functions and their Environment Variables
ANGLE_THRESHOLD = 20  # Based on the angle and this set threshold, we either use rotation or homography to get the correct orientation
MEMBRANE_LOCALIZATION_THRESHOLD = 0.60  # Set threshold of overlap percentage between predicted membrane mask and expected membrane mask given by manufacturer specs
INSET_REDNESS_THRESHOLD = 4  # Threshold for measuring the redness in the inset / inlet location of the membrane
INLET_LOCALIZATION_VARIABILITY = 0.15  # +/- error or variability when taking the manufacturer specs for inlet into consideration
INLET_CHECK = False # Set True to enable checking inlet / inset and ensuring there is some redness (i.e. user blood)
SAVE_NPY = False  # Set to True to save .npy format membranes alongisde .jpg format membranes
FLIP_TEST = False  # Set to True to run the flipped version of each image through the model
MANUAL_IOU_CALCULATION = False  # Set to True to measure the IOU between predicted membrane mask and ground-truth membrane mask; will fail if ground-truth not available
PREDICTION_KIT_DETECTION_THRESHOLD = 0.85  # Kits that have below this threshold are discared
PREDICTION_MEMBRANE_DETECTION_THRESHOLD = 0.85  # Membranes that have below this threshold are discarded
PREDICTION_KIT_SEGMENTATION_THRESHOLD = 0.85  # Pixels that are below this threhsold on the kit mask are discared
PREDICTION_MEMBRANE_SEGMENTATION_THRESHOLD = 0.85  # Pixels that are below this trehsold on the membrane mask are discarded
#####################################################################################################

# (1.3) Set other hyperparameters (e.g. might be used for both training and evaluation etc.)
#####################################################################################################
CONTINUE = True
SAVE_PATH = os.path.join('saved_models', 'aconag_weights.pth')  # Change 'TEMP.pth' to the desired model name (e.g. btnx_maskrcnn_weights.pth); only relevant for training
LOAD_PATH = os.path.join('saved_models', 'aconag_weights.pth')  # Change 'TEMP.pth' to desired model path (e.g. oraquick_maskrcnn_weights.pth); relevant for both training and inference
# NOTE: `LOAD_PATH` can be used for finetuning on top of a pretrained model if `CONTINUE` global variable is set to True
# LOAD_PATH = os.path.join('saved_models', 'oraquick_maskrcnn_weights.pth')  # This is a common choice for a pretrained model
# LOAD_PATH = os.path.join('saved_models', 'btnx_maskrcnn_weights.pth')  # This is another common choice for a pretrained model
#####################################################################################################

# (2) Initialize and merge datasets
# Specify datasets for the focus group/test kit
aconag_train_dataset = get_lfa_dataset(name='aconag', split='train', train=True, resize=RESIZE, shots=SHOTS, seed=SEED)
# The training datasets loads inputs / images from `aconag_train_images` and labels / masks from `aconag_train_masks` with this line.
aconag_test_dataset = get_lfa_dataset(name='aconag', split='test', train=False, resize=RESIZE, seed=SEED)
# The test datasets loads inputs / images from `aconag_test_images` and labels / masks from `aconag_test_masks` with this line.

train_dataset = ConcatDataset([
    aconag_train_dataset
])
test_dataset = ConcatDataset([
    aconag_test_dataset
])
# NOTE: The concatenate various datasets for whatever reason, add items to the list within `ConcatDataset()`

# (3) Initialize data loaders
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE if NUM_EPOCHS != 0 else 1,
                          shuffle=True,
                          num_workers=0,  # TODO: Figure out why num_workers > 0 sometimes fails?
                          collate_fn=utils.collate_fn,
                          pin_memory=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=1,
                         shuffle=False,
                         num_workers=0,
                         collate_fn=utils.collate_fn,
                         pin_memory=True)

# (4) Get the pretrained instance segmentation model
model = get_instance_segmentation_model(num_classes=NUM_CLASSES, hidden_size=HIDDEN_SIZE).to(DEVICE)

# (5) Load weights from previous training if applicable
if CONTINUE and os.path.exists(LOAD_PATH):
    model.load_state_dict(torch.load(LOAD_PATH, map_location=DEVICE))
    model.train()
    print('LOADED MODEL WEIGHTS FROM %s' % LOAD_PATH)
else:
    print('STARTING MODEL TRAINING FROM SCRATCH')

# (5) Initialize optimizer and learning rate scheduler
parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
# optimizer = SGD(params=parameters, lr=0.00005, momentum=0.9, weight_decay=0.0005)
# scheduler = StepLR(optimizer=optimizer, step_size=3, gamma=0.1)
optimizer = Adam(params=parameters, lr=LEARNING_RATE)

### (6) TRAINING ###
if NUM_EPOCHS != 0:
    print('TRAINING!')
    # Repeat the experiment multiple times to get confident results!
    test_aps, best_test_aps = [], []
    for _ in tqdm(range(NUM_REPEATS), desc='Repeating Experiment for Confidence!'):
        best_test_ap = float('-inf')
        for epoch in range(NUM_EPOCHS):
            # SAVE_PATH = os.path.join('saved_models', 'iou_demo', '%d.pth' % epoch)
            if not ONLY_EVAL:
                # Train for one epoch & print every x iterations s.t. it prints 5 times per epoch
                model.train()
                train_one_epoch(model, optimizer, train_loader, DEVICE, epoch, print_freq=len(train_dataset) // 5 if len(train_dataset) >= 5 else 1)

            # Update the learning rate if scheduler enabled
            # scheduler.step()

            # Evaluate on the test dataset (always compute on CPU for memory saving!)
            model.eval()
            # train_metrics = evaluate(model, train_loader, device=DEVICE) # device='cpu')
            test_metrics = evaluate(model, test_loader, device=DEVICE) # device='cpu')
            # Get "Average Precision (AP) @[IoU=0.50:0.95|area=all|maxDets=100]" as test performance metric!
            test_ap = test_metrics.coco_eval['segm'].stats[0] 
            if test_ap > best_test_ap:
                print('BEST MODEL ON TEST REACHED AT EPOCH=%d' % (epoch + 1))
                best_test_ap = test_ap
            
            # NOTE: Here we are choosing to save / overwite the model after every epoch
            # TODO: A better strategy would be to add an in-between validation set,
            #       only save the model that does best on validation set, and then
            #       run inference separately on the test set to report final performance
            torch.save(model.state_dict(), SAVE_PATH)

        test_aps.append(test_ap)
        best_test_aps.append(best_test_ap)

    train_metrics = evaluate(model, train_loader, device=DEVICE)
    test_metrics = evaluate(model, test_loader, device=DEVICE)
    print('Best Test Performance (AP) @[IoU=0.50:0.95|area=all|maxDets=100]: %0.4f' % np.mean(best_test_aps))
    print('Final Test Performance (AP) @[IoU=0.50:0.95|area=all|maxDets=100]: %0.4f' % np.mean(test_aps))
    exit(0)
    # NOTE: We exit the script here such that when NUM_EPOCHS != 0, inference is not run on new images and we only do training & evaluation
else:
    print('SKIPPING TRAINING!')

print('---------------------FINISHED---------------------')
model = model.eval()


### (7) Helper Functions for Object Detection Inference ###
def rotate_image(image, angle):
    """Function to rotate image in a specified (integer) angle"""
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def compute_bbox_for_rotated_rect_with_max_area(h, w, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible axis-aligned rectangle 
    (maximal area) within the rotated rectangle and return bbox coordinates. Source:
    https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    :param (int) h: height of the image
    :param (int) w: width of the image
    :param (float) angle: the angle of rotation in radians
    :return: (tuple) xmin, xmax, ymin, ymax -> bbox coords. of the rotated image
    """
    if w <= 0 or h <= 0:
        return 0,0
    
    width_is_longer = w >= h
    side_long, side_short = (w,h) if width_is_longer else (h,w)
    
    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))

    ## Commented this out because all cases I see are with 4 sides...
    # if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
    if False:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
    
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a
        
    w_bb = w*cos_a + h*sin_a
    h_bb = w*sin_a + h*cos_a
    inset_horizontal = (w_bb - wr) / 2
    inset_vertical = (h_bb - hr) / 2
    
    # Calculate coordinates and return them instead
    ymax = int(hr + inset_vertical/2) - 1
    xmax = int(wr + inset_horizontal/2) - 1
    ymin = round(round(inset_vertical)/2) + 1
    xmin = round(round(inset_horizontal)/2) + 1
    
    return xmin, xmax, ymin, ymax 


def apply_homography(src, homography_matrix):
    """
    Function to apply homography H onto the source coordinate.
    :param (np.ndarray) src: source coordinate as (x, y)
    :param (np.ndarray) homography_matrix: denotes homography from (x, y) to (x', y') with shape (3, 3)
    :return: dst, destination coordinate as (x', y')
    """
    x, y = src
    # Convert source coordinate space (i.e. cartesian) to homogenous coordinate space
    homogenous_src = np.asarray([[x], [y], [1]])
    # Compute destination coordinate in homogenous coordinate space
    homogenous_dst = np.dot(homography_matrix, homogenous_src)

    # We now have [[w*x'], [w*y'], [w]] as homogenous coordinate
    # We have to divide first two terms by w for cartesian coordinates of destination
    w = homogenous_dst[2][0]
    x_prime, y_prime = homogenous_dst[0][0] / w, homogenous_dst[1][0] / w

    # Convert float coordinates to integer
    dst = (int(round(x_prime)), int(round(y_prime)))
    return dst

def compute_bbox(mask):
    """
    Function to compute the bounding box (i.e. rectangular) coordinates of a mask.
    :param (np.ndarray) mask: image mask with shape (H, W, C) and with values [0, 255]
    :return: bounding box (a.k.a bbox) coordinates -> (xmin, xmax, ymin, ymax)
    """
    height, width, num_channels = mask.shape

    ## (1) Get horizontal values ##
    left_edges = np.where(mask.any(axis=1), mask.argmax(axis=1), width + 1)
    # Flip horizontally to get right edges
    flip_lr = cv2.flip(mask, flipCode=1) 
    right_edges = width - np.where(flip_lr.any(axis=1), flip_lr.argmax(axis=1), width + 1)

    ## (2) Get vertical values ##
    top_edges = np.where(mask.any(axis=0), mask.argmax(axis=0), height + 1)
    # Flip vertically to get bottom edges
    flip_ud = cv2.flip(mask, flipCode=0) 
    bottom_edges = height - np.where(flip_ud.any(axis=0), flip_ud.argmax(axis=0), height + 1)

    # Find the minimum and maximum values -> bbox coordinates
    xmin, xmax, ymin, ymax = left_edges.min(), right_edges.max(), top_edges.min(), bottom_edges.max()
    return xmin, xmax, ymin, ymax

def compute_bquad(mask, return_box=False):
    """
    Function to compute the bounding quadrilateral (i.e. kite, parallelogram, trapezoid)
    coordinates of a mask
    :param (np.ndarray) mask: image mask with shape (H, W, C) and with values [0, 255]
    :return: bounding quadrilateral coordinates -> ((y1, x1), (y2, x2), (y3, x3), (y4, x4))
    """
    def order_points(points):
        """Function to return box-points in [lefttop, righttop, rightbottom, leftbottom] order"""
        # Convert from (x,y) representation to (y,x)
        points = np.array([point[::-1] for point in points])

        # Sort the points based on their y-coordinates
        points_ysorted = points[np.argsort(points[:, 0]), :]

        # Grab the bottommost and topmost points from the sorted y-coordinate points
        topmost, bottommost = points_ysorted[:2, :], points_ysorted[2:, :]

        # Sort the topmost coordinates according to their x-coordinates
        lefttop, righttop = topmost[np.argsort(topmost[:, 1]), :]

        # Apply the Pythagorean theorem
        distances = np.sum((righttop[np.newaxis] - bottommost)**2, axis=1)
        leftbottom, rightbottom = bottommost[np.argsort(distances)[::-1], :]

        return np.array([lefttop, righttop, rightbottom, leftbottom], dtype='int')

    # Valid mask coordinates are those that are not 0 and are 255
    mask_coords = np.argwhere(mask[:, :, 0] == 255)
    # Get the center (e.g (y/2, x/2)) of the mask
    center = np.mean(mask_coords, axis=0).astype('int')

    # Find contours and approximate a quadrilateral using the mask
    contours, _ = cv2.findContours(mask.copy()[:, :, 0].astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype('int')

    lefttop_coord, righttop_coord, rightbottom_coord, leftbottom_coord = order_points(box)

    if return_box:
        return lefttop_coord, righttop_coord, rightbottom_coord, leftbottom_coord, box
    else:
        return lefttop_coord, righttop_coord, rightbottom_coord, leftbottom_coord


### (8) PREDICT & VISUALIZE ###
if not os.path.exists(os.path.join('output', OUTPUT_FOLDERNAME)):
    os.makedirs(os.path.join('output', OUTPUT_FOLDERNAME))

with open('kit_data.json') as f:
    ALL_INFO = json.load(f)

    KIT_INFO = ALL_INFO[TEST_ID]
    if INLET_CHECK:
        INLET_INFO = KIT_INFO['sample_inlet']


# Iterate over images in the specified path and run inference on them!
for filename in tqdm(os.listdir(os.path.join('data', DATA_FOLDERNAME)), desc='Getting Predictions'):
    try:
        if SPECIFIC_FILENAME is not None:
            if SPECIFIC_FILENAME not in filename:
                continue

        if not OVERWRITE and os.path.exists(os.path.join('output', OUTPUT_FOLDERNAME, filename)):
            continue

        print(filename)
        test_image = cv2.imread(os.path.join('data', DATA_FOLDERNAME, filename))
        original_resolution_image = test_image.copy()
        if RESIZE_TO_800:
            new_height = 800
            new_width = int((new_height / test_image.shape[0]) * test_image.shape[1])
            test_image = cv2.resize(test_image, (new_width, new_height))
            if FLIP_TEST:
                test_image = cv2.rotate(test_image, cv2.ROTATE_180)

        from torchvision.transforms import functional as F
        test_image = F.to_tensor(test_image)
        C, H, W = test_image.shape

        with torch.no_grad():
            # Pass through model (i.e. inference) and get predictions
            predictions = model([test_image.to(DEVICE)])[0]

            # Get labels (i.e. 1 -> kit, 2 -> membrane) and scores, which are aligned
            labels = predictions['labels'].cpu().numpy().tolist()
            scores = predictions['scores'].cpu().numpy().tolist()
            assert len(labels) == len(scores)

            # Get boxes and masks, which are also naturally aligned
            boxes = predictions['boxes'].cpu().numpy()
            masks = predictions['masks'].cpu().numpy()
            assert boxes.shape[0] == masks.shape[0] == len(labels)

            # Get the maximum confidence kit and membrane location (i.e. first occurrence in list)
            try:
                kit_loc, membrane_loc = labels.index(btnx_train_dataset.kit_id), labels.index(btnx_train_dataset.membrane_id)
                kit_score, membrane_score = scores[kit_loc], scores[membrane_loc]
            except:
                # raise ValueError('Either kit or membrane is missing from the prediction or the image!')
                continue

            # Make sure prediction confidence scores are above the set threshold
            if kit_score < PREDICTION_KIT_DETECTION_THRESHOLD:
                pass
                # raise ValueError('Kit confidence score is smaller than threshold %s' % str(threshold))
                # continue
            if membrane_score < PREDICTION_MEMBRANE_DETECTION_THRESHOLD:
                pass
                # raise ValueError('Membrane confidence score is smaller than threshold %s' % str(threshold))
                # continue

            # Get best kit and membrane boxes
            kit_box, membrane_box = boxes[kit_loc, :], boxes[membrane_loc, :]
            kit_mask, membrane_mask = masks[kit_loc, :, :, :], masks[membrane_loc, :, :, :]
        
            # Segment the actual masks (i.e. kit and membrane) via the set threshold and convert to int dtype
            kit_mask[kit_mask >= PREDICTION_KIT_SEGMENTATION_THRESHOLD] = 1.
            kit_mask[kit_mask < PREDICTION_KIT_SEGMENTATION_THRESHOLD] = 0.
            kit_mask = np.array(np.concatenate([kit_mask.reshape((H, W, 1)) * 255] * 3, axis=-1), dtype=np.uint8)

            membrane_mask[membrane_mask >= PREDICTION_MEMBRANE_SEGMENTATION_THRESHOLD] = 1.
            membrane_mask[membrane_mask < PREDICTION_MEMBRANE_SEGMENTATION_THRESHOLD] = 0.
            membrane_mask = np.array(np.concatenate([membrane_mask.reshape((H, W, 1)) * 255] * 3, axis=-1), dtype=np.uint8)

            # Compute IOU scores manually (with help from scikit-learn's jaccard_score()) if ground-truth masks exist
            if MANUAL_IOU_CALCULATION:
                if os.path.exists(os.path.join('data', DATA_FOLDERNAME.replace('images', 'masks'))):
                    gt_mask = cv2.imread(os.path.join('data', DATA_FOLDERNAME.replace('images', 'masks'), filename.replace('.jpg', '.png')))

                    if RESIZE_TO_800:
                        new_height = 800
                        new_width = int((new_height / gt_mask.shape[0]) * gt_mask.shape[1])
                        gt_mask = cv2.resize(gt_mask, (new_width, new_height))

                    # Construct a membrane only GT mask
                    gt_membrane_mask = gt_mask.copy()
                    gt_membrane_mask[np.any(gt_membrane_mask != (0, 255, 0), axis=-1)] = (0, 0, 0)
                    gt_membrane_mask[np.all(gt_membrane_mask == (0, 255, 0), axis=-1)] = (255, 255, 255)

                    # Binarize before score calculation
                    gt_membrane_mask_binary, membrane_mask_binary = gt_membrane_mask.copy(), membrane_mask.copy()
                    gt_membrane_mask_binary[gt_membrane_mask_binary == 255] = 1
                    membrane_mask_binary[membrane_mask_binary == 255] = 1

                    membrane_iou_score = jaccard_score(y_true=gt_membrane_mask_binary[:, :, 0].flatten(), y_pred=membrane_mask_binary[:, :, 0].flatten())
                    membrane_iou_score = str(round(membrane_iou_score, 4))
                    print('Membrane IOU Score: ', membrane_iou_score)

                    if SHOW_IMGS:
                        test_image_np =  np.array(test_image.cpu().numpy().transpose(1, 2, 0) * 255, dtype=np.uint8)
                        viz = cv2.addWeighted(src1=test_image_np, alpha=0.50, src2=membrane_mask, beta=0.50, gamma=0)
                        cv2.imshow('IOU Performance Visualization', viz)
                        cv2.waitKey(0)
            
            # Re-read the image and potentially resize since the image was converted to a tensor previously
            test_image = cv2.imread(os.path.join('data', DATA_FOLDERNAME, filename))
            if FLIP_TEST:
                test_image = cv2.rotate(test_image, cv2.ROTATE_180)
            if RESIZE_TO_800:
                new_height = 800
                new_width = int((new_height / test_image.shape[0]) * test_image.shape[1])
                test_image = cv2.resize(test_image, (new_width, new_height))

            empty_initial_image = test_image.copy()
            empty_initial_image[:, :, :] = 0

            # Calculate angle based on kit or membrane coordinates
            if ANGLE_CALCULATION_METHOD == 'kit_mask':
                lefttop_coord, righttop_coord, rightbottom_coord, leftbottom_coord, box = compute_bquad(kit_mask, return_box=True)
            elif ANGLE_CALCULATION_METHOD == 'membrane_mask':
                lefttop_coord, righttop_coord, rightbottom_coord, leftbottom_coord, box = compute_bquad(membrane_mask, return_box=True)

            left_angle = math.atan((leftbottom_coord[1] - lefttop_coord[1]) / 
                                   (leftbottom_coord[0] - lefttop_coord[0])) * (180 / math.pi)
            right_angle = math.atan((rightbottom_coord[1] - righttop_coord[1]) /
                                    (rightbottom_coord[0] - righttop_coord[0])) * (180 / math.pi)
            angle = int(round((left_angle + right_angle) / 2))
            print('CONSENSUS ANGLE: ', angle)
            if abs(angle) >= ANGLE_THRESHOLD:
                print('Image is >= %d degrees rotated!' % ANGLE_THRESHOLD)  ## raise Error

            # Compute bounding box and bounding quadrilateral coordinates for kit and membrane masks
            kit_xmin, kit_xmax, kit_ymin, kit_ymax = compute_bbox(kit_mask)
            if (kit_xmax - kit_xmin) > (kit_ymax - kit_ymin):
                print('Image is probably 90 degrees or > 45 degrees rotated!')  ## raise Error
            kit_lefttop_coord, kit_righttop_coord, kit_rightbottom_coord, kit_leftbottom_coord = compute_bquad(kit_mask)

            membrane_xmin, membrane_xmax, membrane_ymin, membrane_ymax = compute_bbox(membrane_mask)
            membrane_lefttop_coord, membrane_righttop_coord, membrane_rightbottom_coord, membrane_leftbottom_coord = compute_bquad(membrane_mask)

            if SHOW_IMGS:
                viz = test_image.copy()
                viz = cv2.circle(viz, (kit_xmin, kit_ymin), 3, (0,255,0), -1)
                viz = cv2.circle(viz, (kit_xmax, kit_ymax), 3, (0,255,0), -1)
                viz = cv2.circle(viz, (kit_xmax, kit_ymin), 3, (0,255,0), -1)
                viz = cv2.circle(viz, (kit_xmin, kit_ymax), 3, (0,255,0), -1)
                viz = cv2.circle(viz, tuple(kit_lefttop_coord[::-1]), 6, (0,255,0), -1)
                viz = cv2.circle(viz, tuple(kit_righttop_coord[::-1]), 6, (0,255,0), -1)
                viz = cv2.circle(viz, tuple(kit_leftbottom_coord[::-1]), 6, (0,255,0), -1)
                viz = cv2.circle(viz, tuple(kit_rightbottom_coord[::-1]), 6, (0,255,0), -1)

                viz = cv2.circle(viz, (membrane_xmin, membrane_ymin), 3, (255,0,0), -1)
                viz = cv2.circle(viz, (membrane_xmax, membrane_ymax), 3, (255,0,0), -1)
                viz = cv2.circle(viz, (membrane_xmax, membrane_ymin), 3, (255,0,0), -1)
                viz = cv2.circle(viz, (membrane_xmin, membrane_ymax), 3, (255,0,0), -1)
                viz = cv2.circle(viz, tuple(membrane_lefttop_coord[::-1]), 6, (255,0,0), -1)
                viz = cv2.circle(viz, tuple(membrane_righttop_coord[::-1]), 6, (255,0,0), -1)
                viz = cv2.circle(viz, tuple(membrane_leftbottom_coord[::-1]), 6, (255,0,0), -1)
                viz = cv2.circle(viz, tuple(membrane_rightbottom_coord[::-1]), 6, (255,0,0), -1)

                cv2.imshow('Test Image and Coordinates for Kit (Green) and Membrane (Blue)', viz)
                cv2.waitKey(0)
            
            # Crop the image s.t. that it only contains the kit (and hence the membrane)
            test_image = test_image[kit_ymin: kit_ymax, kit_xmin: kit_xmax]
            # cv2.imwrite('quidel_demo/6_kit_cropped_image.jpg', test_image)

            # Update coordinates (i.e. subtract (kit_ymin, kit_xmin) from each)
            kit_lefttop_coord, membrane_lefttop_coord = kit_lefttop_coord - (kit_ymin, kit_xmin), membrane_lefttop_coord - (kit_ymin, kit_xmin)
            kit_righttop_coord, membrane_righttop_coord = kit_righttop_coord - (kit_ymin, kit_xmin), membrane_righttop_coord - (kit_ymin, kit_xmin)
            kit_rightbottom_coord, membrane_rightbottom_coord = kit_rightbottom_coord - (kit_ymin, kit_xmin), membrane_rightbottom_coord - (kit_ymin, kit_xmin)
            kit_leftbottom_coord, membrane_leftbottom_coord = kit_leftbottom_coord - (kit_ymin, kit_xmin), membrane_leftbottom_coord - (kit_ymin, kit_xmin)

            # Update xmin, xmax, ymin, and ymax for kit and membrane (i.e. subtract kit_ymin OR kit_xmin from each)
            kit_xmin, kit_xmax, kit_ymin, kit_ymax = 0, test_image.shape[1] - 1, 0, test_image.shape[0] - 1
            membrane_xmin, membrane_xmax, membrane_ymin, membrane_ymax = membrane_xmin - kit_xmin, membrane_xmax - kit_xmin, membrane_ymin - kit_ymin, membrane_ymax - kit_ymin

            if SHOW_IMGS:
                viz = test_image.copy()
                viz = cv2.circle(viz, tuple(kit_lefttop_coord[::-1]), 3, (0,255,0), -1)
                viz = cv2.circle(viz, tuple(kit_righttop_coord[::-1]), 3, (0,255,0), -1)
                viz = cv2.circle(viz, tuple(kit_rightbottom_coord[::-1]), 3, (0,255,0), -1)
                viz = cv2.circle(viz, tuple(kit_leftbottom_coord[::-1]), 3, (0,255,0), -1)

                viz = cv2.circle(viz, tuple(membrane_lefttop_coord[::-1]), 3, (255,0,0), -1)
                viz = cv2.circle(viz, tuple(membrane_righttop_coord[::-1]), 3, (255,0,0), -1)
                viz = cv2.circle(viz, tuple(membrane_rightbottom_coord[::-1]), 3, (255,0,0), -1)
                viz = cv2.circle(viz, tuple(membrane_leftbottom_coord[::-1]), 3, (255,0,0), -1)
                
                cv2.imshow('Cropped Image Kit (Green) and Membrane (Blue) Coordinates', viz)
                cv2.waitKey(0)

            ## Compute homography using kit coordinates ##
            # NOTE: Due to conventions of cv2's homography utilites, we will switch (y, x) around to be (x, y) in this section...
            # Take the source coordinates as the previously calculated ones
            kit_src_coords = np.array([kit_lefttop_coord[::-1], kit_righttop_coord[::-1], kit_rightbottom_coord[::-1], kit_leftbottom_coord[::-1]])

            # Get the ratio of the width to the height (i.e. aspect ratio)
            aspect_ratio = KIT_INFO['dimensions']['aspect_ratio']

            new_height, new_width = test_image.shape[0], int(test_image.shape[0] * aspect_ratio)
            # Take the destination coordinates as the far edges of the newly defined shape
            kit_dst_coords = np.array([[0, 0][::-1], [0, new_width-1][::-1], [new_height-1, new_width-1][::-1], [new_height-1, 0][::-1]])

            # Calculate homography
            homography_matrix, _ = cv2.findHomography(kit_src_coords, kit_dst_coords)

            # Warp source image to destination based on homography
            warped_test_image = cv2.warpPerspective(test_image, homography_matrix, (new_width, new_height))

            # Apply the homography matrix to get the new membrane bquad coordinates
            membrane_lefttop_coord = apply_homography(src=membrane_lefttop_coord[::-1], homography_matrix=homography_matrix)[::-1]
            membrane_righttop_coord = apply_homography(src=membrane_righttop_coord[::-1], homography_matrix=homography_matrix)[::-1]
            membrane_rightbottom_coord = apply_homography(src=membrane_rightbottom_coord[::-1], homography_matrix=homography_matrix)[::-1]
            membrane_leftbottom_coord = apply_homography(src=membrane_leftbottom_coord[::-1], homography_matrix=homography_matrix)[::-1]

            # Get new kit and membrane bbox coordinates based on the warped test image and above bquad coordinates
            kit_xmin, kit_xmax, kit_ymin, kit_ymax = 0, warped_test_image.shape[1] - 1, 0, warped_test_image.shape[0] - 1
            membrane_xmin = min(membrane_lefttop_coord[1], membrane_leftbottom_coord[1])
            membrane_xmax = max(membrane_righttop_coord[1], membrane_rightbottom_coord[1])
            membrane_ymin = min(membrane_lefttop_coord[0], membrane_righttop_coord[0])
            membrane_ymax = max(membrane_leftbottom_coord[0], membrane_rightbottom_coord[0])

            # Make sure that all coordinates are >= 0
            membrane_xmin = max(0, membrane_xmin)
            membrane_xmax = max(0, membrane_xmax)
            membrane_ymin = max(0, membrane_ymin)
            membrane_ymax = max(0, membrane_ymax)

            # Enforce location constraint on membrane
            info = KIT_INFO['dimensions']['membrane']

            expected_membrane_xmin = int(kit_xmin + (kit_xmax * info['x']))
            expected_membrane_xmax = int(kit_xmin + (kit_xmax * (info['x'] + info['w'])))
            expected_membrane_ymin = int(kit_ymin + (kit_ymax * info['y']))
            expected_membrane_ymax = int(kit_ymin + (kit_ymax * (info['y'] + info['h'])))

            expected_membrane_zone = np.zeros(warped_test_image.shape)[:, :, 0]
            expected_membrane_zone[expected_membrane_ymin: expected_membrane_ymax, expected_membrane_xmin: expected_membrane_xmax] = 1
            membrane_zone = np.zeros(warped_test_image.shape)[:, :, 0]
            membrane_zone[membrane_ymin: membrane_ymax, membrane_xmin: membrane_xmax] = 1
            overlap = expected_membrane_zone + membrane_zone
            overlap[np.where(overlap == 1)] = 0

            overlap_percentage = np.sum(overlap) / (np.sum(expected_membrane_zone) + np.sum(membrane_zone))
            
            if SHOW_IMGS:
                viz = warped_test_image.copy()
                viz = cv2.circle(viz, (membrane_xmin, membrane_ymin), 3, (0,255,0), -1)
                viz = cv2.circle(viz, (membrane_xmax, membrane_ymin), 3, (0,255,0), -1)
                viz = cv2.circle(viz, (membrane_xmax, membrane_ymax), 3, (0,255,0), -1)
                viz = cv2.circle(viz, (membrane_xmin, membrane_ymax), 3, (0,255,0), -1)

                viz = cv2.circle(viz, (expected_membrane_xmin, expected_membrane_ymin), 3, (0,0,255), -1)
                viz = cv2.circle(viz, (expected_membrane_xmax, expected_membrane_ymin), 3, (0,0,255), -1)
                viz = cv2.circle(viz, (expected_membrane_xmax, expected_membrane_ymax), 3, (0,0,255), -1)
                viz = cv2.circle(viz, (expected_membrane_xmin, expected_membrane_ymax), 3, (0,0,255), -1)

                cv2.imshow('Warped Image Membrane Coordinates (Green) and Expected (Red)', viz)
                cv2.waitKey(0)

            print('MEMBRANE OVERLAP PERCENTAGE: ', overlap_percentage)
            if overlap_percentage < MEMBRANE_LOCALIZATION_THRESHOLD:
                print('MEMBRANE LOCALIZATION FAILED FOR %s!' % filename)

            # Keep homography calculated membrane just in case if angle is > ANGLE_THRESHOLD
            homography_calculated_membrane = warped_test_image[membrane_ymin: membrane_ymax, membrane_xmin: membrane_xmax, :]

            if INLET_CHECK:
                # Check inset and ensure there is some redness
                found_kit_width, found_kit_height = kit_xmax - kit_xmin, kit_ymax - kit_ymin
                inset_xcenter = found_kit_width - (INLET_INFO['right_to_center'] * found_kit_width)
                inset_ycenter = found_kit_height - (INLET_INFO['bottom_to_center'] * found_kit_height)
                inset_xmin = int(inset_xcenter - (INLET_INFO['diameter/width'] * found_kit_width))
                inset_xmax = int(inset_xcenter + (INLET_INFO['diameter/width'] * found_kit_width))
                inset_ymin = int(inset_ycenter - (INLET_INFO['diameter/height'] * found_kit_height))
                inset_ymax = int(inset_ycenter + (INLET_INFO['diameter/height'] * found_kit_height))
                inset_width, inset_height = inset_xmax - inset_xmin, inset_ymax - inset_ymin
                
                inset_xmin = int(inset_xmin - (inset_width * INLET_LOCALIZATION_VARIABILITY))
                inset_xmax = int(inset_xmax + (inset_width * INLET_LOCALIZATION_VARIABILITY))
                inset_ymin = int(inset_ymin - (inset_height * INLET_LOCALIZATION_VARIABILITY))
                inset_ymax = int(inset_ymax + (inset_height * INLET_LOCALIZATION_VARIABILITY))
                inset = warped_test_image[inset_ymin: inset_ymax, inset_xmin: inset_xmax, :]
                if SHOW_IMGS:
                    viz = inset.copy()
                    cv2.imshow('Inlet', viz)
                    cv2.waitKey(0)

                inset_hsv = cv2.cvtColor(inset, cv2.COLOR_BGR2HSV)

                # NOTE: For both red masks, their V-channel (last channel) is reduced to 15 from 50
                #       in order to capture older kits with black inlets...
                # 1st Red Mask (0-10)
                lower_red, upper_red = np.array([0,70,15]), np.array([10,255,255])
                red_mask1 = cv2.inRange(inset_hsv, lower_red, upper_red)
                # 2nd Red Mask (170-180)
                lower_red, upper_red = np.array([170,70,15]), np.array([180,255,255])
                red_mask2 = cv2.inRange(inset_hsv, lower_red, upper_red)
                # Create a one joint Red Mask
                red_mask = red_mask1 + red_mask2

                # Set image to zero everywhere except the red mask
                inset_red_thresholded = inset_hsv.copy()
                inset_red_thresholded[np.where(red_mask==0)] = 0
                inset_red_thresholded[np.where(red_mask==255)] = 255
                if SHOW_IMGS:
                    viz = inset_red_thresholded.copy()
                    cv2.imshow('Inset Red Thresholded', viz)
                    cv2.waitKey(0)

                inset_redness = int(round(np.mean(red_mask)))
                print('REDNESS ON INSET: ', np.mean(red_mask))
                if inset_redness < INSET_REDNESS_THRESHOLD:
                    print('REDNESS ON INSET CHECK FAILED FOR %s!' % filename)

            # Re-read the image and potentially resize since the image was cropped previously
            test_image = cv2.imread(os.path.join('data', DATA_FOLDERNAME, filename))
            if RESIZE_TO_800:
                new_height = 800
                new_width = int((new_height / test_image.shape[0]) * test_image.shape[1])
                test_image = cv2.resize(test_image, (new_width, new_height))

            # Get the rotated membrane using the angle and the membrane mask from before
            membrane_xmin, membrane_xmax, membrane_ymin, membrane_ymax = compute_bbox(membrane_mask)

            if USE_ORIGINAL_RESOLUTION:
                y_scale = original_resolution_image.shape[0] / membrane_mask.shape[0]
                x_scale = original_resolution_image.shape[1] / membrane_mask.shape[1]

                membrane_xmin_upscaled = int(membrane_xmin * x_scale) + 1
                membrane_xmax_upscaled = int(membrane_xmax * x_scale) - 1
                membrane_ymin_upscaled = int(membrane_ymin * y_scale) + 1
                membrane_ymax_upscaled = int(membrane_ymax * y_scale) - 1

                membrane = rotate_image(original_resolution_image[membrane_ymin_upscaled: membrane_ymax_upscaled, membrane_xmin_upscaled: membrane_xmax_upscaled], -angle)

            else:
                membrane = rotate_image(test_image[membrane_ymin: membrane_ymax, membrane_xmin: membrane_xmax], -angle)

            if abs(angle) < ANGLE_THRESHOLD:
                # Compute the largest are bbox w/o any black zones that occurs due to angle
                membrane_xmin, membrane_xmax, membrane_ymin, membrane_ymax = compute_bbox_for_rotated_rect_with_max_area(membrane.shape[0], 
                                                                                                                         membrane.shape[1], 
                                                                                                                         angle * (math.pi / 180))
                membrane = membrane[membrane_ymin: membrane_ymax, membrane_xmin: membrane_xmax, :]
            else:
                membrane = homography_calculated_membrane

            if SHOW_IMGS:
                cv2.imshow('Final Membrane', membrane)
                cv2.waitKey(0)


            # Convert the test image to NumPy in correct shape
            test_image = F.to_tensor(test_image)
            test_image = test_image.cpu().numpy().transpose(1, 2, 0) * 255
            test_image = np.array(test_image, dtype=np.uint8)

            # Quick check for shapes
            assert kit_mask.shape == membrane_mask.shape == test_image.shape == (H, W, C)

            # Create visualizations as overlayed images for kit and mask 
            kit_mask_visualization = cv2.addWeighted(src1=test_image, alpha=0.20, 
                                                     src2=kit_mask, beta=0.80, 
                                                     gamma=0)
            membrane_mask_visualization = cv2.addWeighted(src1=test_image, alpha=0.20, 
                                                          src2=membrane_mask, beta=0.80, 
                                                          gamma=0)
            
            # Visualize the created images
            if SHOW_IMGS:
                cv2.imshow('Predicted Kit Mask', cv2.resize(kit_mask_visualization, (500, 800)))
                cv2.waitKey(0)

                cv2.imshow('Predicted Membrane Mask', cv2.resize(membrane_mask_visualization, (500, 800)))
                cv2.waitKey(0)

                cv2.imshow('Membrane', cv2.resize(membrane, (200, 600)))
                cv2.waitKey(0)
            
            if SAVE:
                cv2.imwrite(os.path.join('output', OUTPUT_FOLDERNAME, filename), membrane)
                if SAVE_NPY:
                    np.save(os.path.join('output', OUTPUT_FOLDERNAME, os.path.splitext(filename)[0] + '.npy'), membrane)

    except Exception as e:
        print('ERROR OCCURRED WITH %s, INVESTIGATE FURTHER!' % filename)
        print('ERROR MESSAGE: ', e)
        print('TRACEBACK: ', traceback.format_exc())
        continue
