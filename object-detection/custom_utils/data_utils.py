import os
import numpy as np
import torch
import cv2
from scipy.ndimage.measurements import label
import random

import custom_utils.transforms as T


def get_valid_filenames(dir_, masks=False):
    """Function to get all valid image and mask filenames in a directory"""
    all_filenames = [f for f in os.listdir(dir_) if not f.startswith('.')]

    # For masks, we only want PNGs for no lossiness
    if masks:
        valid_filenames = [f for f in all_filenames if f.endswith('.png')]
    else:
        valid_filenames = [f for f in all_filenames if f.endswith('.png') or f.endswith('.jpg')]

    if all_filenames != valid_filenames:
        raise ValueError('(i) bad-formatted images (e.g. JPG for masks) ' + 
                         'OR (ii) non-image files exist ' +
                         'OR (iii) num. masks does NOT match num. images in %s' % dir_)

    return valid_filenames


def get_transform(train):
    """Function to perform data augmentations and transformations"""
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomRotation(prob=0.5, degrees=(160, 200)))
        # NOTE: The above line is added specifically for Quidel as the R&D wanted us to detect flipped images.
        #       Feel free to remove this for other kits; it might be harmful for the overall performance...
        transforms.append(T.RandomHorizontalFlip(prob=0.6))
        transforms.append(T.RandomRotation(prob=0.4, degrees=(90, -90)))
        transforms.append(T.RandomPerspectiveTransform(prob=0.3, scale=(0.01, 0.20)))
        # transforms.append(T.RandomColorJitter(prob=0.4, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3))
        # TODO: Implement ColorJitter

    return T.Compose(transforms=transforms)


def get_lfa_dataset(name, train, split=None, shots=None, resize=False, seed=42):
    """Function to return a LFA dataset given a test name (e.g. 'oraquick')"""
    if split:
        return LFADataset(image_dir=os.path.join('data', '%s_%s_images' % (name, split)),
                          mask_dir=os.path.join('data', '%s_%s_masks' % (name, split)),
                          split=split,
                          transforms=get_transform(train=train),
                          shots=shots,
                          resize=resize,
                          seed=seed)
    else:
        return LFADataset(image_dir=os.path.join('data', '%s_images' % name),
                          mask_dir=os.path.join('data', '%s_masks' % name),
                          split=split,
                          transforms=get_transform(train=train),
                          shots=shots,
                          resize=resize,
                          seed=seed)


class LFADataset(object):
    def __init__(self, image_dir, mask_dir, transforms, split='train', use_prevariables=True, shots=None, resize=False, seed=42):
        assert image_dir.split('_')[0] == mask_dir.split('_')[0]
        assert 'images' in image_dir and 'masks' in mask_dir
        self.name = os.path.split(image_dir)[-1].split('_')[0]

        # Feel free to specify `prevalues_dir` which will contain intermediate object detection files.
        # The code loads these from this specified path when run next time on same set of images.
        # For example, I didn't have enough storage available on my current drive, so I decided to
        # use the D Drive with the following line.
        # self.prevalues_dir = 'D://lfa-prevalues/%s_prevalues' % self.name
        
        self.prevalues_dir = '%s_prevalues' % image_dir.split('_')[0]
        
        if not os.path.exists(self.prevalues_dir):
            print('CREATING NEW PRECOMPUTED SEGMENTATION VARIABLES FOLDER: %s' % self.prevalues_dir)
            os.makedirs(self.prevalues_dir)

        self.transforms = transforms
        self.use_prevariables = use_prevariables

        # Load image and mask filenames and make sure there is %100 correspondence
        image_filenames = list(sorted(get_valid_filenames(dir_=image_dir, masks=False)))
        mask_filenames = list(sorted(get_valid_filenames(dir_=mask_dir, masks=True)))
        # Use only some of the images if 'shots' is not None
        if shots:
            assert isinstance(shots, int)
            random.seed(seed)
            samples = random.sample(list(zip(image_filenames, mask_filenames)), k=shots)
            image_filenames, mask_filenames = [s[0] for s in samples], [s[1] for s in samples]                                         
        assert len(image_filenames) == len(mask_filenames)

        # Load image and mask full filepaths
        self.image_paths = [os.path.join(image_dir, f) for f in image_filenames]
        self.mask_paths = [os.path.join(mask_dir, f) for f in mask_filenames]
        print('Loaded %d %s Images!' % (len(image_filenames), self.name.upper()))

        # Dataset specific attributes -> Give color in (B, G, R) == OpenCV format
        self.background_id, self.background_color = 0, (255, 0, 0)  # blue
        self.kit_id, self.kit_color = 1, (0, 0, 255)  # red
        self.membrane_id, self.membrane_color = 2, (0, 255, 0)  # green

        self.resize = resize

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def extract_category_mask(mask, target_color):
        category_mask = np.zeros(mask.shape)
        category_mask[np.where(np.all(mask == target_color, axis=-1))] = 1
        return category_mask[:, :, 0]
        # NOTE: Returning a 2D version because no need for the 3rd channel

    @staticmethod
    def extract_category_instance_mask(mask, assigned_value):
        category_instance_mask = np.zeros(mask.shape)
        category_instance_mask[np.where(mask == assigned_value)] = 1
        return category_instance_mask

    @staticmethod
    def compute_bounding_box_coordinates(mask):
        all_coordinates = np.where(mask == 1)
        xmin = np.min(all_coordinates[1])
        ymin = np.min(all_coordinates[0])
        xmax = np.max(all_coordinates[1])
        ymax = np.max(all_coordinates[0])
        return xmin, ymin, xmax, ymax

    def compute_segmentation_variables(self, mask):
        # Initialize inputs of the model:
        # (1) masks (N, H, W, 1) -> [0-1] binary masks of the N instances
        # (2) boxes (N, 4) -> coordinates of the N bounding boxes in [x0, y0, x1, y1] format
        # (3) labels (N) -> label for each bounding box (0 represents background)
        masks, boxes, labels = [], [], []

        # Extract category (e.g. 'background', 'kit', 'membrane') masks from the mask
        background_mask = self.extract_category_mask(mask=mask, target_color=self.background_color)
        kit_mask = self.extract_category_mask(mask=mask, target_color=self.kit_color)
        membrane_mask = self.extract_category_mask(mask=mask, target_color=self.membrane_color)

        # NOTE: Since we observed that segmenting kit w/o the membrane is hard, we will also
        # include membrane mask in the kit mask; why deprive the model of an important feature?
        kit_mask += membrane_mask

        # Get connected components for each category except 'background' -> we'll not utilize this
        # Structure with all 1s allows all (i.e. diagonal, horizontal, and vertical) connections
        structure = np.ones((3, 3), dtype=np.int)
        kit_mask, num_kits = label(input=kit_mask, structure=structure)
        membrane_mask, num_membranes = label(input=membrane_mask, structure=structure)
        num_total_instances = num_kits + num_membranes

        # Process kits
        for k in range(1, num_kits+1):
            # (1) Get a single instance of kit and its mask
            kit_instance_mask = self.extract_category_instance_mask(mask=kit_mask, assigned_value=k)
            masks.append(kit_instance_mask)

            # (2) Compute the bounding box coordinates
            xmin, ymin, xmax, ymax = self.compute_bounding_box_coordinates(mask=kit_instance_mask)
            # print(xmin, ymin, xmax, ymax)
            if xmax <= xmin or ymax <= ymin:
                raise ValueError('xmax < xmin OR ymax < ymin for kit -> this should not be the case!')
            boxes.append([xmin, ymin, xmax, ymax])

            # (3) Update labels
            labels.append(self.kit_id)

        # Process membranes
        for k in range(1, num_membranes+1):
            # (1) Get a single instance of kit and its mask
            membrane_instance_mask = self.extract_category_instance_mask(mask=membrane_mask, assigned_value=k)
            masks.append(membrane_instance_mask)

            # (2) Compute the bounding box coordinates
            xmin, ymin, xmax, ymax = self.compute_bounding_box_coordinates(mask=membrane_instance_mask)
            # print(xmin, ymin, xmax, ymax)
            if xmax <= xmin or ymax <= ymin:
                raise ValueError('xmax < xmin OR ymax < ymin for membrane -> this should not be the case!')
            boxes.append([xmin, ymin, xmax, ymax])

            # (3) Update labels
            labels.append(self.membrane_id)

        # Check length of observations
        assert len(masks) == len(boxes) == len(labels) == num_total_instances

        return np.array(masks), np.array(boxes), np.array(labels)

    def __getitem__(self, idx):
        # Get paths and read image and mask as NumPy arrays of equal shape 
        image_path, mask_path = self.image_paths[idx], self.mask_paths[idx]
        image, mask = cv2.imread(image_path), cv2.imread(mask_path)

        if self.resize:
            ratio = image.shape[1] / image.shape[0]
            image, mask = cv2.resize(image, (int(800*ratio), 800)), cv2.resize(mask, (int(800*ratio), 800))

        if image is None or mask is None or image.shape != mask.shape:
            raise ValueError('Error reading image or mask: (%s, %s)' % (image_path, mask_path))

        # Get precomputed segmentation variables, if applicable
        filename = os.path.split(image_path)[-1].replace('.png', '').replace('.jpg', '')
        
        masks_filepath = os.path.join(self.prevalues_dir, '%s_masks.npy' % filename)
        boxes_filepath = os.path.join(self.prevalues_dir, '%s_boxes.npy' % filename)
        labels_filepath = os.path.join(self.prevalues_dir, '%s_labels.npy' % filename)

        masks, boxes, labels = None, None, None
        if os.path.exists(masks_filepath):
            masks = np.load(masks_filepath)
        if os.path.exists(boxes_filepath):
            boxes = np.load(boxes_filepath)
        if os.path.exists(labels_filepath):
            labels = np.load(labels_filepath)
        
        # If not loaded, compute them and save for next time
        if masks is None or boxes is None or labels is None:
            try: 
                masks, boxes, labels = self.compute_segmentation_variables(mask=mask)
            except: 
                raise ValueError('Error computing segmentation on mask: %s' % mask_path)
            np.save(masks_filepath, masks)
            np.save(boxes_filepath, boxes)
            np.save(labels_filepath, labels)
            
        # [DEBUGGING]: for mask in masks: cv2.imshow('mask', cv2.resize(mask, (600, 800))); cv2.waitKey(0);

        # NOTE: The below is for getting 'kit' or 'membrane' masks only!
        # masks, boxes, labels = masks[1:2, ], boxes[1:2, ], labels[1:2, ]
        # masks, boxes, labels = masks[0:1, ], boxes[0:1, ], labels[0:1, ]

        # Convery everything to a torch.Tensor
        num_total_instances = masks.shape[0]
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Get image attributes and specifics
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_total_instances, ), dtype=torch.int64)

        # Prepare target with ground-truths and image information
        target = {}
        target['masks'] = masks
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        ## UNCOMMENT BELOW CODE IF you want to create visualizations as overlayed images for image and mask (i.e. ground-truth) for training and test set
        # viz1 = cv2.addWeighted(src1=image, alpha=0.20, src2=masks[0].unsqueeze(-1).repeat(1, 1, 3).numpy() * 255, beta=0.80, gamma=0)
        # viz2 = cv2.addWeighted(src1=image, alpha=0.20, src2=masks[1].unsqueeze(-1).repeat(1, 1, 3).numpy() * 255, beta=0.80, gamma=0)
        # cv2.imshow('Mask Vizualization 1', viz1)
        # cv2.waitKey(0)
        # cv2.imshow('Mask Vizualization 2', viz2)
        # cv2.waitKey(0)

        # Apply transforms if applicable
        if self.transforms is not None:
            image, target = self.transforms(image, target)



        return image, target
