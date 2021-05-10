import random
import cv2
import numpy as np
import torch
import imgaug.augmenters as iaa

from torchvision.transforms import functional as F
import torchvision.transforms as T


def compute_bounding_box_coordinates(mask):
    all_coordinates = np.where(mask == 1)
    try: 
        xmin = np.min(all_coordinates[1])
        ymin = np.min(all_coordinates[0])
        xmax = np.max(all_coordinates[1])
        ymax = np.max(all_coordinates[0])
    except:
        # Means that the transformed image is out-of-bounds for the object!
        xmin, ymin, xmax, ymax = None, None, None, None

    return xmin, ymin, xmax, ymax

def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target

class RandomRotation(object):
    def __init__(self, prob, degrees):
        self.prob = prob
        assert len(degrees) == 2
        self.degrees = degrees

    def __call__(self, image, target):
        if random.random() < self.prob:
            # Sample angle using the min and max specified degrees
            angle = random.randint(min(self.degrees), max(self.degrees))
            # Get boxes and masks and make sure we have the same number for each of them
            boxes, masks = target['boxes'], target['masks']
            assert boxes.shape[0] == masks.shape[0]
            # NOTE: Each of [0] dimension represents an individual object
            
            # Initialize new lists for rotated bboxes and masks
            boxes_, masks_ = [], []

            for mask in masks:
                # Rotate mask and append to ret
                mask_ = rotate_image(image=mask.numpy(), angle=angle)
                masks_.append(mask_)

                # Calculate new bbox coordinates and append to ret, or cancel op. if out-of-bounds
                xmin, ymin, xmax, ymax = compute_bounding_box_coordinates(mask=np.expand_dims(mask_, axis=-1))
                if any(c is None for c in (xmin, ymin, xmax, ymax)):
                    return image, target

                boxes_.append([xmin, ymin, xmax, ymax])
            
            # Convert to tensors
            masks_ = torch.as_tensor(np.array(masks_), dtype=torch.uint8)
            boxes_ = torch.as_tensor(np.array(boxes_), dtype=torch.float32)

            # Assign back to target
            target['boxes'], target['masks'] = boxes_, masks_

            # Rotate the image itself 
            # NOTE: permute() to get it to conventional (H, W, C) form
            image = rotate_image(image=image.permute(1, 2, 0).numpy(), angle=angle)
            image = F.to_tensor(image)

        return image, target


class RandomPerspectiveTransform(object):
    def __init__(self, prob, scale):
        self.prob = prob
        assert len(scale) == 2
        self.scale = scale

    def __call__(self, image, target):
        if random.random() < self.prob:
            self.aug = iaa.PerspectiveTransform(scale=self.scale, keep_size=True)
            # Get boxes and masks and make sure we have the same number for each of them
            boxes, masks = target['boxes'], target['masks']
            assert boxes.shape[0] == masks.shape[0]
            # NOTE: Each of [0] dimension represents an individual object
            
            # Initialize new lists for rotated bboxes and masks
            boxes_, masks_ = [], []

            for mask in masks:
                # Rotate mask and append to ret
                mask_ = self.aug(image=mask.numpy())
                masks_.append(mask_)

                # Calculate new bbox coordinates and append to ret, or cancel op. if out-of-bounds
                xmin, ymin, xmax, ymax = compute_bounding_box_coordinates(mask=np.expand_dims(mask_, axis=-1))
                if any(c is None for c in (xmin, ymin, xmax, ymax)):
                    return image, target
                boxes_.append([xmin, ymin, xmax, ymax])
            
            # Convert to tensors
            masks_ = torch.as_tensor(np.array(masks_), dtype=torch.uint8)
            boxes_ = torch.as_tensor(np.array(boxes_), dtype=torch.float32)

            # Assign back to target
            target['boxes'], target['masks'] = boxes_, masks_

            # Rotate the image itself 
            # NOTE: permute() to get it to conventional (H, W, C) form
            image = self.aug(image=image.permute(1, 2, 0).numpy())
            image = F.to_tensor(image)

        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
