import os
from tqdm import tqdm
import argparse

import numpy as np
import cv2
from PIL import Image

from pycocotools.coco import COCO

# Give color values in BGR order
KIT_ID, KIT_COLOR = 0, (0, 0, 255) # RED
MEMBRANE_ID, MEMBRANE_COLOR = 1, (0, 255, 0) # GREEN
BACKGROUND_COLOR = (255, 0, 0) # BLUE

parser = argparse.ArgumentParser(description="...")
parser.add_argument(
  "--images_dir",
  help="...",
  type=str,
  required=True
)

parser.add_argument(
  "--json_path", 
  help="...", 
  type=str, 
  required=True
)

parser.add_argument(
  "--output_dir", 
  help="...", 
  type=str, 
  required=True
)

args = parser.parse_args()
IMAGES_DIR = args.images_dir
JSON_PATH = args.json_path
OUTPUT_DIR = args.output_dir

if not os.path.exists(OUTPUT_DIR):
  os.makedirs(OUTPUT_DIR)

coco = COCO(JSON_PATH)

category_ids = coco.getCatIds()
assert KIT_ID in category_ids and MEMBRANE_ID in category_ids
assert len(category_ids) == 2

image_ids = coco.getImgIds()
images = coco.loadImgs(image_ids)


for image in tqdm(images, desc='Creating Masks'):
	filename = image['file_name']

	annotation_ids_kits = coco.getAnnIds(imgIds=image['id'], catIds=KIT_ID, iscrowd=None)
	annotation_ids_membranes = coco.getAnnIds(imgIds=image['id'], catIds=MEMBRANE_ID, iscrowd=None)

	annotations_kits = coco.loadAnns(annotation_ids_kits)
	annotations_membranes = coco.loadAnns(annotation_ids_membranes)

	mask = np.zeros((image['height'], image['width'], 3), dtype=np.uint8) + BACKGROUND_COLOR

	for annotation in annotations_kits:
		mask_ = coco.annToMask(annotation)

		for y, x in zip(*np.where(mask_ == 1)):
			mask[y, x, :] = KIT_COLOR

	for annotation in annotations_membranes:
		mask_ = coco.annToMask(annotation)


		for y, x in zip(*np.where(mask_ == 1)):
			mask[y, x, :] = MEMBRANE_COLOR

	# NOTE: Important to save as PNG for no-loss! Otherwise, 255 -> 254
	cv2.imwrite(os.path.join(OUTPUT_DIR, filename.replace('.jpg', '.png')), mask)