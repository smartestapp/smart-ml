import os
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="...")
parser.add_argument(
  "--dir",
  help="...",
  type=str,
  required=True
)

args = parser.parse_args()

DIR = args.dir
SHOW = False

def is_img(filename):
  """Checks whether a file is an image file or not."""
  if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPEG') or filename.endswith('.jpeg'):
    return True

  return False


counter = 0

# Copy images to target directory
for path, subdirs, filenames in tqdm(os.walk(DIR), desc='Checking Subdirectories'):
  for img_filename in tqdm(filenames, 'Saving Images w/o EXIF'):

    # Make sure it is actually an image file (.JPEG, .jpg, .png)
    if is_img(img_filename):
        # Get the image path and read image as NumPy array
        img_path = os.path.join(path, img_filename)
        image = cv2.imread(img_path)

        # Check height and width; if orientation is correct the image
        # (i) has no EXIF information, or (ii) has been fixed previously
        # Move onto the next image in these cases.
        H, W, C = image.shape
        if not H > W:
          # Rotate image to correct orientation:
          #         ^
          # <-  TO  |
          image = np.rot90(image, 3)

        # Overwrite anyway
        cv2.imwrite(img_path, image)

        # Optinally show the image
        if SHOW:
            cv2.imshow(img_path, image)
            cv2.waitKey(0)

        counter += 1

print('Fixed %d images!' % counter)