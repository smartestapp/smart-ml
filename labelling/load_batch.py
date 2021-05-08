import os
from tqdm import tqdm
import shutil

import argparse

parser = argparse.ArgumentParser(description="...")
parser.add_argument(
  "--source",
  help="...",
  type=str,
  required=True
)

parser.add_argument(
  "--target", 
  help="...", 
  type=str, 
  required=True
)

args = parser.parse_args()

SOURCE_DIR = args.source
TARGET_DIR = args.target

# Remove the old batch and create a new one
if not os.path.exists(TARGET_DIR):
  os.mkdir(TARGET_DIR)
else:
  shutil.rmtree(TARGET_DIR)
  os.mkdir(TARGET_DIR)


def is_img(filename):
  """Checks whether a file is an image file or not."""
  if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPEG'):
    return True

  return False


counter = 0

# Copy images to target directory
for path, subdirs, filenames in tqdm(os.walk(SOURCE_DIR), desc='Checking Subdirectories'):
  for img_filename in tqdm(filenames, 'Moving Images'):
    if is_img(img_filename):
      img_path = os.path.join(path, img_filename)

      json_filename = img_filename.rsplit('.', 1)[0] + '.json'
      json_path = os.path.join(path, json_filename)

      if os.path.exists(json_path):
        shutil.copy(img_path, os.path.join(TARGET_DIR, img_filename))
        shutil.copy(json_path, os.path.join(TARGET_DIR, json_filename))
        counter += 1


print('Got %d labeled images!' % counter)


