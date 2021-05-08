import os
from PIL import Image
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="...")
parser.add_argument("--dir", help="...", type=str, required=True)
args = parser.parse_args()


def get_bad_filenames(filenames):
    return [f for f in filenames 
            if f.endswith('.jpg') 
            or f.endswith('.JPG') 
            or f.endswith('.JPEG')
            or f.endswith('.jpeg')]


for path, subdirs, filenames in tqdm(os.walk(args.dir), desc='Exploring Folders'):
    # Get image files that need fixing
    bad_filenames = get_bad_filenames(filenames)

    for bad_filename in tqdm(bad_filenames, 'Converting JPGs to PNGs'):
        # Read the image
        image = Image.open(os.path.join(path, bad_filename))

        # Convert bad extensions to PNG
        fixed_filename = bad_filename
        for bad_extension in ['.jpg', '.JPG', '.JPEG', '.jpeg']:
            fixed_filename = fixed_filename.replace(bad_extension, '.png')

        # Save the new image
        image.save(os.path.join(path, fixed_filename))

        # Remove the old, lossy image
        os.remove(os.path.join(path, bad_filename))