import argparse
import os
from tqdm import tqdm
import cv2
import json


parser = argparse.ArgumentParser(description="Script for generating zones and visually inspecting them.")
parser.add_argument("--membranes_dir", help="Path to directory containing membrane images.", type=str, required=True)
parser.add_argument("--kit_id", help="ID of the kit; should match the key of the JSON.", type=str, required=True)
parser.add_argument("--json_path", help="Path to the JSON that has kit dimensions.", type=str, required=True)
args = parser.parse_args()

output_dir = '%s-zones' % args.kit_id
if not os.path.exists(output_dir):
	os.makedirs(output_dir)
print('The cropped zones will be saved to %s' % output_dir)

kit_data = json.load(open(args.json_path, 'r'))[args.kit_id]
num_zones = kit_data['dimensions']['zones']['n']

membrane_paths = [os.path.join(args.membranes_dir, f) for f in os.listdir(args.membranes_dir) 
                  if f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.JPEG') or f.endswith('.jpeg') or f.endswith('.png') or f.endswith('.PNG')]

for p in tqdm(membrane_paths, desc='Cropping Zones & Saving Them'):
	membrane = cv2.imread(p)
	height, width, num_channels = membrane.shape

	for i in range(num_zones):
		coordinates = kit_data['dimensions']['zones']['zone%d' % (i + 1)]
		zone = membrane[int(coordinates['y'] * height): int(coordinates['y'] * height) + int(coordinates['h'] * height), :, :]
		cv2.imwrite(os.path.join(output_dir, '%s-zone%d.jpg' % (os.path.splitext(os.path.split(p)[-1])[0], i)), zone)
