"""
Example Call:
python download_logs_s3.py --directory_prefix='logs/Saturday August, 08' --ids_file=../batch_test_inputs.txt  --get_special='best_rotated_membrane'

python download_logs_s3.py --bucket_name=sialabkitimages --directory_prefix='shs_uploads'
python download_logs_s3.py --directory_prefix='logs/' --images_dir=shs_uploads/SH00066 --get_special='best_rotated_membrane' 
"""

import os
import boto3
from tqdm import tqdm
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Script to download logs from S3 bucket')
parser.add_argument('--bucket_name', default='sagemaker-us-east-2-364546270619', help='S3 bucket name', type=str)
parser.add_argument('--directory_prefix', help='Directory prefix to download logs from.', type=str, required=True)
parser.add_argument('--ids_file', default=None, type=str, help='A .txt file where each line is an image URL whose ID will be extracted!')
parser.add_argument('--get_special', default=None, type=str, help='Specify specific filenames and only extract those!')
parser.add_argument('--darrell_csv', default=None, type=str, help='The specific formatted CSV/Excel file Darrell sent us')
parser.add_argument('--images_dir', default=None, type=str, help='Specify the directory containing images to extract (NOTE: This is a local directory)')
args = parser.parse_args()

if args.darrell_csv and args.images_dir:
    print('This mode is not known!')

if args.ids_file:
    ids = [os.path.splitext(os.path.split(line.strip())[-1])[0] for line in open(args.ids_file, 'r').readlines()]
    print('Got IDs: ', ids)

if args.darrell_csv:
    if args.darrell_csv.endswith('.csv'):
        df = pd.read_csv(args.darrell_csv)
        filenames = df['cu_key (S)'].values.tolist()
    elif args.darrell_csv.endswith('.xlsx'):
        df = pd.read_excel(args.darrell_csv)
        # filenames = df['cu_key'].values.tolist()
        filenames = [os.path.splitext(f)[0] for f in df['cu_key (S)'].values.tolist()]
        print('Got %d filenames!' % len(filenames))
    else:
        raise ValueError('Extension not recognized!')

elif args.images_dir:
    filenames = [os.path.splitext(f)[0] for f in os.listdir(args.images_dir) if f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.png') or f.endswith('.PNG') or f.endswith('.JPEG') or f.endswith('.jpeg')]
    print('Got %d filenames!' % len(filenames))


def download(bucket_name, directory_prefix):
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)

    for obj in tqdm(bucket.objects.filter(Prefix=directory_prefix), desc='Downloading Folders'):
        if args.ids_file and os.path.dirname(obj.key).split(' | ')[-1] not in ids:
            continue
        if not os.path.exists(os.path.dirname(obj.key)) and not args.get_special:
            os.makedirs(os.path.dirname(obj.key))

        if args.darrell_csv or args.images_dir:
            # if obj.key not in filenames:
            if not os.path.dirname(obj.key).split(' | ')[-1] in filenames:
                continue

        if args.get_special:
            if not os.path.exists(args.directory_prefix):
                os.makedirs(args.directory_prefix)
            if args.get_special in obj.key:
                bucket.download_file(obj.key, os.path.join(args.directory_prefix, '%s.jpg' % os.path.dirname(obj.key).split(' | ')[-1]))
        else:
            bucket.download_file(obj.key, obj.key)

download(bucket_name=args.bucket_name, directory_prefix=args.directory_prefix)
