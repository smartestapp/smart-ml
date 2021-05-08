# `s3`

Contains code for downloading logs and folders from S3 for investigating failure cases and checking the logs of our AWS Lambda Functions which contain intermediate image processing outputs.

## Contents

* `download_logs_s3.py`: Script for downloading logs. It is implemented with `argparse` and the script comes with two modes that use command arguments. **NOTE**: The description and default values of the command arguments can be found in the script. Sample calls are also included in the docstring of the script.
	* *Downloading Folders*: Pass in the desired AWS S3 bucket name with `--bucket_name` and specify the directory prefix URL with `--directory_prefix` which is with respect to the bucket with name `--bucket_name`. This will download the folders from S3 to local, i.e. this same folder `s3`.
	* *Downloading Logs*: Our logs are also folders, but they have intermediate image outputs saved as `.jpg`. Our logs are saved in our AWS default S3 bucket, which is `sagemaker-us-east-2-364546270619`, and therefore you can leave the `--bucket_name` argument for this. You need to pass a text filepath through `--ids-file`. The specified text file is created by writing one filepath in each line for each image we want to obtain logs of (**NOTE**: given that this image has bee previously run through the cloud model). The filepaths are with respect to the specified AWS Bucket through `--bucket_name`.