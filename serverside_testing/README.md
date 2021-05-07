# `serverside_testing`

Contains code for testing the serverside locally. With the script here, you can run batch inputs and get predictions from the cloud model quickly.

## Contents

* `batch_serverside.test.py`: Script for getting predictions from the cloud. The parameters of this script are within the code, feel free to change it to `argparse` if you want to run from command line. The important parameters are:
	* `cloud_function_url`: The URL of the Lambda Function for inference. This can be found in API Gateway. Commented-out lines should also be the most current URLs.
	* `LABELS_PATH`: Path to a Excel file which contains the labels. Set to `None` if you don't have the labels to compare. **NOTE**: This file should be in a specific format, you might have to change the node if a new format is introduced (e.g. the inclusion of a new column or name change of a column).
	* `PREDICTIONS_DIR`: Feel free to skip this variable.
	* `OUTPUT_PATH`: Feel free to skip this variable. It's default value is `output.txt` and this just writes the predictions to a text file.
	* This script will also create `filenames_correspondence.txt` file inside the specified `PREDICTIONS_DIR`. Feel free to ignore this part as well; this is just for getting filenames that we have processed.
	* We pass the filenames that we want to run inference on via the `batch_test_inputs.txt` file which we'll talk about next.

* `batch_test_inputs.txt`: We write one filepath in each line for each image we want to process. The filepaths are with respect to our AWS Bucket `sialabkitimages`.
