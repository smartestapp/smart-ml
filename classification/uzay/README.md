# `classification/uzay`

This is the main folder for classification evaluation and deployment. **NOTE**: For the actual code for training and development, please contact Jiawei Ma from DVMM. An ideal setup would be to put the main code directly under `classification`. Since the contents here are mainly focused on evaluation, we'll be skipping all details related to training.

## Contents

* `deployment/`: Files for deployment to cloud. **NOTE**: Before running the code within this sub-folder, make sure to place the `.pt` or `.pth` file of the saved classification model here. The deployment procedure for all types of machine learning models are described in [*Cloud Model Deployment Documentation*](https://docs.google.com/document/d/1EAmBFSLx-ufW4sXXMWB2YcmJvLxy9XkA-dbNRiu1M6M/edit?usp=sharing).
	* `deploy.sh`: This bash script creates a `.mar` file for the classification model, and subsequently starts `torchserve` for local deployment. The `.mar` file will be later deployed to AWS SageMaker. **NOTE**: Make sure to replace `--serialized-file quidelag_classifier.pth` with the desired saved model filepath.
	* `kill.sh`: Kills `torchserve` and deletes the temporary folders `model_store` and `logs`.
	* `classifier_handler.py`: Handles the inference of the deployed model.
	* `classifier_model.py`: Contains the implementation of the model.
	* `test.sh`: Bash script for testing. Feel free to change this to match different tests you want to perform.
	* `kit_data.json`: JSON that contains manufacturer specs for various kits. Make sure this is the most up-to-date version.


