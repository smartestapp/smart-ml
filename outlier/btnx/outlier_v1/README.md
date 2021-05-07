# `outlier/btnx/outlier_v1`

This folder contains the code for BTNx outlier detection / anomaly detection model we have deployed. We have used the VAE model.

## Contents

* `losses.py`: Losses used for training the outlier model.
* `models.py`: Includes implementations for ConvolutionalVAE and VAE, two generative models which we used for outlier detection.
* `outlier_v1_info.txt`: Hyperparameters used for training the outlier model.
* `uzays_outlier_detector.ipynb`: This is the Jupyter Notebook we ran on Google Collab to train the model and acquire a saved model file for deployment.
* `deployment`: Files for deployment to cloud. **NOTE**: Before running the code within this sub-folder, make sure to place the `.pt` or `.pth` file of the saved outlier model here.
	* `deploy.sh`: This bash script creates a `.mar` file for the outlier model, and subsequently starts `torchserve` for local deployment. The `.mar` file will be later deployed to AWS SageMaker.
	* `kill.sh`: Kills `torchserve` and deletes the temporary folders `model_store` and `logs`.
	* `kit_data.json`: JSON that contains manufacturer specs for various kits. Make sure this is the most up-to-date version.
	* `outlier_handler.py`: Handles the inference of the deployed model.
	* `outlier_model.py`: Contains the implementation of the model.
	* `test.sh`: Bash script for testing. Feel free to change this to match different tests you want to perform.