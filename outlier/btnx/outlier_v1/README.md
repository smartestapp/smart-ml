# `outlier/btnx/outlier_v1`

This folder contains the code for BTNx outlier detection / anomaly detection model we have deployed. We have used the VAE model. The outlier model is used to check the control zones within the membrane of the test kit, and returns a reconstruction loss. The reconstruction loss can be interpreted as the likelihood of the given control zone based on prior evidence. Prior evidence is the training dataset we are using for this task, which include "normal" membranes. "Normal" here refers to the fact that they don't have anomalies. Therefore, under this prior, we expect that if the reconstruction error is high, then the control zone and hence the membrane is anomalous. If it is low, we can assume that the membrane is "normal". We use a threshold to decide this. The actual inference can be observed in `lambda-functions/btnx-test-predict` and the mentioned threshold is the `OUTLIER_LOSS_THRESHOLD` Lambda environment variable. **NOTE**: We've only trained and deployed an outlier model for `btnx`.

## Contents

* `losses.py`: Losses used for training the outlier model.
* `models.py`: Includes implementations for ConvolutionalVAE and VAE, two generative models which we used for outlier detection.
* `outlier_v1_info.txt`: Hyperparameters used for training the outlier model.
* `uzays_outlier_detector.ipynb`: This is the Jupyter Notebook we ran on Google Collab to train the model and acquire a saved model file for deployment.
* `deployment`: Files for deployment to cloud. **NOTE**: Before running the code within this sub-folder, make sure to place the `.pt` or `.pth` file of the saved outlier model here. The deployment procedure for all types of machine learning models are described in [*Cloud Model Deployment Documentation*](https://docs.google.com/document/d/1EAmBFSLx-ufW4sXXMWB2YcmJvLxy9XkA-dbNRiu1M6M/edit?usp=sharing).
	* `deploy.sh`: This bash script creates a `.mar` file for the outlier model, and subsequently starts `torchserve` for local deployment. The `.mar` file will be later deployed to AWS SageMaker. **NOTE**: Make sure to replace `--serialized-file btnx_outlier.pt` with the desired saved model filepath.
	* `kill.sh`: Kills `torchserve` and deletes the temporary folders `model_store` and `logs`.
	* `kit_data.json`: JSON that contains manufacturer specs for various kits. Make sure this is the most up-to-date version.
	* `outlier_handler.py`: Handles the inference of the deployed model.
	* `outlier_model.py`: Contains the implementation of the model.
	* `test.sh`: Bash script for testing. Feel free to change this to match different tests you want to perform.