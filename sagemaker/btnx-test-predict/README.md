# `sagemaker/btnx-test-predict/`

This folder is merely a snapshot of our AWS SageMaker Notebook Instance named `deploy-to-aws`. Please visit `AWS -> Console -> Notebook instances -> deploy-to-aws -> Open Jupyter OR Open JupyterLab` to see the complete notebook. We are including this here mainly for documentation purposes, and it is limited in content due to size limits of GitHub. For example, the current folder would normally contain `.mar` files for all of the to-be-deployed models. These `.mar` files can be generated from the respective `deployment/` subdirectories in this repository (e.g. `object-detection/deployment`. The current folder only shows the notebook for the BTNx test kit, whereas the complete notebook includes code for all test kits. These notebooks handle the deployment of i) object-detection, ii) outlier, and iii) classification models to AWS SageMaker which are then consumed by our AWS Lambda Functions as discussed in `lambda-functions/`. Furthermore, the cloud deployment process is explained in more detail in [*Cloud Model Deployment Documentation*](https://docs.google.com/document/d/1EAmBFSLx-ufW4sXXMWB2YcmJvLxy9XkA-dbNRiu1M6M/edit?usp=sharing).

## Contents

* `serve`: This is a copy of the `torchserve` library, which we'll be using for model deployment.
* `Dockerfile`: Contains all the required commands for assembling a Docker image.
* `aws-sagemaker-notebook-instance-config.png`: Configuration for the AWS SageMaker Notebook Instance named `deploy-to-aws`. You will be using this notebook to deploy the models.
* `btnx_deploy_and_predict.ipynb`: Main notebook for deploying BTNx models (object detection, outlier, and classification) to AWS SageMaker.
* `config.properties`: Configuration file for to-be-deployed SageMaker models.
* `docker-entrpoint.sh`: Docker entrypoint commands; spins up `torchserve` for model deployment just as we do locally.
* `extracted_membrane.jpg`: The extracted membrane of the `initial_image.jpg`. We use these for testing the deployed classification model and making sure it gives outputs as expected.
* `initial_image.jpg`: The raw input image to be fed to the object detection model. We use these for testing the deployed object detection model and making sure it gives outputs as expected.
* `invalid_zone.jpg`: The cropped control zone of an anomalous membrane. We use these for testing the deployed outlier model and making sure it gives outputs as expected.
* `requirements.txt`: List of libraries and versions for model deployment. **NOTE**: You don't need to install these locally. The `btnx_deploy_and_predict.ipynb` and other respective Juypter Notebooks for other kits handle this.

