# `lambda-functions`

This folder contains AWS Lambda Function codes, configuration files, AWS Lambda Layers, and individual manufacturer spec files for each test kit.

## Contents

* `<KIT_ID>-test-predict/`: `<KIT_ID>` can be `aconag`, `oraquick`, `abbott`, `btnx`, `quidelag`, `accessbioag`, `deepblueag`, `rapidconnectab`, or `aconab`. Respectively for each test kit we have deployed, these folders contain the main lambda function code within `lambda_function.py` and config file `<KIT_ID>`-test-predict.yaml` with environment variables and layers. This setup is replicated exactly in our AWS Account `SMARTest` and can be found in `Lambda -> Functions` in AWS Console. Remember to use `US East (Ohio) us-east-2` as the region (can be changed at the top right) in AWS Console. The environment variables can be configured and modified from `Lambda -> Functions -> <CHOOSE A FUNCTION> -> Configuration -> Environment variables`. **NOTE**: Most of the code is duplicated in each `lambda_function.py`, and these are adopted from `object-detection/main.py`. [*Cloud Pipeline Inference Documentation*](https://docs.google.com/document/d/1Lj-oPvLd338PodmBPKz50tBA_p9gLbAnB81T9-gMYDA/edit?usp=sharing)covers `btnx-test-predict` specifically, but the concepts and code described there applies to all lambda functions currently in use.

* `layers/`: AWS Layers build folders. To add a new layer, you can follow the steps described in `misc/lambda_layer_creation.png`.
	* `cv2-044353a0-ae21-40d6-976e-faffdcfe1e36.zip`: Custom AWS Layer `.zip` for `OpenCV`.
	* `sklearn-b291219a-d95f-435e-9c4b-02da2d5524ea.zip`: Custom AWS Layer `.zip` for `scikit-learn`. **NOTE**: This is not really used / not needed for most lambda functions but still included here for completeness.

* `specs/`: Folder which contains JSON manufacturer specs for each test kit (i.e. `<KIT_ID>`). **NOTE**: These JSONs are uploaded to our default Amazon S3 bucket `sagemaker-us-east-2-364546270619` under the `misc/` subfolder.