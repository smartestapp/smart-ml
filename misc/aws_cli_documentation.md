# AWS CLI Deployment Documentation & Utilities

## LOCAL

* Train your PyTorch model.
* Save the trained model as a `.pth`. 
* Put the `.pth` inside a folder.
* Call `tar -zcvf <NAME>.tar.gz <NAME>` from command line, replacing `<NAME>` with the folder name from the previous step. It might be the case that `.DS_Store` is also included in the tarball, and this is fine.

## SERVER

* Load the data AWS S3 Bucket via: `aws s3 cp <NAME>.tar.gz s3://<BUCKET>/<PATH>`.
The `<BUCKET>` will almost always be replaced by the default bucket, which is `sagemaker-us-east-2-364546270619`. At any point, you can check these from the AWS Console as well.

* Create the model: 
```aws sagemaker create-model --model-name <MODEL_NAME> --primary-container
ContainerHostname=<CONTAINER_HOSTNAME>,Image=<IMAGE>,Mode=<MODE>,ModelDataUrl=<MODELDATAURL> --execution-role-arn <EXECUTION_ROLE_ARN>```

	* The `<MODEL_NAME>` is arbitrary but it should be descriptive. We also describe a primary container, which will have the Docker image that contains inference code, artifacts (i.e. saved model tarballs from previous local training). 

	* `<CONTAINER_NAME>` is also arbitrary but it could be set to `<MODEL_NAME>-container` as a convention. For 	example, if model name is `membrane-detection`, then the container name would be `membrane-detection-container`. 

	* `<IMAGE>` is the URL to the Docker image; for use of ease, we will be using AWS-provided docker images which 	have global URLs within the AWS system. The full list can 	be found from [this](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-images.html) link. Preferably, we are using 	**elastic inference containers**, which are resources attached to an Amazon EC2 instance to accelerate deep 	learning inference workloads. As opposed to deploying a container with a readily available GPU, this option is cost-effective. Currently, we are using `763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-inference-eia:1.3.1-cpu-py36-ubuntu16.04` for this parameter. Notice the inclusion of `us-east-2`, which is the region of our AWS instance, and `1.3.1` version requirement for PyTorch. Unfortunately, only 1.3.1 is available at the time of writing for elastic inference. 

	* `<MODE>` is either one of `SingleModel` or `MultiModel`. For now, we will be rolling with 	`SingleModel` but in the future we should have more to say regarding the latter.

	* `<MODELDATAURL>` is the AWS S3 path of the tarball from 	the first step.

	* `<EXECUTION_ROLE_ARN>` is the Amazon Resource Name (ARN) of the IAM role that Amazon Sagemaker can assume to access model artifacts and docker image for deployment. You can check the `Roles` tab in the `IAM` section in AWS Console to locate this role. In the time of writing, this parameter is `arn:aws:iam::364546270619:role/service-role/AmazonSageMaker-ExecutionRole-20200215T211088`.

* Create and endpoint configuration: `aws sagemaker create-endpoint-config --endpoint-config-name <ENDPOINT_CONFIG_NAME> --production-variants VariantName=<VARIANTNAME>,ModelName=<MODEL_NAME>,InitialInstanceCount=<INITIALINSTANCECOUNT>,InstanceType=<INSTANCETYPE>,InitialVariantWeight=<INITIALVARIANTWEIGHT>,AcceleratorType=<ACCELERATORTYPE>`.

	* `<ENDPOINT_CONFIG_NAME>` could match `<MODEL_NAME>` from the previous step as a convention.

	* `--production-variants` is a list of objects, one for each model that you want to host at this endpoint.

	* `<VARIANTNAME>` should be set `AllTraffic`.

	* `<MODEL_NAME>` should match `<MODEL_NAME>` from the previous step.

	* `<INITIALINSTANCECOUNT>` should be set 1.

	* `<INSTANCETYPE>` is the machine learning instance type that you want to use. You can check [here](https://aws.amazon.com/sagemaker/pricing/instance-types/) for the full list and check the pricing from [here](https://aws.amazon.com/sagemaker/pricing/). A default option for this parameter would be `ml.t2.xlarge`.

	* `<INITIALVARIANTWEIGHT>` should be set to 1.0.

	* `<ACCELERATORTYPE>` is an instance type that specifically does inference acceleration. Check the `Inference Acceleration` section on the previous links to local the available options. A default option for this parameter would be `ml.eia1.large`.


* Create the endpoint: `aws sagemaker create-endpoint --endpoint-name <ENDPOINT_NAME> --endpoint-config-name <ENDPOINT_CONFIG_NAME>`.

	* `<ENDPOINT_CONFIG_NAME>` should be copied from the previous step.

	* `<ENPOINT_NAME>` can also be set equal to the `<ENDPOINT_CONFIG_NAME>` for convenience.

## UTILITIES

### Testing
* Create an notebook instance via: `aws sagemaker create-notebook-instance --notebook-instance-name <NOTEBOOK_INSTANCE_NAME> --instance-type <INSTANCE_TYPE> --role-arn <ROLE_ARN>`.
	* `<NOTEBOOK_INSTANCE_NAME>` is an arbitrary notebook name that you want to give.
	* `<INSTANCE_TYPE>` is the same as before, no need to use a different option.
	* `<ROLE_ARN>` is the Amazon Resource Name (ARN) of the role you want to create the notebook with. By default, we can use the Executor Role, which is given by the following name: `arn:aws:iam::364546270619:role/service-role/AmazonSageMaker-ExecutionRole-20200215T211088`.
	* In the SageMaker console, we can click `Open Jupyter` to open the notebook, and open a new notebook using one of the specified notebook options. For PyTorch and Python 3.6, this would be `conda_pytorch_p36`.
### Listing & Verifying
* Show existing models via: `aws sagemaker list-models`
* Show existing endpoint configs via: `aws sagemaker list-endpoint-configs`
* Show existing endpoints via: `aws sagemaker list-endpoints`
* Show existing notebook instances: `aws sagemaker list-notebook-instances`

### Cleaning & Deleting
Make sure to do the following three steps in order to clean a model:

* You can delete a model via: `aws sagemaker delete-model --model-name <MODEL_NAME>`
* You can delete an endpoint config via: `aws sagemaker delete-endpoint-config --endpoint-config-name <ENDPOINT_CONFIG_NAME>`
* You can delete an endpoint via: `aws sagemaker delete-endpoint --endpoint-name <ENDPOINT_NAME>`

### Misc
* Check [this](https://stackoverflow.com/questions/55791047/how-to-send-numpy-array-to-sagemaker-endpoint-using-lambda-function) link to understand how to send numpy array to sagemaker endpoint using lambda function.
* See [this](https://aws.amazon.com/blogs/machine-learning/bring-your-own-pre-trained-mxnet-or-tensorflow-models-into-amazon-sagemaker/) to understand the folder structure.
* Get the region of AWS account via: `aws configure get region`.


