# `smart-ml`

This repository contains the latest machine learning efforts for automated interpretation of rapid test kits (e.g. LFA).

Within each directory inside this repository, you will find a `README.md` which lists preliminary information about the directory and its contents. These documentations will often reference various Google Docs documents we've created to supplement the code and explain things in more detail. All of these can be found in the [*Smart-ML*](https://drive.google.com/drive/folders/1QeddvKRp2d0rvLOPvWf5dpihIsNCTjS4?usp=sharing) Google Drive folder. More explicitly, the documents include: [*Master Documentation*](https://docs.google.com/document/d/1eRP_gG-3BTyd0klc8dh-bUcfu7-rAgIgQA7JTyZlhcU/edit?usp=sharing), [*Object Detection Training & Evaluation Documentation*](https://docs.google.com/document/d/1Fr7jmvq7pT32gJiXAZSnoXWM-ILXang1Cnx0l-aXoaY/edit?usp=sharing), [*Lambda Function Deployment Documentation*](https://docs.google.com/document/d/1Bc8auMMP5YS6ITGmb6v83w_IyznPnPRZsd9wsfgkQbs/edit?usp=sharing), [*Cloud Pipeline Inference Documentation*](https://docs.google.com/document/d/1Lj-oPvLd338PodmBPKz50tBA_p9gLbAnB81T9-gMYDA/edit?usp=sharing), [*Cloud Model Deployment Documentation*](https://docs.google.com/document/d/1EAmBFSLx-ufW4sXXMWB2YcmJvLxy9XkA-dbNRiu1M6M/edit?usp=sharing), [*Cloud Functions Overview*](https://docs.google.com/document/d/1NL2qoY9VUvFgX5ALlg1pukxzaWW245PteRWHxt3nDx8/edit?usp=sharing).

## Contents

* `classification/`: This is the main folder fo classification training, evaluation, and deployment. For the actual code for training and development, please contact Jiawei Ma from DVMM. An ideal setup would be to put the main training code directly under `classification`. You can update the README after this.

* `data/`: Placeholder data folder for storing **raw** and membrane images. **NOTE**: There are many placeholder `data/` folders in this repository, but you can feel free to use this outermost folder for all operations. Alternatively, as different machine learning processes require different types of data, you might use the `data/` folder in each component.

* `labelling/`: This folder contains code and utilities for object-detection bounding-box and polygon mask labelling via the `labelme` software. **NOTE**: The folder includes a `labelme_requiremenst.txt` which has the required packages for labelling and can be used to create a virtual environment.

* `lambda-functions/`: This folder contains AWS Lambda Function codes, configuration files, AWS Lambda Layers, and individual manufacturer spec files for each test kit. You can also find the same files in `AWS -> Console -> Lambda`.

* `misc/`: This folder contains misc documents that can be useful for various tasks.

* `object-detection/`: This folder contains code and utilities for training and evaluation of, running inference on, and deploying object detection models.

* `outlier/`: This folder (for now) contains the code for BTNx outlier detection / anomaly detection model we have deployed. Feel free to add other outlier models here if needed.

* `s3/`: Contains code for downloading logs and folders from S3 for investigating failure cases and checking the logs of our AWS Lambda Functions which contain intermediate image processing outputs.

* `sagemaker/`: This folder is merely a snapshot of our AWS SageMaker Notebook Instance named `deploy-to-aws`. Please visit `AWS -> Console -> Notebook instances -> deploy-to-aws -> Open Jupyter OR Open JupyterLab` to see the complete notebook. Only includes `btnx_deploy_and_predict.ipynb` for now, which handles the deployment of all related models for the BTNx test kit.

* `serverside_testing/`: Contains code for testing the serverside locally. With the script inside this folder, you can run batch inputs and get predictions from the cloud model quickly.

* `requirements.txt`: This is the general requirements file, which contains the Python libraries and packages we have utilized in this project. **NOTE**: However, please note that it also includes irrelevant libraries and packages. I haven't created a virtual environment (with the exception of `labelling/`) for this project, and this file includes all of the packages I've installed for various reasons. Feel free to dig into this deeper and remove unnecessary packages. Nevertheless, you can run `pip install -r requirements.txt` to install all of these packages.
