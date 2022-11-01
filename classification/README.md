# `AutoAdapt POC Classification`

This subfolder contains the latest machine learning efforts on classification side for automated interpretation of rapid test kits (e.g. LFA).

Within each directory inside this repository, you will find a `README.md` which lists preliminary information about the directory and its contents. 

## Contents

* `evaluation/`: This is the folder for demo. The command is provided in the bash file and you can easily run it with `sh evaluation/run.sh` where the test output will be automatically generated and summarized in an xlsx file. 


## System Requirement

* 1. This code is evaluated on Linux Server with Pytorch. The function has been tested on Pytorch with versions 1.6~1.8.

* Environment Configuration. The environment configuration consists of the python environment through Anaconda and pytorch environment. The network code is based on pytorch and we would recommend using pytorch with versions no earlier than 1.5. 

(Skip the next two bullets if you use instances of Amazon EC2 with a pre-built anaconda environment)

* 2. Anaconda Configuration. To install anaconda, please go <a href="https://www.anaconda.com/products/individual#macos/" target="_blank">here</a> to download the installation package. All versions of anaconda packages will be available on this <a href="https://repo.anaconda.com/archive/" target="_blank">archive</a>. A quick command for downloading and running is 
```
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
sh Anaconda3-2021.05-Linux-x86_64.sh 
``` 

* 3. Pytorch Configuration Prep. If the pytorch environment is pre-built, when you use conda env list, a list of environments will appear and you may find things similar to “torch_latest_py37”. We would recommend first create a clean virtual environment through anaconda, i.e., `conda create -n <env_name> python=3.7 anaconda`. Then, go to the environment through `conda activate <env_name>`. Please first check the CUDA version you have. Generally, you will be able to read all available CUDA versions after going to `cd /usr/local`.

* Pytorch Instillation. Pick up the Pytorch version and the webpage will automatically generate the command. The following is an example with CUDA version 10.2 and the latest pytorch version. 
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch.
```

* 4. Additional Packages/Tools. I have listed a few more packages in the image_classification/requirements.txt, you can run pip install -r requirements.txt for a quick installation. In detail, it includes
```
pandas for Excel sheet processing,
pdb for code debugging
cv2 for image processing (pip install opencv-python)
```

* (The overall processing time for installation is only about 2 mins)

## Detailed Instruction

### Data Preparation

* Go to the directory `classification/evaluation` and download `demo_data.tar.gz` with the <a href="https://drive.google.com/file/d/15ayr0Q9zC_o_NSwvoKjPqq7LDnQ0zVs8/view?usp=sharing" target="_blank">link</a> . Then, the unzipped folder contains the data for demo and labels for evaluation.

* Go to the directory `classification/evaluation` and download `logs.tar.gz` with the  <a href="https://drive.google.com/file/d/1JyK8AW8WGJhO5SDztnFCVUuy-1_4QnFw/view?usp=share_link" target="_blank">link</a>. Then, the unzipped folder contains the model checkpoints for testing.

### Hyper-Parameter & Parameters

* gpu = 1 → specify the gpu(s) for the network employing, jointly work with Function set_gpu. Default value: 0

* kit_id = ACON_Ab → specify the test kit product, which is necessary for zone cropping meta-data loading, and pre-trained model parameter loading.  

* threshold = 0.5 → for binary classification task, a sample with detection score larger than the threshold will be classified as positive. The threshold will be applied for both control zone and test zone. If you want different standards for the classification of different zones, please refer to the function postprocessing. 

### Usage

The function defined in evaluation/classifier_ss.py is for evaluation and can be used as reference for deployment. In the evaluation/main.py, after you initialize the whole system, you just need to input the kit_id to specify the kit-specific metadata and model.

```
kit=ACON_Ab
python main.py --kit-id=${kit}
```

Note, the function defined in evaluation/classifier_ss.py is only for prediction. To calculate the metric, you need to customize them in the evaluation/main.py.

### Running Time
The processing time for loading model is about 6s. The processing time for each image is less than 0.1s. As such, we read the list of files from the `data/Demo-Sample.xlsx` (after unzipping) and processing all of them together.

### Output
A xlsx file starting with `prediction` is generated, the contents includes the file name, the label, prediction, and related accuracy metrics.