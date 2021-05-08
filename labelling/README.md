# `labelling`

This folder contains code for object detection labelling. Checkout **Labelling Instructions** in this README to see the steps involved for labelling.

## Contents

* `batches/`: This is a folder which will be populated at the last step via `load_batch.py`.
* `data/`: This is where you are expected to put the data you want to label. Other paths will also work with the mentioned scripts down below.
* `get_masks.py`: Script for creating the masks from the JSON file with annotations.
* `labelme2coco.py`: Script for creating a JSON file to be consumed by `get_masks.py`.
* `load_batch.py`: Script for transferring the annotated images and their masks 
* `labelme_requirements.txt`: Text file with Python libraries and versions to be used together with the `labelme` software.


## Labelling Instructions

We have uploaded a video [here](https://drive.google.com/drive/folders/1QeddvKRp2d0rvLOPvWf5dpihIsNCTjS4?usp=sharing) which shows the instructions without the postprocessing part (i.e. only labelling). The entire workflow is outlined below:

1. Create a new virtual environment called `labelme` with your preferred method (e.g. `conda`, `virtualenv`, etc.) and run `pip install -r labelme_requirements.txt` to create the environment with `labelme`, which is the software we'll use to label our images. **NOTE**: At the time we downloaded the `labelme` software, only certain versions of certain libraries were compatible, and therefore we highly recommend a new virtual environment setup for this portion.

2. Place images to `data/{test_kit_id}_{subset}_images`.

3. Remove / fix EXIF information via `python ../misc/remove_exif.py --dir=`data/{test_kit_id}_{subset}_images`. This will fix the problem with rotated images. **NOTE**: However, this might introduce wrong-direction rotation in some images. Go through the images and make sure none of them are upside-down, if so fix it manually!

4. Run `conda activate labelme` and then `labelme` from command line.

5. Annotate images in `data/{test_kit_id}_{subset}_images` using `labelme` -> checkout from here: https://github.com/wkentaro/labelme. This process will create .JSON files inside `data/{test_kit_id}_{subset}_images`. 

6. Call 'conda deactivate' to get back to your original environment where you have OpenCV and other required tools installed!

7. Call `python labelme2coco.py --dir=data/{test_kit_id}_{subset}_images` which will create a `{test_kit_id}_{subset}_images.json` JSON file`.

8. Call `python get_masks.py --images_dir=data/{test_kit_id}_{subset}_images --json_path={test_kit_id}_{subset}_images.json --output_dir=batches/{test_kit_id}_{subset}_masks` to create the masks -> these will be used to train the Mask R-CNN.

9. Call `python load_batch.py --source=data/{test_kit_id}_{subset}_images --target=batches/{test_kit_id}_{subset}_images` to load *only* the annotated images to a new folder with the same name under dir. `batches`. You can then put this new folder under `object-detection/data` to be used
for training and evaluation purposes. **NOTE**: You should also delete the intermediate JSON files, each of which belong to a image, in this folder as the `batches/{test_kit_id}_{subset}_masks` will have the annotations and you no longer need these. **NOTE**: `object-detection` expects the data folders to contain only image files and therefore this step is important!

**NOTE**: Above {subset} could be 'train', 'test', 'val', or just empty ('').
