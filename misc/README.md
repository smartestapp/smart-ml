# `misc`

This folder contains misc documents that can be useful for various tasks. 

## Contents

* `aws_cli_documentation.md`: This is a documentation file for deploying AWS SageMaker models via CLI. We haven't used the CLI too much, all of the model deployments were done from the AWS Console. Nevertheless, this document can be helpful for quicker deployment from the command-line.

* `google_style_guide.pdf`: Google's Python style guide. When in doubt to what programming convention to use, this document can come in handy.

* `lambda_layer_creation.png`*: Describes the steps involved in creating a custom AWS Lambda Layer for using non-standard Python libraries (e.g. `numpy`) in AWS Lambda Functions.

* `remove_exif.py`: Photos downloaded from phones might have EXIF information, which might cause rotation issues. For example, you might notice that certain images appear upright on your desktop, but when read with `OpenCV` they might appear rotated. It's always useful to run this on a newly acquired folders of images to make sure. **NOTE**: This script additionally rotates images in landscape mode to portrait mode; this was an assumption we could make with our existing datasets. This might need to be changed in the future.

* `resize.py`: A script to resize images by capping them at maximum 800 pixels of height, and adjusting the width in proportion. The object detection model has a processing limit and therefore downsampling to 800 pixels might be a useful operation.

* `jpg_to_png.py`: A script to convert JPG / JPEG images to PNG format. This script may come in handy for `object-detection/`; we usually want the **masks** to be in PNG format.