# `data`

You can use this folder to place kit and membrane images to be processed with code inside `classification`, or to be used in training or evaluating the object detection model with code inside `object-detection`, or to be labelled with code inside `labelling`. In the current setup, the mentioned folders (e.g. `object-detection`) might contain their own `data` folders, but ideally in the future, everything can be configured to run with this outermost `data` folder. An alternative design choice is to delete this folder and keep and create `data` folders for each related folder, but this might cause duplication of images as often different processes use the same data.

## CONTENTS

* `PLACEHOLDER`: Feel free to delete this file.