import io
import logging
import json
import numpy as np
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

logger = logging.getLogger(__name__)


class MaskRCNNInstanceSegmenterHandler(object):
    """
    MaskRCNN handler class for instance segmentation on kits and membranes. 
    This handler takes an RGB image and returns list of detected classes, 
    bounding boxes, and instance segmentation masks respectively.
    """

    def __init__(self):
        ############################################### ONLY CHANGE THIS!!
        self.model_filename = "btnx_maskrcnn_weights.pth"
        ###############################################

        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False

        self.checkpoint_extension = '.pth'

    def initialize(self, ctx):
        """First try to load torchscript else load eager mode state_dict based model"""

        properties = ctx.system_properties
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        model_dir = properties.get("model_dir")

        # Read model via serialized .pt/.pth file
        model_pt_path = os.path.join(model_dir, self.model_filename)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt or model.pth file")

        # Read model definition file
        model_def_path = os.path.join(model_dir, "maskrcnn_model.py")
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file for model definition")

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")

        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:
            logger.warning('Missing the index_to_name.json file. Inference output will not include class name.')


        from maskrcnn_model import MaskRCNNInstanceSegmenter
        state_dict = torch.load(model_pt_path, map_location=self.device)
        self.model = MaskRCNNInstanceSegmenter()
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.debug('Model file {0} loaded successfully'.format(model_pt_path))
        self.initialized = True

    def preprocess(self, data):
        """
         Scales, crops, and normalizes a image for a PyTorch model,
         returns an Numpy array
        """
        image = data[0].get("data")
        if image is None:
            image = data[0].get("body")

        my_preprocess = transforms.Compose([transforms.ToTensor()])
        image = Image.open(io.BytesIO(image))
        image = my_preprocess(image)
        return image

    def inference(self, img, threshold=0.9):
        """Predict the classes and bounding boxes in an image using a trained deep learning model."""

        # NOTE: No need to reshape image to (C, H, W), the usual expected shape for a Torch variable 
        # to be passed through a model, transforms.ToTensor() already takes care of this.
        img = Variable(img).to(self.device)

        # Pass the image to the model
        pred = self.model([img])  

        # Get the prediction labels (i.e. kit or membrane) and their confidence scores
        pred_labels = [i for i in list(pred[0]['labels'].cpu().numpy())] 
        pred_scores = list(pred[0]['scores'].cpu().detach().numpy()) 

        # Get bounding boxes (i.e. 4 coordinates) and masks (i.e. 2D array with shape (H, W))
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].cpu().detach().numpy())]
        pred_masks =  pred[0]['masks'].cpu().detach().numpy().tolist()

        # Get list of index with score greater than threshold
        idx = [pred_scores.index(x) for x in pred_scores if x > threshold][-1]
        pred_labels = pred_labels[:idx + 1]
        pred_scores = pred_scores[:idx + 1]
        pred_boxes = pred_boxes[:idx + 1]
        pred_masks = pred_masks[:idx + 1]
        
        return [pred_labels, pred_scores, pred_boxes, pred_masks]

    def postprocess(self, inference_output):
        pred_labels, pred_scores, pred_boxes, pred_masks = inference_output
        assert len(pred_labels) == len(pred_scores) == len(pred_boxes) == len(pred_masks)

        # Get names for the labels through the index_to_names JSON if applicable 
        if self.mapping:
            pred_labels = [self.mapping['object_type_names'][i] for i in pred_labels]  

        # Return value should be in a serializable format
        retval = []

        for i, pred_label in enumerate(pred_labels):
            pred_score, pred_box, pred_mask = pred_scores[i], pred_boxes[i], pred_masks[i]

            retval.append({
                'label': str(pred_label),
                'score': str(pred_score),
                'box': str(pred_box),
                'mask': str(pred_mask)
            })

        return [retval]


_service = MaskRCNNInstanceSegmenterHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data