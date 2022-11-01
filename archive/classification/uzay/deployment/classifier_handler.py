import io
import logging
import json
import numpy as np
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ClassifierHandler(object):
    """
    Metalearning Classifier handler class for binary classification (i.e. diagnosis)
    on membrane segment. This handler takes an RGB image of membrane pieces and 
    returns the detected class (i.e. 0 for negative and 1 for positive)
    """

    def __init__(self):

        ############################################ ONLY CONFIGURE THESE EACH TIME!!!
        self.kit_id = 'quidelag'
        self.model_filename = "%s_classifier.pth" % self.kit_id
        ############################################

        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False

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
        model_def_path = os.path.join(model_dir, "classifier_model.py")
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file for model definition")

        class Namespace:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        args = Namespace(model_type='ResNet18ORI')

        from classifier_model import FeatureNet
        self.model = FeatureNet(args=args, mode='cloud_ss', n_class=2, flag_meta=True)

        # Load model
        state_dict = torch.load(model_pt_path, map_location=self.device)['params']

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.debug('Model file {0} loaded successfully'.format(model_pt_path))
        self.initialized = True

        kit_data_path = os.path.join(model_dir, "kit_data.json")
        with open(kit_data_path) as json_file:  
            self.meta = json.load(json_file)

        self.meta_file = self.meta[self.kit_id]

        self.loc_split = self.meta_file['dimensions']['zones']
        self.num_split = self.loc_split['n']
        del self.loc_split['n']

        self.diag_map = self.meta_file['diagnosis_mapping']

        for idx,key in enumerate(self.diag_map):
            zones_str = key[1:-1].split(', ')
            zones_cls = [int(zone_cls) for zone_cls in zones_str]
            if idx == 0:
                self.diagnosis = np.zeros([2]*len(zones_cls),dtype=np.int8)
            self.diagnosis[tuple(zones_cls)] = self.diag_map[key]


    def preprocess(self, data):
        """
         Scales, crops, and normalizes a image for a PyTorch model,
         returns an Numpy array
        """
        image = data[0].get("data")
        if image is None:
            image = data[0].get("body")

        transform = transforms.Compose([
            transforms.Resize((160, 480)),
            transforms.ToTensor(),
            transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), 
                                           np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))]
        )

        image = Image.open(io.BytesIO(image)).convert('RGB').rotate(90, expand=True)
        image = transform(image)

        return image


    def inference(self, img, threshold=0.85):
        """Predict the classes and bounding boxes in an image using a trained deep learning model."""

        # NOTE: Image is reshaped to (C, H, W) already via the previous ToTensor() transformation.
        # This is the expected shape for a Torch variable to be passed through the model.
        img_memb = Variable(img).to(self.device)

        img_zone = []
        img_memb = img_memb.unsqueeze(0)
        for idx,zone_key in enumerate(self.loc_split):
            pos = self.loc_split[zone_key]
            img_zone.append(F.upsample(img_memb[:,:,:,int(pos['y']*480):int(pos['y']*480)+int(pos['h']*480)],size=[160,160],mode='bilinear'))
        img_zone = torch.cat(img_zone, dim=0)

        zone_logits, _ = self.model(img_zone)

        zone_pred = F.softmax(zone_logits, dim=1).argmax(dim=1).cpu().detach().numpy()
        confidence_scores = F.softmax(zone_logits).cpu().detach().numpy()
        diagnosis = self.diagnosis[tuple(zone_pred)]

        return [zone_pred.tolist(), confidence_scores.tolist(), diagnosis.tolist()]


    def postprocess(self, inference_output):
        zone_pred, confidence_scores, diagnosis = inference_output

        return [[{'zone_classification': str(zone_pred), 
                  'confidence_scores': str(confidence_scores),
                  'diagnosis': str(diagnosis)}]]


_service = ClassifierHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data