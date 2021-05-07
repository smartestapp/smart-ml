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


class OutlierHandler(object):
    """
    Outlier handler class for binary classification on whether the inputted 
    control zone is normal or anomalous (i.e. outlier), based on comparing
    to a threshold on the reconstruction loss of an (V)AE.
    This handler takes an RGB image of control zone and
    returns the detected class (i.e. 0 for normal and 1 for anomaly)
    """

    def __init__(self):

        ############################################ ONLY CONFIGURE THESE EACH TIME!!!
        self.num_channels = 3
        self.batch_size = 32
        self.img_size = 64
        self.kernel_size = 3
        self.latent_dim = 100
        self.hidden_dims = [32, 64, 128, 256, 512]
        self.beta = 0
        self.gamma = 255
        self.encoder_activation_fn = F.leaky_relu
        self.decoder_activation_fn = F.leaky_relu
        self.output_activation_fn = torch.sigmoid
        self.use_batch_normalization = True
        self.threshold = 1.50
        self.model_filename = "btnx_outlier.pt"
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
        model_def_path = os.path.join(model_dir, "outlier_model.py")
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file for model definition")


        class Namespace:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        from outlier_model import ConvolutionalVAE
        self.model = ConvolutionalVAE(num_channels=self.num_channels,
                                      batch_size=self.batch_size,
                                      img_size=self.img_size,
                                      kernel_size=self.kernel_size,
                                      latent_dim=self.latent_dim,
                                      hidden_dims=self.hidden_dims,
                                      beta=self.beta,
                                      gamma=self.gamma,
                                      encoder_activation_fn=self.encoder_activation_fn,
                                      decoder_activation_fn=self.decoder_activation_fn,
                                      output_activation_fn=self.output_activation_fn,
                                      use_batch_normalization=self.use_batch_normalization,
                                      device=self.device)

        # Load baseline feature extraction + adapted parameters ON SD IGG test
        state_dict = torch.load(model_pt_path, map_location=self.device)

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

        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])

        image = Image.open(io.BytesIO(image)).convert('RGB')
        image = transform(image)

        return image


    def inference(self, img):
        """Predict the classes and bounding boxes in an image using a trained deep learning model."""

        # NOTE: Image is reshaped to (C, H, W) already via the previous ToTensor() transformation.
        # This is the expected shape for a Torch variable to be passed through the model.
        zone = Variable(img).to(self.device)

        x_hat, mu, logvar = self.model(zone.unsqueeze(0))
        loss = self.model.loss_function(x=zone, x_hat=x_hat, mu=mu, logvar=logvar)
        return [0, loss.item()] if loss < self.threshold else [1, loss.item()]


    def postprocess(self, inference_output):
        prediction, loss = inference_output
        return [[{'anomaly_detection': str(prediction), 'loss': str(loss)}]]


_service = OutlierHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data