from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class MaskRCNNInstanceSegmenter(MaskRCNN):
    def __init__(self, num_classes=3, hidden_size=256, **kwargs):
        # NOTE: num_classes should also include the background

        # No need to load pretrained weights, all weights will be replaced by the ones in .mar
        model = maskrcnn_resnet50_fpn(pretrained=False)
        backbone = model.backbone

        # Replace the box classifier with a new one that suits our needs
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        box_predictor = FastRCNNPredictor(in_features_box, num_classes)

        # Replace the mask classifier with a new one that suits our needs
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_size, num_classes)

        # NOTE: num_classes argument for MaskRCNN() should be None 
        # when mask_predictor is specified.
        super(MaskRCNNInstanceSegmenter, self).__init__(backbone=backbone, 
                                                        box_predictor=box_predictor, 
                                                        mask_predictor=mask_predictor,
                                                        **kwargs)