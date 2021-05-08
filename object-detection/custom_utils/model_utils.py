from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def get_instance_segmentation_model(num_classes, hidden_size):
    """USE THIS: This configuration gets higher accuracy than the above..."""
    # Load full MaskRCNN with ResNet50 backbone trained on COCO
    model = maskrcnn_resnet50_fpn(pretrained=True)

    # Replace the box classifier with a new one that suits our needs
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # Replace the mask classifier with a new one that suits our needs
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_size, num_classes)
    
    return model