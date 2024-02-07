import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

from .build import BACKBONE_REGISTRY
from .backbone import Backbone
import pdb


model_urls = {
    "googlenet": "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",
}


# class GoogLeNet(Backbone):
    
#     def __init__(self):
#         super().__init__()

       
@BACKBONE_REGISTRY.register()
def googlenet(pretrained=True, **kwargs):
    model = models.googlenet(pretrained = pretrained)
    # for fine-tuning
    # pdb.set_trace()
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 8)

    # if pretrained:
    #     init_pretrained_weights(model, model_urls["alexnet"])

    return model