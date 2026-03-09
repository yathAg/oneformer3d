import torch.nn as nn

from mmdet3d.registry import MODELS
from .ptv3 import PointTransformerV3


@MODELS.register_module()
class PointTransformerV3Backbone(nn.Module):
    """Wrapper to register PTv3 as an mmdet3d backbone."""

    def __init__(self, **kwargs):
        super().__init__()
        self.model = PointTransformerV3(**kwargs)

    def forward(self, data_dict):
        return self.model(data_dict)
