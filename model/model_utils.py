from copy import deepcopy
from types import MethodType
import megengine.hub
import megengine.functional as F


def load_torch_resnet50_pretrained_dict():
    from torchvision.models.utils import load_state_dict_from_url
    from torchvision.models.resnet import model_urls

    pretrained_dict = load_state_dict_from_url(model_urls["resnet50"], progress=True)
    return {k: v.detach().numpy() for k, v in pretrained_dict.items()}


def load_backbone():
    model = megengine.hub.load("megengine/models", "resnet50", pretrained=False)
    pretrained_dict = load_torch_resnet50_pretrained_dict()
    model.load_state_dict(pretrained_dict, strict=True)
    model.out_features = model.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    model.forward = MethodType(forward, model)

    def copy_head(self):
        return deepcopy(self.fc)

    model.copy_head = MethodType(copy_head, model)
    return model
