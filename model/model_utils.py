from copy import deepcopy
from types import MethodType
import megengine.hub
import megengine.functional as F


def load_backbone():
    model = megengine.hub.load("megengine/models", "resnet50", pretrained=True)
    model.out_features = model.fc.in_features

    def forward(self, x):
        x = x[:, ::-1, :, :]  # N(RBG)HW -> N(BGR)HW
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
