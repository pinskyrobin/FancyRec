import torchvision.models as models
import torch.nn as nn


# 提取图像特征的训练好的resnet152
class ResNet152(nn.Module):
    def __init__(self, model):
        super(ResNet152, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
        )

    def forward(self, feature):
        feature = self.features(feature)
        return feature
