import torch
import torchvision
import torch.nn.modules
import torch.nn as nn
import torchvision.models as models

# class vgg16_bn(nn.Module):
#     def __init__(self, pretrained=False):
#         super(vgg16_bn, self).__init__()
#
#         model = list(torchvision.models.vgg16_bn(pretrained=pretrained).features.children())
#
#         self.model1 = torch.nn.Sequential(*model[:33])
#         self.model2 = torch.nn.Sequential(*model[34:44])
#
#     def forward(self, x):
#         out1 = self.model1(x)
#         out2 = self.model2(out1)
#         return out1, out2
#
# class vgg16(nn.Module):
#     def __init__(self, pretrained=True):
#         super(vgg16, self).__init__()
#
#         model = models.vgg16(pretrained=pretrained)
#
#         vgg_feature_layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2',
#                               'pool1', 'conv2_1', 'relu2_1', 'conv2_2',
#                               'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
#                               'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
#                               'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
#                               'relu4_2', 'conv4_3', 'relu4_3']
#
#         last_layer = 'relu4_3'
#         last_layer_idx = vgg_feature_layers.index(last_layer)
#
#         self.model1 = nn.Sequential(*list(model.features.children())[:last_layer_idx + 1])
#         self.model2 = nn.Sequential(*list(model.features.children())[last_layer_idx + 1:-1])
#
#     def forward(self, x):
#         out1 = self.model1(x)
#         out2 = self.model2(out1)
#
#         return out1, out2

class resnet(nn.Module):
    def __init__(self, layers, pretrained=False):
        super(resnet,self).__init__()

        if layers == '18':
            model = torchvision.models.resnet18(pretrained=pretrained)

        elif layers == '34':
            model = torchvision.models.resnet34(pretrained=pretrained)

        elif layers == '50':
            model = torchvision.models.resnet50(pretrained=pretrained)

        elif layers == '101':
            model = torchvision.models.resnet101(pretrained=pretrained)

        elif layers == '152':
            model = torchvision.models.resnet152(pretrained=pretrained)

        elif layers == '50next':
            model = torchvision.models.resnext50_32x4d(pretrained=pretrained)

        elif layers == '101next':
            model = torchvision.models.resnext101_32x8d(pretrained=pretrained)

        elif layers == '50wide':
            model = torchvision.models.wide_resnet50_2(pretrained=pretrained)

        elif layers == '101wide':
            model = torchvision.models.wide_resnet101_2(pretrained=pretrained)

        else:
            raise NotImplementedError

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x2, x3, x4
