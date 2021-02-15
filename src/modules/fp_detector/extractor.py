import torch
import torch.nn as nn
import torchvision

# There is no pretrained model in torch.hub for mnasnet0_75, mnasnet1_3, shufflenetv2_x1.5, and shufflenetv2_x2.0
SUPPORTED_MODELS = {
    'group1': [
        'alexnet',
        'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
        'squeezenet1_0', 'squeezenet1_1',
        'densenet121', 'densenet169', 'densenet161', 'densenet201',
        'mobilenet_v2',
    ],
    'group2': [
        'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3',
    ],
    'group3': [
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'resnext50_32x4d', 'resnext101_32x8d',
        'wide_resnet50_2', 'wide_resnet101_2',
    ],
    'group4': [
        'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
    ],
    'group5': ['inception_v3'],
    'group6': ['googlenet'],
}

class Extractor(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained

        self.supported_models = SUPPORTED_MODELS

        if not self.model_name in self.get_supported_models():
            raise ValueError(f'{self.model_name} is unsupported extractor')
        self.__init_model()

    def __init_model(self):
        if self.model_name in ['inception_v3', 'googlenet']:
            self.model = getattr(torchvision.models, self.model_name)(pretrained=self.pretrained, init_weights=True)
        else:
            self.model = getattr(torchvision.models, self.model_name)(pretrained=self.pretrained)

    def __resnet_forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x

    def __shufflenet_forward(self, x):
        x = self.model.conv1(x)
        x = self.model.maxpool(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        x = self.model.stage4(x)
        x = self.model.conv5(x)
        return x

    def __inception_forward(self, x):
        x = self.model._transform_input(x)
        x = self.model.Conv2d_1a_3x3(x)
        x = self.model.Conv2d_2a_3x3(x)
        x = self.model.Conv2d_2b_3x3(x)
        x = self.model.maxpool1(x)
        x = self.model.Conv2d_3b_1x1(x)
        x = self.model.Conv2d_4a_3x3(x)
        x = self.model.maxpool2(x)
        x = self.model.Mixed_5b(x)
        x = self.model.Mixed_5c(x)
        x = self.model.Mixed_5d(x)
        x = self.model.Mixed_6a(x)
        x = self.model.Mixed_6b(x)
        x = self.model.Mixed_6c(x)
        x = self.model.Mixed_6d(x)
        x = self.model.Mixed_6e(x)
        x = self.model.Mixed_7a(x)
        x = self.model.Mixed_7b(x)
        x = self.model.Mixed_7c(x)
        return x

    def __googlenet_forward(self, x):
        x = self.model._transform_input(x)
        x = self.model.conv1(x)
        x = self.model.maxpool1(x)
        x = self.model.conv2(x)
        x = self.model.conv3(x)
        x = self.model.maxpool2(x)
        x = self.model.inception3a(x)
        x = self.model.inception3b(x)
        x = self.model.maxpool3(x)
        x = self.model.inception4a(x)
        x = self.model.inception4b(x)
        x = self.model.inception4c(x)
        x = self.model.inception4d(x)
        x = self.model.inception4e(x)
        x = self.model.maxpool4(x)
        x = self.model.inception5a(x)
        x = self.model.inception5b(x)
        return x

    def forward(self, x):
        if x.shape[1] == 1:
            x = torch.cat([x, x, x], axis=1)

        if self.model_name in self.supported_models['group1']:
            return self.model.features(x)
        elif self.model_name in self.supported_models['group2']:
            return self.model.layers(x)
        elif self.model_name in self.supported_models['group3']:
            return self.__resnet_forward(x)
        elif self.model_name in self.supported_models['group4']:
            return self.__shufflenet_forward(x)
        elif self.model_name in self.supported_models['group5']:
            return self.__inception_forward(x)
        elif self.model_name in self.supported_models['group6']:
            return self.__googlenet_forward(x)

    def get_supported_models(self):
        model_list = []
        for v in self.supported_models.values():
            model_list += v
        return model_list

if __name__ == '__main__':
    model_list = []
    for v in SUPPORTED_MODELS.values():
        model_list += v
    input_tensor = torch.rand(1, 1, 80, 80)
    for model_name in model_list:
        try:
            extractor = Extractor(model_name, pretrained=True)
            feats = extractor(input_tensor)
            print(f'{model_name.ljust(20)}{feats.shape}')
        except Exception as e:
            print(e)
