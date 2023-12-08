import torch.nn as nn
import torchaudio.transforms as T
import torchvision

class Conv_2D(nn.Module):
  def __init__(self, input_channels, output_channels, shape=3, pooling=(2, 2), dropout=0.1):
    super(Conv_2D, self).__init__()
    self.convolution = nn.Conv2d(input_channels, output_channels, shape, padding=shape//2)
    self.batch_norm = nn.BatchNorm2d(output_channels)
    self.leaky_ReLU = nn.LeakyReLU()
    self.max_pool = nn.MaxPool2d(pooling)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = self.convolution(x)
    x = self.batch_norm(x)
    x = self.leaky_ReLU(x)
    x = self.max_pool(x)
    out = self.dropout(x)
    return out

class SpectrogramCNN(nn.Module):
  def __init__(self, num_channels=16, num_classes=10):
    super().__init__()
    self.input_batch_norm = nn.BatchNorm2d(3)

    # convolutional layers
    self.layer1 = Conv_2D(3, num_channels, pooling=(3, 2))
    self.layer2 = Conv_2D(num_channels, num_channels, pooling=(4, 3))
    self.layer3 = Conv_2D(num_channels, num_channels * 2, pooling=(5, 4))
    self.layer4 = Conv_2D(num_channels * 2, num_channels * 2, pooling=(2, 3))
    self.layer5 = Conv_2D(num_channels * 2, num_channels * 4, pooling=(2, 4))

    # dense layers
    self.dense1 = nn.Linear(num_channels * 4, num_channels * 4)
    self.dense_bn = nn.BatchNorm1d(num_channels * 4)
    self.dense2 = nn.Linear(num_channels * 4, num_classes)
    self.dropout = nn.Dropout(0.5)
    self.relu = nn.ReLU()

  def forward(self, spectrogram):
    # input batch normalization
    out = spectrogram.float()
    out = self.input_batch_norm(out)
    print(out.shape)
    # convolutional layers
    out = self.layer1(out)
    print(out.shape)
    out = self.layer2(out)
    print(out.shape)
    out = self.layer3(out)
    print(out.shape)
    out = self.layer4(out)
    print(out.shape)
    out = self.layer5(out)
    print(out.shape)

    # reshape. (batch_size, num_channels, 1, 1) -> (batch_size, num_channels)
    out = out.reshape(len(out), -1)
    print(out.shape)

    # dense layers
    out = self.dense1(out)
    print(out.shape)
    out = self.dense_bn(out)
    print(out.shape)
    out = self.relu(out)
    print(out.shape)
    out = self.dropout(out)
    print(out.shape)
    out = self.dense2(out)
    print(out.shape)

    return out
  
class ResNet18(nn.Module):
  def __init__(self, num_classes=10):
    super().__init__()
    # Load the pretrained (on ImageNet) ResNet-18 model
    self.resnet18 = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    # Fix the trainable parameters
    for parameter in self.resnet18.parameters():
      parameter.requires_grad = False
    # Replacing the Last Fully Connected Layer for Transfer Learning
    self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)
    # Updating the Weights and Bias of the last layer
    params_to_update = []
    for _, param in self.resnet18.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            
    # Define the Loss and Optimizer Functions
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(params_to_update, lr=0.001)

  def forward(self, x):
    x = x.float()
    return self.resnet18(x)
  
class EfficientNetV2S(nn.Module):
  def __init__(self, num_classes = 10) -> None:
    super().__init__()
    # Load the pretrained (on ImageNet) EfficentNetV2-S model
    self.effnet_v2s = torchvision.models.efficientnet_v2_s(weights="IMAGENET1K_V1")
    # Fix the trainable parameters
    for parameter in self.effnet_v2s.parameters():
      parameter.requires_grad = False
    # Replacing the Last Fully Connected Layer for Transfer Learning
    self.effnet_v2s.classifier[1] = nn.Linear(self.effnet_v2s.classifier[1].in_features, num_classes)
    # Updating the Weights and Bias of the last layer
    params_to_update = []
    for _, param in self.effnet_v2s.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            
    # Define the Loss and Optimizer Functions
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(params_to_update, lr=0.001)
    
  def forward(self, x):
    x = x.float()
    return self.effnet_v2s(x)