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

class ConvolutionalNeuralNetwork(nn.Module):
  def __init__(self, num_channels=16, sample_rate=22050, n_fft=1024, f_min=0.0, f_max=11025.0, num_mels=128, num_classes=10):
    super(ConvolutionalNeuralNetwork, self).__init__()
    self.melspec = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, f_min=f_min, f_max=f_max, n_mels=num_mels)
    self.amplitude_to_db = T.AmplitudeToDB()
    self.input_batch_norm = nn.BatchNorm2d(1)

    # convolutional layers
    self.layer1 = Conv_2D(1, num_channels, pooling=(2, 3))
    self.layer2 = Conv_2D(num_channels, num_channels, pooling=(3, 4))
    self.layer3 = Conv_2D(num_channels, num_channels * 2, pooling=(2, 5))
    self.layer4 = Conv_2D(num_channels * 2, num_channels * 2, pooling=(3, 3))
    self.layer5 = Conv_2D(num_channels * 2, num_channels * 4, pooling=(3, 4))

    # dense layers
    self.dense1 = nn.Linear(num_channels * 4, num_channels * 4)
    self.dense_bn = nn.BatchNorm1d(num_channels * 4)
    self.dense2 = nn.Linear(num_channels * 4, num_classes)
    self.dropout = nn.Dropout(0.5)
    self.relu = nn.ReLU()

  def forward(self, wav):
    # input Preprocessing
    out = self.melspec(wav)
    out = self.amplitude_to_db(out)

    # input batch normalization
    out = out.unsqueeze(1)
    out = self.input_batch_norm(out)

    # convolutional layers
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.layer5(out)

    # reshape. (batch_size, num_channels, 1, 1) -> (batch_size, num_channels)
    out = out.reshape(len(out), -1)

    # dense layers
    out = self.dense1(out)
    out = self.dense_bn(out)
    out = self.relu(out)
    out = self.dropout(out)
    out = self.dense2(out)

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
    self.effnet_v2s.classifier = nn.Linear(self.effnet_v2s.classifier.in_features, num_classes)
    # Updating the Weights and Bias of the last layer
    params_to_update = []
    for _, param in self.effnet_v2s.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            
    # Define the Loss and Optimizer Functions
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(params_to_update, lr=0.001)
    
  def forward(self, x):
    return self.effnet_v2s(x)