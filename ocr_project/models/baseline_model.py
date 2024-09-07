import torch
import torch.nn as nn
import torchvision.models as models


class BaselineModel(nn.Module):
    def __init__(self, num_classes=77):
        super(BaselineModel, self).__init__()
        
        # Load EfficientNet_b1 model
        self.backbone = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT).features
        
        # Remove the last 30% of the layers
        num_layers = len(self.backbone)
        num_layers_to_remove = int(num_layers * 0.3)
        self.backbone = self.backbone[:num_layers - num_layers_to_remove]

        # reduce strides
        self.reduce_strides()
        
        # Freeze the first 50% of the remaining layers
        num_layers_after_removal = len(self.backbone)
        num_layers_to_freeze = int(num_layers_after_removal * 0.5)
        self.backbone[:num_layers_to_freeze].requires_grad_(False)

        self.lstm = nn.LSTM(input_size=192, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout1d(0.3)
        self.log_softmax  = nn.LogSoftmax(dim=2)

    def reduce_strides(self):
        self.backbone[0][0].stride = (2, 1)

        self.backbone[3][0].block[1][0].stride = (2, 1)
        self.backbone[3][0].block[1][0].padding = (2, 1)

        self.backbone[6][0].block[1][0].stride = (2, 1)
        self.backbone[6][0].block[1][0].padding = (2, 1)

    def forward(self, x):
        out = self.backbone(x)
        out = nn.MaxPool2d((out.shape[2], 1))(out)

        out = out.permute(0, 3, 2, 1)
        out = torch.reshape(out, (out.shape[0], out.shape[1], -1))

        out, _ = self.lstm(out)

        out = out.permute(0, 2, 1) # change chape to (N, C, L) for dropout
        out = self.dropout(out)
        out = out.permute(0, 2, 1) # change chape back to (N, L, C)

        out = self.fc(out)
        out = self.log_softmax(out)
        return out
