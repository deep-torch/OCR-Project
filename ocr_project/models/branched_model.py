import torch
import torch.nn as nn
import torchvision.models as models

from ocr_project.models.branch import Branch


class BranchedModel(nn.Module):
    def __init__(self, num_classes=77):
        super(BranchedModel, self).__init__()
        
        # Trimmed EfficientNet Backbone with modified strides
        self.backbone = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT).features[:5]
        self.reduce_strides()
        backbone_out = 80

        branches_configs = [
            {
                "channels": [backbone_out],
                "strides": []
            },
            {
                "channels": [backbone_out, 128],
                "strides": [(2,1)]
            },
            {
                "channels": [backbone_out, 128, 256],
                "strides": [(2,1), (2, 1)]
            }
        ]

        self.branches = nn.ModuleList()
        for config in branches_configs:
            self.branches.append(
                Branch(
                  channels=config["channels"],
                  strides=config["strides"]
                )
            )
        
        self.lstm = nn.LSTM(input_size=768, hidden_size=256, num_layers=3, bidirectional=True, batch_first=True)

        self.temporal_channels_dropout = nn.Dropout1d(0.3)
        
        self.final_projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
        self.log_softmax  = nn.LogSoftmax(dim=2)

    def reduce_strides(self):
        self.backbone[0][0].stride = (1, 2)
        self.backbone[2][0].block[1][0].stride = (2, 1)
        self.backbone[3][0].block[1][0].stride = (2, 2)
        self.backbone[4][0].block[1][0].stride = (2, 1)

    def forward(self, x):
        features = self.backbone(x)

        branches_outputs = []
        for branch in self.branches:
            branches_outputs.append(branch(features))

        multi_scope_features = torch.cat(branches_outputs, dim=2)

        temporal_features, _ = self.lstm(multi_scope_features)

        # change shapes to apply dropout on features axis
        temporal_features = temporal_features.permute(0, 2, 1)
        temporal_features = self.temporal_channels_dropout(temporal_features)
        temporal_features = temporal_features.permute(0, 2, 1)

        logits = self.final_projector(temporal_features)
        log_probs = self.log_softmax(logits)

        return log_probs
