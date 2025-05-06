import timm
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EEncoderEfficientNetB4(nn.Module):
    def __init__(self, image_feature_size=14):
        super().__init__()
        self.image_feature_size = image_feature_size

        effnet = timm.create_model('efficientnet_b4', pretrained=True)

        self.effnet = nn.Sequential(*list(effnet.children())[:-2])

        self.feature_dim = effnet.num_features
        self.pool = nn.AdaptiveAvgPool2d((image_feature_size, image_feature_size))
        self.finetune(enabled=False)

    def forward(self, x):
        out = self.effnet(x)
        out = self.pool(out)
        print(out)
        return out

    def finetune(self, enabled=True):
        for param in self.effnet.parameters():
            param.requires_grad = False

        if enabled:
            last_blocks = list(self.effnet.children())[-3:]
            for block in last_blocks:
                for param in block.parameters():
                    param.requires_grad = True

buna = EEncoderEfficientNetB4()
buna.forward()
