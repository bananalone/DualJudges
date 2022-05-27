import torch.nn as nn
import torch.nn.functional as F


class PFeedback(nn.Module):
    
    def __init__(self, model1: nn.Module, model2: nn.Module):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        
    def forward(self, input):
        out1 = self.model1(input)
        out2 = self.model2(input)
        return out1, out2


class PFeedback2(nn.Module):

    def __init__(self, in_channels, out_channels, out_size):
        """
        @in_channels encodings 的维度
        @out_channels decodings 的维度
        @out_size 上采样的大小
        """
        super().__init__()
        channels = [in_channels, in_channels // 2, in_channels // 4, in_channels // 8, in_channels // 16]
        modules = []
        for i in range(len(channels) - 1):
            modules += [
                nn.ConvTranspose2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(num_features=channels[i+1]),
                nn.ReLU(),
            ]
        modules.append(nn.ConvTranspose2d(in_channels=channels[-1], out_channels=out_channels, kernel_size=4, stride=2, padding=1))
        self.decoder = nn.Sequential(*modules)
        self.out_size = out_size

    def forward(self, encodings, decodings):
        """
        @encodings 编码器输出 例如 encodings = torch.rand(4, 2048, 8, 8)
        @decodings 解码器输出 例如 decodings = torch.rand(4, 10, 256, 256)
        """
        out = self.decoder(encodings)
        out = F.upsample(out, size=self.out_size, mode="bilinear", align_corners=True)
        coarsePred = out
        finePred = decodings
        return finePred, coarsePred