import torch.nn as nn


class AE(nn.Module):
    def __init__(self, ipt_size, opt_size):
        super(AE, self).__init__()

        # 编码器层
        self.encoder = nn.Sequential(
            nn.Linear(ipt_size, opt_size),
            nn.ReLU(),
        )

        # 解码器层
        self.decoder = nn.Sequential(
            nn.Linear(opt_size, ipt_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
