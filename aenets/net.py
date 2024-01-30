import torch.nn as nn
import torch.nn.functional as F
import torch


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


class AE_test(nn.Module):
    def __init__(self, ipt_size, opt_size):
        super(AE_test, self).__init__()

        # 编码器层
        self.encoder = nn.Sequential(
            nn.Linear(ipt_size, opt_size),
            nn.ReLU(),
            nn.Linear(opt_size, opt_size),
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



class AE_sw(nn.Module):
    def __init__(self, ipt_size, opt_size):
        super(AE_sw, self).__init__()

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
        self.encoder[0].weight = nn.Parameter(torch.transpose(self.decoder[0].weight, 0, 1))
        return x


class AE2layers(nn.Module):
    def __init__(self, ipt_size, opt_size):
        super(AE2layers, self).__init__()

        # 编码器层
        self.encoder = nn.Sequential(
            nn.Linear(ipt_size, int((ipt_size + opt_size) / 2.0)),
            nn.ReLU(),
            nn.Linear(int((ipt_size + opt_size) / 2.0), opt_size),
            nn.ReLU(),
        )

        # 解码器层
        self.decoder = nn.Sequential(
            nn.Linear(opt_size, int((ipt_size + opt_size) / 2.0)),
            nn.ReLU(),
            nn.Linear(int((ipt_size + opt_size) / 2.0), ipt_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
