import torch.nn as nn


class AutoEncoder2(nn.Module):
    def __init__(self, ipt_size, opt_size):
        super(AutoEncoder2, self).__init__()

        # 编码器层
        self.encoder = nn.Sequential(
            nn.Linear(ipt_size, opt_size),
            nn.ReLU(),
            # nn.Linear(1024, 128),
            # nn.ReLU(),
            #nn.Linear(128, opt_size),
            #nn.ReLU(),
        )

        # 解码器层
        self.decoder = nn.Sequential(
            nn.Linear(opt_size, ipt_size),
            #nn.ReLU(),
            # nn.Linear(128, 1024),
            # nn.ReLU(),
            #nn.Linear(128, ipt_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
