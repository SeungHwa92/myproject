import torch
from torch import nn
import encoding


class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.fc = nn.Linear(1000, 1000)
        self.bn = encoding.nn.SyncBatchNorm(1000)
        #self.bn = BN(10)
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)

        return x

net = network()

net.cuda()
net = nn.DataParallel(net)


import numpy as np
a = np.random.random(1000000).reshape((1000, 1000))
a = torch.Tensor(a)

for i in range(100000):
    out = net(a)

print(out)
