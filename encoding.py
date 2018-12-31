import torch
from torch import nn
import encoding

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.fc = nn.Linear(10, 10)
        self.bn = encoding.nn.BatchNorm1d(10)
        #self.bn = nn.BatchNorm1d(10)
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)

        return x

net = network()

net.cuda()
net = nn.DataParallel(net)
encoding.parallel.patch_replication_callback(net)

import numpy as np
a = np.random.random(100).reshape((10, 10))
a = torch.Tensor(a)
out = net(a)

print(out)