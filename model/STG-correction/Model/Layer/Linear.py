# Import Libs
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, Text_InFeature, Text_OutFeature):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=Text_InFeature, out_features=Text_OutFeature)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return x