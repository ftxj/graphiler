import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()

    def forward(self, x, h):
        new_h = torch.tanh(x + h)
        return new_h, new_h

my_cell = MyCell()


model = torch.jit.script(my_cell) 

print(model.inlined_graph)

model.save("simple_model.pt")

exit()

x = torch.rand(3, 4)
h = torch.rand(3, 4)


print(my_cell(x, h))