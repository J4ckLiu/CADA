import torch.nn as nn
'''
Save the logits in advance and use the DirectModel
to ensure compatibility with the existing code framework.
'''
class DirectModel(nn.Module):
  def __init__(self):
    super(DirectModel, self).__init__()
    
  def forward(self, logits):
    return logits