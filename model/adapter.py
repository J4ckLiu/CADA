# adapted from intra order-preserving function
# @inproceedings{rahimi2020intra,
#   title={Intra Order-preserving Functions for Calibration of Multi-Class Neural Networks},
#   author={Rahimi, Amir and Shaban, Amirreza and Cheng, Ching-An and Hartley, Richard and Boots, Byron},
#   booktitle={Advances in Neural Information Processing Systems},
#   year={2020}
# }

import torch.nn as nn
import torch
import torch.nn.functional as F

class FCModel(nn.Module):
    def __init__(self, num_hiddens,  num_classes = 1000,
                      batch_norm=False):
        super(FCModel, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_classes = num_classes
        last_hidden = num_classes
        calib_layers = []
        for num_hidden in num_hiddens:
          if batch_norm:
            calib_layers.append(nn.BatchNorm1d(last_hidden))
          calib_layers.append(nn.Linear(last_hidden, num_hidden, bias=True))
          calib_layers.append(nn.ReLU())
          last_hidden = num_hidden
        if len(calib_layers) > 0:
          self.calib_layers = nn.Sequential(*calib_layers)
        else:
          self.calib_layers = lambda x: x
        if batch_norm:
          self.bn = nn.BatchNorm1d(num_hiddens[-1])
        else:
          self.bn = lambda x: x
        if len(num_hiddens) > 0:
          self.fc = nn.Linear(num_hiddens[-1], num_classes)
        else:
          self.fc = nn.Linear(num_classes, num_classes)
      
    def forward(self, logits):
        out = self.calib_layers(logits)
        out = self.fc(self.bn(out))
        return out
    
class OrderPreservingModel(nn.Module):
    def __init__(self, base_classifier, num_classes, m_activation=F.sigmoid):
        super(OrderPreservingModel, self).__init__()
        self.num_classes = num_classes
        self.base_model = FCModel(num_hiddens=[], num_classes = self.num_classes)
        self.classifier = base_classifier
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.num_classes = num_classes 
        self.m_activation = m_activation


    def compute_u(self, sorted_logits):
        diffs = sorted_logits[:,:-1] - sorted_logits[:,1:]
        diffs = torch.cat((diffs, torch.ones((diffs.shape[0],1),
                                              dtype=diffs.dtype,
                                              device=diffs.device)), dim=1)
        diffs = (diffs != 0).float()
        return diffs.flip([1])
  
    def forward(self, x):
        ori_logits= self.classifier(x)
        logits = F.softmax(ori_logits, dim=1)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        _, unsorted_indices = torch.sort(sorted_indices, descending=False)
        u = self.compute_u(sorted_logits)
        m = self.base_model(logits)
        m[:,1:] = self.m_activation(m[:,1:].clone())
        m[:,0] = 0
        um = torch.cumsum(u*m,1).flip([1])
        out = torch.gather(um,1,unsorted_indices)
        out = out + ori_logits
        return out
      
    