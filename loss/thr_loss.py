import torch
from torch import nn


class ThresholdScore(nn.Module):

    def __init__(self, T = 1e-4, log = False):
        super(ThresholdScore,self).__init__()
        self.epsilon = T
        self.log = log

    def forward(self, outputs,):
        if self.log:
            proba_values= torch.softmax(outputs, dim=-1)
        else:
            proba_values= torch.log_softmax(outputs, dim=-1)
        return proba_values
    
    def nonconformity(self, score_inputs, targets):
        pred_probs = torch.gather(score_inputs, -1, targets[..., None])[...,0]
        return 1 - pred_probs

    def conformal_probability(self, score_inputs, threshold):
        scores = 1 - score_inputs  
        membership_logits = scores.unsqueeze(2) - threshold.unsqueeze(0).unsqueeze(0)
        membership_scores = torch.sigmoid(-membership_logits / self.epsilon)
        membership_scores = torch.mean(membership_scores,dim = -1)
        return membership_scores
    