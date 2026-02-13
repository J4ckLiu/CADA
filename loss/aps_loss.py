import torch
from torch import nn
from utils.neural_sort import neural_sort, soft_quantile



class AdaptiveScoreFunction(nn.Module):

    def __init__(self, epsilon: float = 0.1, log = False) -> None:
        super().__init__()
        self.epsilon = epsilon 
        self.log = log

    def forward(self, outputs):
        if self.log:
            proba_values= torch.log_softmax(outputs, dim=-1)
        else:
            proba_values= torch.softmax(outputs, dim=-1)
        n, K = proba_values.shape
        proba_values = proba_values + 1e-6*torch.rand(proba_values.shape,dtype = proba_values.dtype).cuda()
        proba_values = proba_values / torch.sum(proba_values,1)[:,None]
        return proba_values
    
    def nonconformity(self, score_inputs, targets):
        J = score_inputs.shape[-1]
        permutation = neural_sort(score_inputs, self.epsilon)
        sorted_probs = (permutation @ score_inputs[...,None])[...,0]
        
        cumulative_dist = torch.cumsum(sorted_probs, -1)
        scores = (permutation.transpose(-2,-1) @ cumulative_dist[...,None])[...,0]
        return torch.gather(scores, -1, targets[..., None])[...,0]
    
    def conformal_probability(self, score_inputs, threshold):
        J = score_inputs.shape[-1]
        permutation = neural_sort(score_inputs, self.epsilon)
        sorted_probs = (permutation @ score_inputs[...,None])[...,0]
        
        cumulative_dist = torch.cumsum(sorted_probs, -1)
        scores = (permutation.transpose(-2,-1) @ cumulative_dist[...,None])[...,0]

        membership_logits = scores.unsqueeze(2) - threshold.unsqueeze(0).unsqueeze(0)
        membership_scores = torch.sigmoid(-membership_logits / self.epsilon)
        membership_scores = torch.mean(membership_scores,dim = -1)
        return membership_scores
    


    