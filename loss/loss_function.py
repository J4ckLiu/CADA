import torch.nn as nn
import torch.nn.functional as F
import torch
class SDLoss(nn.Module):

    def __init__(self, score_function):
        super(SDLoss, self).__init__()
        self.score_function = score_function

    def forward(self, logits, labels):
        scores = self.score_function(logits)
        true_scores = self.score_function.nonconformity(scores, labels)
        probability_sum = self.score_function.conformal_probability(scores, true_scores)
        row_sums = probability_sum.mean(dim=1)
        return row_sums.mean()
    


    