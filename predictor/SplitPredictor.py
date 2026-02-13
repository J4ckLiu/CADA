import warnings
import math
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.utils import split_logits_labels, build_score
from utils.metric import Metrics


class Predictor(nn.Module):
    def __init__(self, model, conformal, alpha, device):
        super(Predictor, self).__init__()
        self.model = model
        self.model.eval()
        self.score_function = build_score(conformal)
        self.alpha = alpha
        self.num_classes = 1000
        self.metric = Metrics()
        self.device = device

    def calibrate(self, calibloader):
        logits, labels = split_logits_labels(self.model, calibloader, self.device)
        self.calculate_threshold(logits, labels)
        return logits, labels

    def calibrate_with_logits_labels(self, logits, labels):
        self.calculate_threshold(logits, labels)

    def calculate_threshold(self, logits, labels):
        alpha = self.alpha
        if alpha >= 1 or alpha <= 0:
            raise ValueError("Significance level 'alpha' must be in (0,1).")
        logits = logits.to(self.device)
        labels = labels.to(self.device)
        scores = self.score_function(logits, labels) 
        self.q_hat = self.calculate_conformal_value(scores, alpha)


    def calculate_conformal_value(self, scores, alpha):
        if len(scores) == 0:
            warnings.warn(
                "The number of scores is 0, which is a invalid scores. To avoid program crash, the threshold is "
                "set as torch.inf.")
            return torch.inf
        qunatile_value = math.ceil((scores.shape[0] + 1) * (1 - alpha)) / scores.shape[0]

        if qunatile_value > 1:
            warnings.warn(
                "The value of quantile exceeds 1. It should be a value in (0,1). To avoid program crash, the threshold "
                "is set as torch.inf.")
            return torch.inf

        return torch.quantile(scores, qunatile_value, interpolation="higher").to(self.device)

    def predict(self, x_batch):
        output = self.model(x_batch.to(self.device))
        tmp_logits = output.float()
        sets = self.predict_with_logits(tmp_logits)
        return sets

    def predict_with_logits(self, logits, q_hat=None):
        scores = self.score_function(logits).to(self.device)
        if q_hat is None:
            S = self.generate_prediction_set(scores, self.q_hat)
        else:
            S = self.generate_prediction_set(scores, q_hat)
        return S

    def evaluate(self, val_dataloader, alpha, num_classes):
        prediction_sets = []
        probs_list = []
        labels_list = []
        with torch.no_grad():
            for examples in tqdm(val_dataloader):
                tmp_x, tmp_label = examples[0].to(self.device), examples[1].to(self.device)
                prediction_sets_batch = self.predict(tmp_x)   
                prediction_sets.extend(prediction_sets_batch)
                output= self.model(tmp_x)
                tmp_probs = (output).detach()    
                probs_list.append(tmp_probs)
                labels_list.append(tmp_label)
        val_probs = torch.cat(probs_list)
        val_labels = torch.cat(labels_list)
        res_dict = {"Coverage": round(self.metric('coverage_rate')(prediction_sets, val_labels),3),
                    "Size": round(self.metric('average_size')(prediction_sets),3),
                    "CovGap": round(self.metric('CovGap')(prediction_sets, val_labels, alpha, num_classes),3)}       
        return res_dict
    
    def generate_prediction_set(self, scores, q_hat):
        if len(scores.shape) == 1:
            return torch.argwhere(scores <= q_hat).reshape(-1).tolist()
        else:
            return [torch.argwhere(scores[i] <= q_hat).reshape(-1).tolist() for i in range(scores.shape[0])]
