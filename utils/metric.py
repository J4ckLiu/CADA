import numpy as np
from typing import Any
from .utils import Registry

METRICS_REGISTRY = Registry("METRICS")

@METRICS_REGISTRY.register()
def coverage_rate(prediction_sets, labels):
    labels = np.array(labels.cpu())
    coverage = np.array([np.isin(label, prediction_set) for label, prediction_set in zip(labels, prediction_sets)])
    return float(np.mean(coverage))

@METRICS_REGISTRY.register()
def average_size(prediction_sets):
    total_size = sum(map(len, prediction_sets))  
    avg_size = total_size / len(prediction_sets)  
    return avg_size

@METRICS_REGISTRY.register()
def CovGap(prediction_sets, labels, alpha, num_classes):
    if len(prediction_sets) == 0:
        return (1 - alpha) * 100
    labels = labels.cpu()
    rate_classes = []
    for k in range(num_classes):
        idx = np.where(labels == k)[0]
        selected_preds = [prediction_sets[i] for i in idx]
        if len(labels[labels == k]) != 0:
            rate_classes.append(coverage_rate(selected_preds, labels[labels == k]))
    rate_classes = np.array(rate_classes)
    return float(np.mean(np.abs(rate_classes - (1 - alpha))) * 100)

'''
@METRICS_REGISTRY.register()
def SSCV(prediction_sets, labels, alpha, stratified_size=[[0, 1], [2, 2], [3,3],[4, 4],[5, 5],[6,6],[7,7] ,[8,8] ,[9,9],[10,10], [11, 100], [101, 1000]]):
    """
    SSCV seems not a good metric according to recent public comment
    it varies dramastically with pre-defined stratified_size, dataset and error rate, 
    """
    labels = labels.cpu()
    size_array = np.zeros(len(labels))
    correct_array = np.zeros(len(labels))
    for index, ele in enumerate(prediction_sets):
        size_array[index] = len(ele)
        correct_array[index] = 1 if labels[index] in ele else 0
    sscv = -1
    for stratum in stratified_size:
        temp_index = np.argwhere((size_array >= stratum[0]) & (size_array <= stratum[1]))
        if len(temp_index) > (len(labels)/200):
            stratum_violation = abs((1 - alpha) - np.mean(correct_array[temp_index]))
            sscv = max(sscv, stratum_violation)
    return float(sscv)*100
'''

class Metrics:
    def __call__(self, metric) -> Any:
        if metric not in METRICS_REGISTRY.registered_names():
            raise NameError(f"The metric: {metric} is not defined in DeepCP.")
        return METRICS_REGISTRY.get(metric)

class DimensionError(Exception):
    pass
