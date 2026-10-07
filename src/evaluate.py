import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def topk(y, s, k):
    y = np.asarray(y)
    n = int(len(y) * k)
    idx = np.argsort(-np.asarray(s))[:n]
    precision = y[idx].mean()
    recall = y[idx].sum() / y.sum()
    lift = precision / y.mean()
    return round(float(precision), 3), round(float(recall), 3), round(float(lift), 2)

def report(name, y, s):
    print(name)
    print("  PR-AUC :", round(average_precision_score(y, s), 3), "(base rate", round(float(np.mean(y)), 3), ")")
    print(" ROC=AUC", round(roc_auc_score(y,s), 3))
    print(" top 10% (precision, recall, lift):", topk(y, s, 0.10))
    print(" top 20% (precision, recall, lift):", topk(y, s, 0.20))
    