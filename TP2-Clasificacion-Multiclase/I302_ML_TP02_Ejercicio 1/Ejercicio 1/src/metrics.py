import pandas as pd
import numpy as np

def confusion_matrix(y_true, y_pred, labels):
    """
    Compute the confusion matrix.
    """
    n_labels = len(labels)
    cm = np.zeros((n_labels, n_labels), dtype=int)
    label_to_index = {label: index for index, label in enumerate(labels)}

    for true, pred in zip(y_true, y_pred):
        if true in label_to_index and pred in label_to_index:
            cm[label_to_index[true], label_to_index[pred]] += 1

    return cm

def accuracy(y_true, y_pred):
    """
    Compute the accuracy score.
    """
    return np.sum(y_true == y_pred) / len(y_true)

def precision(y_true, y_pred, labels):
    """
    Compute the precision score for each class.
    """
    cm = confusion_matrix(y_true, y_pred, labels)
    precisions = {}
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        precisions[label] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return precisions

def recall(y_true, y_pred, labels):
    """
    Compute the recall score for each class.
    """
    cm = confusion_matrix(y_true, y_pred, labels)
    recalls = {}
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        recalls[label] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return recalls

def f1_score(y_true, y_pred, labels):
    """
    Compute the F1 score for each class.
    """
    precisions = precision(y_true, y_pred, labels)
    recalls = recall(y_true, y_pred, labels)
    f1_scores = {}
    for label in labels:
        p = precisions[label]
        r = recalls[label]
        f1_scores[label] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    return f1_scores

def PR_curve(y_true, y_pred, labels):
    """
    Compute precision-recall curve data for each class.
    """
    from sklearn.metrics import precision_recall_curve

    pr_curves = {}
    for i, label in enumerate(labels):
        y_true_binary = (y_true == label).astype(int)
        y_pred_scores = (y_pred == label).astype(int)  # Assuming y_pred are binary scores
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_scores)
        pr_curves[label] = (precision, recall)
    
    return pr_curves

def ROC_curve(y_true, y_pred, labels):
    """
    Compute ROC curve data for each class.
    """
    from sklearn.metrics import roc_curve

    roc_curves = {}
    for i, label in enumerate(labels):
        y_true_binary = (y_true == label).astype(int)
        y_pred_scores = (y_pred == label).astype(int)  # Assuming y_pred are binary scores
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_scores)
        roc_curves[label] = (fpr, tpr)
    
    return roc_curves

def AUC_ROC_curve(y_true, y_pred, labels):
    """
    Compute AUC-ROC for each class.
    """
    from sklearn.metrics import roc_auc_score

    auc_scores = {}
    for i, label in enumerate(labels):
        y_true_binary = (y_true == label).astype(int)
        y_pred_scores = (y_pred == label).astype(int)  # Assuming y_pred are binary scores
        auc_scores[label] = roc_auc_score(y_true_binary, y_pred_scores)
    
    return auc_scores

def AUC_PR_curve(y_true, y_pred, labels):
    """
    Compute AUC-PR for each class.
    """
    from sklearn.metrics import average_precision_score

    auc_pr_scores = {}
    for i, label in enumerate(labels):
        y_true_binary = (y_true == label).astype(int)
        y_pred_scores = (y_pred == label).astype(int)  # Assuming y_pred are binary scores
        auc_pr_scores[label] = average_precision_score(y_true_binary, y_pred_scores)
    
    return auc_pr_scores

