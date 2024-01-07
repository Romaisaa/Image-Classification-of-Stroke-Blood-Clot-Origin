from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
import os
import sys
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

import cv2

import random
class test_evaluation():
    def __init__(self):
        pass
    def label_with_threshold(self,output,threshold):
        '''This Function takes the output from softmax and return labels according to threshold
        Parameters:
        - output : output of softmax of the model (n,2) where n is number of samples
        - threshold : threshold applied for the first probability
        Returns:
        - labels : predicted classes after applying threshold 
        '''
        labels = []
        for pred in range(len(output)):
            if output[pred][1] >= threshold:
                labels.append(1)
            else:
                labels.append(0)
        return labels
    def calc_tpr_fpr(self,true_labels,predicted_labels):
        """
        Calculate True Positive Rate (TPR) and False Positive Rate (FPR).

        Parameters:
        - actual: List of actual class labels (0 or 1)
        - predicted: List of predicted class labels (0 or 1)

        Returns:
        - tpr: True Positive Rate
        - fpr: False Positive Rate
        """
        tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()

        # Calculate TPR and FPR
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        return tpr, fpr
    def weighted_log_loss_kaggle(self,probs, weights, target):
        """
        Calculate metric for mayo-clinic-strip-ai competition.

        Parameters:
        - probs : Prediction of models as two output of softmax
        - weights : Weights for each class in the metric
        - target : Actual labels

        Returns:
        - Competitiion Metric measure
        """
        probs = np.array(probs)
        log_probs = np.log(probs)
        weights = np.array(weights)

        target = np.array(target)
        res = 0
        for c in np.unique(target):
            class_log_probs = log_probs[target == c][:, c]
            class_weight = weights[c]
            res += class_weight * class_log_probs.mean()
        return - (res / weights.sum())
    def plot_conf_mat(self,true_labels,predicted_labels,title='Confusion Matrix'):
        """ Function to calculate confusion matrix and plot it
        Parameters:
        - true_labels : Actual labels of the images
        - predicted_labels : Predicted labels of the images
        - title : title of the plot and its name

        Returns:
        - png image of confusion matrix
        """
        cm = confusion_matrix(predicted_labels.detach().cpu().numpy(), true_labels)
        # Plot confusion matrix using seaborn
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(predicted_labels.detach().cpu().numpy()), yticklabels=np.unique(true_labels),annot_kws={"size": 35})
        plt.title(f'Confusion Matrix of {title}')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.savefig(f'Confusion_Matrix_{title}.png')
        plt.show()
    def plot_roc(self,prob,title='Roc Curve'):
        """ Function to calculate confusion matrix and plot it
        Parameters:
        - prob : Probability for the classes as array
        - title : title of the plot and its name

        Returns:
        - png image of ROC curve
        """
        thresholds = np.arange(0, 1.0, 0.001)
        tprs = []
        fprs = []
        for threshold in thresholds:
            new_labels = self.label_with_threshold(prob,threshold)
            tpr,fpr = self.calc_tpr_fpr(labels,new_labels)
            tprs.append(tpr)
            fprs.append(fpr)
        plt.plot(fprs, tprs, label='ROC Curve')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'ROC of {title}')
        roc_auc = auc(fprs, tprs)
        print(f"AUC: {roc_auc:.2f}")
        plt.legend()
        plt.savefig(f'ROC_Curve_{title}.png')
        plt.show()
    def calculate_metrics(y_true, y_predicted):
        # Confusion matrix
        tp = sum((true == 0 and pred == 0) for true, pred in zip(y_true, y_predicted))
        tn = sum((true == 1 and pred == 1) for true, pred in zip(y_true, y_predicted))
        fp = sum((true == 1 and pred == 0) for true, pred in zip(y_true, y_predicted))
        fn = sum((true == 0 and pred == 1) for true, pred in zip(y_true, y_predicted))

        # Precision
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0

        # Recall
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0

        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0

        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

        return precision, recall, f1, accuracy, specificity