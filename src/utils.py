import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)


    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(y_test, y_pred, y_proba):

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    logging.info("Model Evaluation Metrics:")
    logging.info(f"Accuracy : {acc:.4f}")
    logging.info(f"Precision: {prec:.4f}")
    logging.info(f"Recall   : {rec:.4f}")
    logging.info(f"F1-score : {f1:.4f}")
    logging.info(f"ROC-AUC  : {roc_auc:.4f}")

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)