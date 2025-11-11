import os 
import sys
from dataclasses import dataclass

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model


@dataclass

class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        logging.info("Model training is initiated.")

        try:
            model = xgb.XGBClassifier(
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1
            )

            param_dist = {
                'n_estimators': [100, 200, 300, 400],
                'max_depth': [3, 4, 5, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 0.1, 0.3, 0.5],
                'min_child_weight': [1, 3, 5]
            }

            scoring = {
                'accuracy': 'accuracy',
                'precision': 'precision',
                'recall': 'recall',
                'f1': 'f1',
                'roc_auc': 'roc_auc'
            }

            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dist,
                n_iter=25,
                cv=kf,
                scoring=scoring,   
                refit='f1',        
                verbose=2,
                random_state=42,
                n_jobs=-1
            )

            random_search.fit(X_train, y_train)

            best_xgb = random_search.best_estimator_
            y_pred = best_xgb.predict(X_test)
            y_proba = best_xgb.predict_proba(X_test)[:, 1]  

            evaluate_model(y_test, y_pred, y_proba)
            

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_xgb
            )
            

        except Exception as e:
            raise CustomException(e, sys)