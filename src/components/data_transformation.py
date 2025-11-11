import sys 
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprcessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()

    def initiate_data_transformation (self, X_train, X_test, y_train, y_test):
        logging.info("Preprocessing is started.")
        logging.info("Features order before preprocessing.")
        logging.info(list(X_train.columns))

        try:
            num_features = X_train.select_dtypes(exclude=["object", "category"]).columns
            cat_features = X_train.select_dtypes(include=["object", "category"]).columns

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()

            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, cat_features),
                     ("StandardScaler", numeric_transformer, num_features),        
                ]
            )

            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            save_object(
                file_path=self.data_tranformation_config.preprcessor_obj_file_path,
                obj=preprocessor
            )

            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

            return(X_train, X_test, y_train, y_test)
        
        except Exception as e:
            raise CustomException(e, sys)


    def clean_engineer_features(self, raw_data_path):
        try:
            df = raw_data_path.dropna()

            logging.info("AgeGroup feature creation is started.")
            bins = [0, 18, 30, 45, 60, 100] 
            labels = ['0–18', '19–30', '31–45', '46–60', '60+']

            df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

            logging.info("FlightDistanceGroup feature creation is started.")
            bins = [0, 1000, 3000, 5000]
            labels = ['Short', 'Medium', 'Long']

            df['FlightDistanceGroup'] = pd.cut(df['Flight Distance'], bins=bins, labels=labels)

            return df


        except Exception as e:
            raise CustomException(e, sys)