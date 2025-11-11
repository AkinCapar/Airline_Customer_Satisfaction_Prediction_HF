import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer


@dataclass 
class DataIngestionConfig:
    X_train_data_path: str=os.path.join("artifacts", "X_train.csv")
    y_train_data_path: str=os.path.join("artifacts", "y_train.csv")
    X_test_data_path: str=os.path.join("artifacts", "X_test.csv")
    y_test_data_path: str=os.path.join("artifacts", "y_test.csv")
    raw_data_path: str=os.path.join("artifacts", "raw.csv")
    cleaned__feature_engineered_data_path: str=os.path.join("artifacts", "cleaned_engineered.csv")

class DataIngestion: 
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion is started.")
        try: 
            data_transformation = DataTransformation()
            df1 = pd.read_csv("notebooks/data/train.csv")
            df2 = pd.read_csv("notebooks/data/test.csv")
            df = df = pd.concat([df1, df2], ignore_index=True)

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv (self.ingestion_config.raw_data_path, index=False, header=True)

            cleaned_feature_engineered_data = data_transformation.clean_engineer_features(df)
            cleaned_feature_engineered_data.to_csv(self.ingestion_config.cleaned__feature_engineered_data_path, index=False, header=True)

            X_train, X_test, y_train, y_test = self.train_test_split(cleaned_feature_engineered_data)
            X_train.to_csv(self.ingestion_config.X_train_data_path, index=False, header=True)
            X_test.to_csv(self.ingestion_config.X_test_data_path, index=False, header=True)
            y_train.to_csv(self.ingestion_config.y_train_data_path, index=False, header=True)
            y_test.to_csv(self.ingestion_config.y_test_data_path, index=False, header=True)


            return(X_train, X_test, y_train, y_test)
        
        except Exception as e:
            raise CustomException(e,sys)
    

    def train_test_split(self, df):
        logging.info("Data train, test is splitting started.")
        try: 
            X = df.drop(columns=['Unnamed: 0', 'id', 'satisfaction'])
            y = df['satisfaction']

            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

            return(X_train, X_test, y_train, y_test)
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
    
if __name__ == "__main__":
    obj = DataIngestion()
    X_train, X_test, y_train, y_test = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    X_train, X_test, y_train, y_test = data_transformation.initiate_data_transformation(X_train, X_test, y_train, y_test)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)







