import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def build_preprocessor(self):
        """
        Создаёт пайплайн предобработки:
        - числовые признаки: заполнение медианой + стандартизация
        - категориальные: заполнение модой + one-hot + стандартизация
        """
        try:
            num_cols = ['writing_score', 'reading_score']
            cat_cols = [
                'gender',
                'race_ethnicity', 
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            
            # пайплайн для числовых признаков
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            # пайплайн для категориальных признаков
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])
            
            logging.info(f"числовые колонки: {num_cols}")
            logging.info(f"категориальные колонки: {cat_cols}")
            
            # объединяем всё в ColumnTransformer
            preprocessor = ColumnTransformer([
                ('num', num_pipeline, num_cols),
                ('cat', cat_pipeline, cat_cols)
            ])
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)

    def run(self, train_path, test_path):
        """
        Применяет препроцессор к train/test данным.
        Возвращает (train_array, test_array, путь к препроцессору)
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f"прочитаны train ({len(train_df)}) и test ({len(test_df)})")
            
            preprocessor = self.build_preprocessor()
            
            target_col = 'math_score'
            
            # разделяем на X и y
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]
            
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]
            
            logging.info("применяю препроцессор к данным...")
            
            # fit на train, transform на оба
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)
            
            # собираем обратно с таргетом
            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]
            
            # сохраняем препроцессор
            save_object(self.config.preprocessor_path, preprocessor)
            logging.info(f"препроцессор сохранён: {self.config.preprocessor_path}")
            
            return train_arr, test_arr, self.config.preprocessor_path
            
        except Exception as e:
            raise CustomException(e, sys)
