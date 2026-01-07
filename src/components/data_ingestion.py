import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    """Пути для сохранения данных"""
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def run(self):
        """
        Читает исходные данные, разбивает на train/test и сохраняет.
        Возвращает пути к train и test файлам.
        """
        logging.info("начинаю загрузку данных")
        
        try:
            # читаем csv (путь может отличаться в зависимости от окружения)
            data = pd.read_csv(os.path.join('notebooks', 'data', 'stud.csv'))
            logging.info(f"загружено {len(data)} строк")
            
            # создаём директорию artifacts если её нет
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            
            # сохраняем сырые данные
            data.to_csv(self.config.raw_data_path, index=False)
            
            # разбиваем на train/test
            train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
            logging.info(f"train: {len(train_df)}, test: {len(test_df)}")
            
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            
            logging.info("данные сохранены в artifacts/")
            
            return self.config.train_data_path, self.config.test_data_path
            
        except Exception as e:
            logging.error(f"ошибка при загрузке данных: {e}")
            raise CustomException(e, sys)
