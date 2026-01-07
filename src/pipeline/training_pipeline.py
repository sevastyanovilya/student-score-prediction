import sys
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_training_pipeline():
    """Запускает полный цикл обучения: загрузка -> трансформация -> обучение"""
    try:
        logging.info("=== запуск training pipeline ===")
        
        # 1. загрузка данных
        ingestion = DataIngestion()
        train_path, test_path = ingestion.run()
        
        # 2. предобработка
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.run(train_path, test_path)
        
        # 3. обучение модели
        trainer = ModelTrainer()
        r2_score = trainer.run(train_arr, test_arr)
        
        logging.info(f"=== pipeline завершён, R2={r2_score:.4f} ===")
        
        return r2_score
        
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    score = run_training_pipeline()
    print(f"финальный R2 score: {score:.4f}")
