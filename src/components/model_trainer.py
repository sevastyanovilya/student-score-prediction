import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor, 
    GradientBoostingRegressor
)
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def run(self, train_array, test_array):
        """
        Обучает несколько моделей, выбирает лучшую по R2 и сохраняет.
        Возвращает R2 score лучшей модели.
        """
        try:
            logging.info("разделяю массивы на X и y")
            
            # последний столбец — это таргет
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            # словарь моделей для тестирования
            models = {
                'linear_regression': LinearRegression(),
                'decision_tree': DecisionTreeRegressor(),
                'random_forest': RandomForestRegressor(),
                'gradient_boosting': GradientBoostingRegressor(),
                'adaboost': AdaBoostRegressor(),
                'catboost': CatBoostRegressor(verbose=False),
            }
            
            # параметры для grid search (базовые)
            params = {
                'linear_regression': {},
                'decision_tree': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                },
                'random_forest': {
                    'n_estimators': [8, 16, 32, 64, 128],
                },
                'gradient_boosting': {
                    'learning_rate': [0.1, 0.01, 0.05],
                    'subsample': [0.6, 0.7, 0.8, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128],
                },
                'adaboost': {
                    'learning_rate': [0.1, 0.01, 0.5],
                    'n_estimators': [8, 16, 32, 64, 128],
                },
                'catboost': {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100],
                },
            }
            
            logging.info("запускаю обучение и оценку моделей...")
            
            model_scores = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )
            
            # находим лучшую модель
            best_score = max(model_scores.values())
            best_model_name = [k for k, v in model_scores.items() if v == best_score][0]
            best_model = models[best_model_name]
            
            logging.info(f"лучшая модель: {best_model_name} с R2={best_score:.4f}")
            
            if best_score < 0.6:
                raise CustomException("ни одна модель не показала R2 > 0.6", sys)
            
            # сохраняем модель
            save_object(self.config.model_path, best_model)
            logging.info(f"модель сохранена: {self.config.model_path}")
            
            # финальная проверка
            predictions = best_model.predict(X_test)
            final_r2 = r2_score(y_test, predictions)
            
            return final_r2
            
        except Exception as e:
            raise CustomException(e, sys)
