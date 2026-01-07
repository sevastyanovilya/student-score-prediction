import os
import sys
import dill

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(file_path, obj):
    """Сохраняет объект (модель/препроцессор) в pickle файл"""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
            
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Загружает объект из pickle файла"""
    try:
        with open(file_path, "rb") as f:
            return dill.load(f)
            
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Обучает несколько моделей с GridSearchCV и возвращает словарь с R2 скорами.
    
    Args:
        X_train, y_train: обучающая выборка
        X_test, y_test: тестовая выборка
        models: словарь {имя: модель}
        param: словарь {имя: параметры для GridSearch}
    
    Returns:
        dict: {имя модели: R2 score на тесте}
    """
    try:
        report = {}
        
        for name, model in models.items():
            params = param[name]
            
            # grid search для подбора гиперпараметров
            gs = GridSearchCV(model, params, cv=3)
            gs.fit(X_train, y_train)
            
            # используем лучшие параметры
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            # предсказания
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # метрики
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)
            
            report[name] = test_score
            
        return report
        
    except Exception as e:
        raise CustomException(e, sys)
