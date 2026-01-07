# Прогнозирование успеваемости студентов

Проект по предсказанию оценок студентов по математике на основе социально-демографических факторов.

## Описание

Модель машинного обучения, которая предсказывает результаты студентов на экзамене по математике. В качестве признаков используются: пол, этническая группа, уровень образования родителей, тип обеда и прохождение подготовительных курсов.

Датасет: [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) (Kaggle)

## Стек технологий

- pandas, numpy — работа с данными
- matplotlib, seaborn — визуализация
- scikit-learn, catboost — модели ML
- Flask — веб-приложение
- Docker — контейнеризация

## Быстрый старт

```bash
# клонирование репозитория
git clone https://github.com/sevastyanovilya/student-score-prediction.git
cd student-score-prediction

# создание виртуального окружения (опционально)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или venv\Scripts\activate для Windows

# установка зависимостей
pip install -r requirements.txt

# запуск приложения
python app.py
```

После запуска приложение доступно по адресу http://localhost:5000

## Структура проекта

```
student_score_prediction/
├── notebooks/          # jupyter notebooks с EDA и экспериментами
│   └── data/          # исходные данные
├── src/
│   ├── components/    # компоненты пайплайна (ingestion, transformation, training)
│   └── pipeline/      # пайплайны обучения и предсказания
├── artifacts/         # сохранённые модели и препроцессор
├── templates/         # html шаблоны
├── static/           # css стили
└── app.py            # flask приложение
```

## Результаты

Лучший результат показал Linear Regression с R² ≈ 0.88 на тестовой выборке.

Протестированные модели:
- Linear Regression
- Random Forest
- Gradient Boosting
- CatBoost
- AdaBoost

## Docker

```bash
docker build -t student-prediction .
docker run -p 5000:5000 student-prediction
```

## Лицензия

MIT
