# Diabetes Prediction API

API для предсказания вероятности диабета с использованием машинного обучения.

## Технологии
- **FastAPI** - веб-фреймворк
- **scikit-learn** - машинное обучение
- **pandas, numpy** - обработка данных
- **uvicorn** - ASGI сервер

## Локальный запуск

### Установка зависимостей
```bash
python setup.py
```

### Запуск сервера
```bash
python start_server.py
```
или
```bash
python main.py
```

API будет доступно по адресу: http://localhost:8000
Документация: http://localhost:8000/docs

## Развертывание на Render

### Способ 1: Через render.yaml
1. Загрузите код в GitHub репозиторий
2. Подключите репозиторий к Render
3. Render автоматически обнаружит `render.yaml` файл

### Способ 2: Ручная настройка
1. Создайте новый Web Service на Render
2. Подключите GitHub репозиторий
3. Настройки:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt && mkdir -p data/datasets && mkdir -p data/diabetes_models`
   - **Start Command**: `python main.py`

## API Endpoints

### Основные эндпоинты
- `GET /` - Информация об API
- `POST /predict` - Предсказание диабета
- `POST /predict-with-recommendations` - Предсказание с рекомендациями
- `GET /health` - Проверка состояния API
- `GET /docs` - Swagger документация

### Модели
- `POST /train/{model_type}` - Обучение модели
- `GET /model/{model_type}` - Информация о модели
- `GET /models/compare` - Сравнение моделей

### Тестирование
- `GET /sample-data` - Примерные данные для тестирования

## Пример использования

```python
import requests

# Данные пациента
patient_data = {
    "pregnancies": 1,
    "glucose": 85.0,
    "blood_pressure": 66.0,
    "skin_thickness": 29.0,
    "insulin": 0.0,
    "bmi": 26.6,
    "diabetes_pedigree": 0.351,
    "age": 31
}

# Запрос предсказания
response = requests.post("http://localhost:8000/predict", json={
    "patient_data": patient_data,
    "model_type": "random_forest"
})

print(response.json())
```

## Автор
Idzey - https://github.com/Idzey
