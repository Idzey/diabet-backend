from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uuid

from app.models.ml_models import DiabetesPredictor
from app.schemas.prediction_schemas import (
    PredictionRequest, PredictionResponse, ModelTrainingResponse,
    ModelInfo, ModelType
)
from app.utils.data_preprocessing import validate_medical_data, get_health_recommendations

# Инициализация приложения
app = FastAPI(
    title="Diabetes Prediction API",
    description="API для определения вероятности диабета с использованием машинного обучения",
    version="1.0.0",
    contact={
        "name": "Idzey",
        "url": "https://github.com/Idzey"
    }
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация ML предиктора
predictor = DiabetesPredictor()


@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске приложения"""
    try:
        # Проверяем наличие обученных моделей, если нет - обучаем
        if not predictor.models['logistic_regression']:
            print("Training Logistic Regression model...")
            predictor.train_model('logistic_regression')

        if not predictor.models['random_forest']:
            print("Training Random Forest model...")
            predictor.train_model('random_forest')

        print("✓ All models loaded successfully")
    except Exception as e:
        print(f"Error during startup: {e}")


@app.get("/")
async def root():
    """Главная страница API"""
    return {
        "message": "Diabetes Prediction API",
        "version": "1.0.0",
        "description": "API для определения вероятности диабета",
        "author": "Idzey",
        "endpoints": {
            "predict": "/predict",
            "train": "/train/{model_type}",
            "model_info": "/model/{model_type}",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_diabetes(request: PredictionRequest):
    """Предсказание вероятности диабета"""
    try:
        # Валидация и предобработка данных
        patient_data = validate_medical_data(request.patient_data.dict())

        # Предсказание
        result = predictor.predict(patient_data, request.model_type.value)

        # Генерация уникального ID для пациента
        patient_id = str(uuid.uuid4())[:8]

        # Формирование ответа
        response = PredictionResponse(
            patient_id=patient_id,
            model_used=request.model_type,
            probability_diabetes=result['probability'],
            risk_level=result['risk_level'],
            prediction=result['prediction'],
            confidence=result['confidence']
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict-with-recommendations")
async def predict_with_recommendations(request: PredictionRequest):
    """Предсказание с рекомендациями"""
    try:
        # Получаем базовое предсказание
        prediction_response = await predict_diabetes(request)

        # Добавляем рекомендации
        patient_data = validate_medical_data(request.patient_data.dict())
        recommendations = get_health_recommendations(
            prediction_response.probability_diabetes,
            patient_data
        )

        return {
            "prediction": prediction_response.dict(),
            "recommendations": recommendations,
            "patient_data_processed": patient_data
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/train/{model_type}", response_model=ModelTrainingResponse)
async def train_model(model_type: ModelType, background_tasks: BackgroundTasks):
    """Обучение модели"""
    try:
        # Обучение модели
        metrics = predictor.train_model(model_type.value)

        response = ModelTrainingResponse(
            model_type=model_type,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            training_samples=metrics['training_samples'],
            message=f"Model {model_type.value} trained successfully"
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")


@app.get("/model/{model_type}", response_model=ModelInfo)
async def get_model_info(model_type: ModelType):
    """Получение информации о модели"""
    try:
        info = predictor.get_model_info(model_type.value)

        model_info = ModelInfo(
            model_type=model_type,
            is_trained=info['is_trained'],
            accuracy=info['metrics'].get('accuracy'),
            last_trained=info['metrics'].get('last_trained'),
            feature_names=info['feature_names']
        )

        return model_info

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model info error: {str(e)}")


@app.get("/models/compare")
async def compare_models():
    """Сравнение производительности моделей"""
    try:
        lr_info = predictor.get_model_info('logistic_regression')
        rf_info = predictor.get_model_info('random_forest')

        comparison = {
            "logistic_regression": {
                "is_trained": lr_info['is_trained'],
                "metrics": lr_info['metrics']
            },
            "random_forest": {
                "is_trained": rf_info['is_trained'],
                "metrics": rf_info['metrics']
            },
            "recommendation": "random_forest" if (
                    rf_info['metrics'].get('f1_score', 0) >
                    lr_info['metrics'].get('f1_score', 0)
            ) else "logistic_regression"
        }

        return comparison

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(e)}")


@app.get("/health")
async def health_check():
    """Проверка состояния API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_status": {
            "logistic_regression": predictor.models['logistic_regression'] is not None,
            "random_forest": predictor.models['random_forest'] is not None
        }
    }


@app.get("/sample-data")
async def get_sample_patient_data():
    """Получение примерных данных пациента для тестирования"""
    return {
        "low_risk_patient": {
            "pregnancies": 1,
            "glucose": 85.0,
            "blood_pressure": 66.0,
            "skin_thickness": 29.0,
            "insulin": 0.0,
            "bmi": 26.6,
            "diabetes_pedigree": 0.351,
            "age": 31
        },
        "high_risk_patient": {
            "pregnancies": 8,
            "glucose": 196.0,
            "blood_pressure": 76.0,
            "skin_thickness": 36.0,
            "insulin": 249.0,
            "bmi": 36.5,
            "diabetes_pedigree": 0.875,
            "age": 29
        }
    }


if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)