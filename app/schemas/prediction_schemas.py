from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class ModelType(str, Enum):
    logistic_regression = "logistic_regression"
    random_forest = "random_forest"

class PatientData(BaseModel):
    pregnancies: int = Field(ge=0, le=20, description="Количество беременностей")
    glucose: float = Field(ge=0, le=300, description="Уровень глюкозы")
    blood_pressure: float = Field(ge=0, le=200, description="Артериальное давление")
    skin_thickness: float = Field(ge=0, le=100, description="Толщина кожной складки")
    insulin: float = Field(ge=0, le=1000, description="Уровень инсулина")
    bmi: float = Field(ge=10, le=60, description="Индекс массы тела")
    diabetes_pedigree: float = Field(ge=0, le=3, description="Наследственность диабета")
    age: int = Field(ge=1, le=120, description="Возраст")

class PredictionRequest(BaseModel):
    patient_data: PatientData
    model_type: ModelType = ModelType.random_forest

class PredictionResponse(BaseModel):
    patient_id: str
    model_used: ModelType
    probability_diabetes: float
    risk_level: str
    prediction: int
    confidence: float

class ModelTrainingResponse(BaseModel):
    model_type: ModelType
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_samples: int
    message: str

class ModelInfo(BaseModel):
    model_type: ModelType
    is_trained: bool
    accuracy: Optional[float] = None
    last_trained: Optional[str] = None
    feature_names: List[str]