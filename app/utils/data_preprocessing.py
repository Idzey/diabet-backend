import pandas as pd
import numpy as np
from typing import Dict, Any


def validate_medical_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Валидация и предобработка медицинских данных"""

    # Базовые проверки и корректировки
    processed_data = data.copy()

    # Ограничения по возрасту
    if processed_data['age'] < 18:
        processed_data['age'] = 18
    elif processed_data['age'] > 120:
        processed_data['age'] = 120

    # Ограничения по глюкозе
    if processed_data['glucose'] < 0:
        processed_data['glucose'] = 0
    elif processed_data['glucose'] > 300:
        processed_data['glucose'] = 300

    # Ограничения по BMI
    if processed_data['bmi'] < 10:
        processed_data['bmi'] = 10
    elif processed_data['bmi'] > 70:
        processed_data['bmi'] = 70

    # Обработка нулевых значений для некоторых параметров
    if processed_data['blood_pressure'] == 0:
        processed_data['blood_pressure'] = 70  # средний показатель

    if processed_data['skin_thickness'] == 0:
        processed_data['skin_thickness'] = 20  # средний показатель

    if processed_data['insulin'] == 0:
        processed_data['insulin'] = 80  # средний показатель

    return processed_data


def get_health_recommendations(probability: float, patient_data: Dict[str, Any]) -> list[str]:
    """Генерация рекомендаций на основе данных пациента"""
    recommendations = []

    if probability > 0.7:
        recommendations.append("Настоятельно рекомендуется консультация эндокринолога")
        recommendations.append("Необходимо регулярно контролировать уровень сахара в крови")

    if patient_data['bmi'] > 30:
        recommendations.append("Рекомендуется снижение веса под контролем специалиста")
        recommendations.append("Увеличить физическую активность")

    if patient_data['glucose'] > 140:
        recommendations.append("Следить за углеводным питанием")
        recommendations.append("Исключить простые углеводы из рациона")

    if patient_data['blood_pressure'] > 90:
        recommendations.append("Контролировать артериальное давление")
        recommendations.append("Ограничить потребление соли")

    if patient_data['age'] > 45:
        recommendations.append("Регулярные профилактические осмотры")

    if not recommendations:
        recommendations.append("Поддерживать здоровый образ жизни")
        recommendations.append("Регулярные профилактические осмотры")

    return recommendations