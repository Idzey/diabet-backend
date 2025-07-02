import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
from datetime import datetime
from typing import Tuple, Dict, Any


class DiabetesPredictor:
    def __init__(self):
        self.models = {
            'logistic_regression': None,
            'random_forest': None
        }
        self.scalers = {
            'logistic_regression': StandardScaler(),
            'random_forest': StandardScaler()
        }
        self.feature_names = [
            'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
            'insulin', 'bmi', 'diabetes_pedigree', 'age'
        ]
        self.model_metrics = {}
        self.models_dir = "data/diabetes_models"
        os.makedirs(self.models_dir, exist_ok=True)

        # Загружаем сохраненные модели при инициализации
        self._load_models()

    def _load_models(self):
        """Загрузка сохраненных моделей"""
        try:
            for model_type in ['logistic_regression', 'random_forest']:
                model_path = os.path.join(self.models_dir, f"{model_type}_model.pkl")
                scaler_path = os.path.join(self.models_dir, f"{model_type}_scaler.pkl")
                metrics_path = os.path.join(self.models_dir, f"{model_type}_metrics.pkl")

                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.models[model_type] = joblib.load(model_path)
                    self.scalers[model_type] = joblib.load(scaler_path)

                    if os.path.exists(metrics_path):
                        self.model_metrics[model_type] = joblib.load(metrics_path)

                    print(f"✅ Модель {model_type} загружена")
                else:
                    print(f"⚠️ Модель {model_type} не найдена, будет создана при первом обучении")

        except Exception as e:
            print(f"⚠️ Ошибка загрузки моделей: {e}")

    def _save_model(self, model_type: str, model, scaler, metrics: Dict):
        """Сохранение модели, скалера и метрик"""
        try:
            model_path = os.path.join(self.models_dir, f"{model_type}_model.pkl")
            scaler_path = os.path.join(self.models_dir, f"{model_type}_scaler.pkl")
            metrics_path = os.path.join(self.models_dir, f"{model_type}_metrics.pkl")

            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            joblib.dump(metrics, metrics_path)

            print(f"✅ Модель {model_type} сохранена")

        except Exception as e:
            print(f"⚠️ Ошибка сохранения модели {model_type}: {e}")

    def _generate_sample_data(self, n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
        """Генерация примерных данных для обучения"""
        np.random.seed(42)

        data = {
            'pregnancies': np.random.poisson(3, n_samples),
            'glucose': np.random.normal(120, 30, n_samples),
            'blood_pressure': np.random.normal(70, 20, n_samples),
            'skin_thickness': np.random.exponential(20, n_samples),
            'insulin': np.random.gamma(2, 50, n_samples),
            'bmi': np.random.normal(28, 7, n_samples),
            'diabetes_pedigree': np.random.beta(2, 5, n_samples),
            'age': np.random.gamma(2, 15, n_samples) + 20
        }

        df = pd.DataFrame(data)

        # Создаем целевую переменную на основе правдоподобной логики
        diabetes_prob = (
                0.01 * df['glucose'] +
                0.005 * df['bmi'] +
                0.002 * df['age'] +
                0.1 * df['diabetes_pedigree'] +
                np.random.normal(0, 0.5, n_samples) - 2
        )

        y = (diabetes_prob > np.percentile(diabetes_prob, 70)).astype(int)

        print(f"✅ Сгенерированы примерные данные: {len(df)} образцов")
        return df, y

    def train_model(self, model_type: str) -> Dict[str, Any]:
        """Обучение модели"""
        try:
            # Генерируем данные для обучения
            X, y = self._generate_sample_data()

            # Разделяем данные
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Масштабирование данных
            scaler = self.scalers[model_type]
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Создание и обучение модели
            if model_type == 'logistic_regression':
                model = LogisticRegression(random_state=42, max_iter=1000)
            elif model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10,
                    min_samples_split=5
                )
            else:
                raise ValueError(f"Неизвестный тип модели: {model_type}")

            model.fit(X_train_scaled, y_train)

            # Предсказания и метрики
            y_pred = model.predict(X_test_scaled)

            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred)),
                'recall': float(recall_score(y_test, y_pred)),
                'f1_score': float(f1_score(y_test, y_pred)),
                'training_samples': len(X_train),
                'last_trained': datetime.now().isoformat()
            }

            # Сохранение модели и скалера
            self.models[model_type] = model
            self.model_metrics[model_type] = metrics

            self._save_model(model_type, model, scaler, metrics)

            print(f"✅ Модель {model_type} обучена. Accuracy: {metrics['accuracy']:.3f}")
            return metrics

        except Exception as e:
            raise Exception(f"Ошибка обучения {model_type}: {str(e)}")

    def predict(self, patient_data: Dict, model_type: str) -> Dict[str, Any]:
        """Предсказание для пациента"""
        try:
            if self.models[model_type] is None:
                raise ValueError(f"Модель {model_type} не обучена")

            # Преобразуем данные в DataFrame
            df = pd.DataFrame([patient_data])

            # Убеждаемся, что колонки в правильном порядке
            df = df[self.feature_names]

            # Масштабирование
            X_scaled = self.scalers[model_type].transform(df)

            # Предсказание
            prediction = self.models[model_type].predict(X_scaled)[0]
            probability = self.models[model_type].predict_proba(X_scaled)[0]

            # Определение уровня риска
            diabetes_prob = probability[1]  # Вероятность диабета

            if diabetes_prob < 0.3:
                risk_level = "Низкий"
            elif diabetes_prob < 0.7:
                risk_level = "Средний"
            else:
                risk_level = "Высокий"

            return {
                'prediction': int(prediction),
                'probability': float(diabetes_prob),
                'risk_level': risk_level,
                'confidence': float(max(probability))
            }

        except Exception as e:
            raise Exception(f"Ошибка предсказания: {str(e)}")

    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """Получение информации о модели"""
        return {
            'is_trained': self.models[model_type] is not None,
            'metrics': self.model_metrics.get(model_type, {}),
            'feature_names': self.feature_names
        }