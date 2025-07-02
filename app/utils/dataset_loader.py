import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
import requests
import os
from typing import Tuple


class DatasetLoader:
    def __init__(self):
        self.data_dir = "data/datasets"
        os.makedirs(self.data_dir, exist_ok=True)

    def load_pima_diabetes_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Загрузка датасета Pima Indians Diabetes"""
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

        # Названия колонок
        column_names = [
            'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
            'insulin', 'bmi', 'diabetes_pedigree', 'age', 'outcome'
        ]

        try:
            # Загружаем данные
            df = pd.read_csv(url, names=column_names)

            # Разделяем на признаки и целевую переменную
            X = df.drop('outcome', axis=1)
            y = df['outcome']

            # Сохраняем локально
            df.to_csv(os.path.join(self.data_dir, 'pima_diabetes.csv'), index=False)

            print(f"✅ Датасет загружен: {len(df)} образцов")
            return X, y

        except Exception as e:
            print(f"❌ Ошибка загрузки датасета: {e}")
            return self._generate_sample_data()

    def load_local_csv(self, file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Загрузка локального CSV файла"""
        try:
            df = pd.read_csv(file_path)

            # Предполагаем, что последняя колонка - это целевая переменная
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            print(f"✅ Локальный датасет загружен: {len(df)} образцов")
            return X, y

        except Exception as e:
            print(f"❌ Ошибка загрузки локального файла: {e}")
            return self._generate_sample_data()

    def _generate_sample_data(self, n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
        """Генерация примерных данных (fallback)"""
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

        # Создаем целевую переменную
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