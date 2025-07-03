#!/bin/bash
# Скрипт сборки для Render
echo "🔧 Установка зависимостей..."
pip install -r requirements.txt

echo "📁 Создание необходимых директорий..."
mkdir -p data/datasets
mkdir -p data/diabetes_models

echo "✅ Сборка завершена!"
