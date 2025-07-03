"""
Скрипт установки зависимостей для Diabetes Prediction API
Author: Idzey
"""

import subprocess
import sys
import os

def install_requirements():
    """Установка всех зависимостей"""
    try:
        print("📦 Установка зависимостей...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Зависимости успешно установлены!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка установки зависимостей: {e}")
        return False

def create_directories():
    """Создание необходимых директорий"""
    directories = [
        "data",
        "data/datasets", 
        "data/diabetes_models"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Создана директория: {directory}")

if __name__ == "__main__":
    print("🔧 Настройка проекта Diabetes Prediction API")
    print("-" * 50)
    
    # Создание директорий
    create_directories()
    
    # Установка зависимостей
    if install_requirements():
        print("\n🎉 Проект успешно настроен!")
        print("🚀 Для запуска сервера используйте: python start_server.py")
        print("📚 Или: python main.py")
    else:
        print("\n❌ Ошибка настройки проекта")
        sys.exit(1)
