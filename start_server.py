#!/usr/bin/env python
"""
Скрипт для запуска Diabetes Prediction API
Author: Idzey
Date: 2025-07-01
"""

import uvicorn
import sys
import os

# Добавляем корневую директорию в sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🚀 Запуск Diabetes Prediction API...")
    print("📍 API будет доступно по адресу: http://localhost:8000")
    print("📚 Документация: http://localhost:8000/docs")
    print("🔧 Для остановки нажмите Ctrl+C")
    print("-" * 50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
