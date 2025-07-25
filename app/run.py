"""
Файл для запуска Diabetes Prediction API
Author: Idzey
Date: 2025-07-01
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
