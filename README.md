 Lab 3: Penguins Classification with XGBoost and FastAPI

Project Overview

This project was created for the AIDI 2004 - AI in Enterprise Systems course. It demonstrates how to build a complete machine learning pipeline using the Penguins dataset, train an XGBoost model, and deploy it with FastAPI. The goal is to classify penguin species based on physical characteristics, while following best practices in preprocessing, validation, logging, and deployment.
## Features

- Loads and preprocesses the Seaborn Penguins dataset
- One-hot encoding for categorical features (`island`, `sex`)
- Label encoding for target variable (`species`)
- XGBoost model with overfitting prevention (`max_depth=3`, `n_estimators=100`)
- FastAPI app with `/predict` endpoint
- Input validation using Pydantic with Enum constraints
- Model and prediction logging using Python’s `logging` module
- Graceful handling of invalid inputs (returns HTTP 400)


 How to Run

1. Install dependencies using `uv`:
   ```bash
   uv venv
   uv pip install -r requirements.txt

Train the model:
python train.py

Run the FastAPI app:
uvicorn app.main:app --reload

Visit http://127.0.0.1:8000/docs to test the API.
