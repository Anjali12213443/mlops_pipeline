# End-to-End MLOps Pipeline — Machine Failure Detection

## Project Overview
An end-to-end MLOps pipeline that detects machine failures using real industrial sensor data. This project covers the full ML lifecycle: data preprocessing, experiment tracking, model serving, and data drift monitoring.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-green)
![Evidently](https://img.shields.io/badge/Evidently-Data%20Drift-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

## Dataset
- **Source:** AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository)
- **Size:** 10,000 records, 14 features
- **Target:** Binary classification — Machine Failure (0 or 1)
- **Failure rate:** 3.4% (imbalanced dataset)

## Project Structure
```
mlops_pipeline/
├── data/
│   ├── processed/        # Train/test splits
│   └── *.png             # EDA visualizations
├── src/
│   └── app.py            # FastAPI prediction service
├── models/               # Saved model artifacts
├── 01_EDA_and_Preprocessing.ipynb
├── 02_Model_Training_MLflow.ipynb
├── 03_Data_Drift_Evidently.ipynb
└── README.md
```

## Tech Stack
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Modeling:** Random Forest, XGBoost, Logistic Regression
- **Experiment Tracking:** MLflow
- **Model Serving:** FastAPI + Uvicorn
- **Drift Detection:** Evidently AI
- **Version Control:** Git + GitHub

## Model Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.8205 | 0.1407 | 0.8382 | 0.2410 | 0.9071 |
| Random Forest | 0.9890 | 0.8594 | 0.8088 | 0.8333 | 0.9696 |
| XGBoost | 0.9865 | 0.7971 | 0.8088 | 0.8029 | 0.9735 |

**Best Model: Random Forest** — highest F1 score and precision with strong recall.

## How to Run

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd mlops_pipeline
```

### 2. Create and activate virtual environment
```bash
python3 -m venv mlops_env
source mlops_env/bin/activate
pip install pandas numpy scikit-learn xgboost mlflow fastapi uvicorn evidently jupyter seaborn
```

### 3. Run the notebooks in order
- `01_EDA_and_Preprocessing.ipynb`
- `02_Model_Training_MLflow.ipynb`
- `03_Data_Drift_Evidently.ipynb`

### 4. Start the prediction API
```bash
uvicorn src.app:app --reload
```

### 5. Test the API
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"Type": 0, "Air_temperature_K": 304.5, "Process_temperature_K": 313.8,
       "Rotational_speed_rpm": 1200, "Torque_Nm": 70.0, "Tool_wear_min": 240,
       "Torque_x_Toolwear": 16800.0, "Temp_difference": 9.3}'
```

## Key Features
- Handles class imbalance using `class_weight='balanced'` and `scale_pos_weight`
- Feature engineering: Torque x Tool wear interaction, temperature difference
- Full experiment tracking with MLflow (parameters, metrics, artifacts)
- REST API for real-time predictions with failure probability score
- Data drift detection comparing training vs production distributions
