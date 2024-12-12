import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import List, Dict, Any
import gradio as gr

# Constants for directories and file names
MODEL_DIR = 'models'
DATA_DIR = 'datasets'
DATA_FILE = 'cleaned_survey_results_public.csv'
MODEL_NAMES = [
    'CatBoost Regressor',
    'XGBoost Regressor',
    'LGBM Regressor',
]

def load_models(model_names: List[str]) -> Dict[str, Any]:
    """Load machine learning models from disk."""
    models = {}
    for name in model_names:
        path = os.path.join(MODEL_DIR, f"{name.replace(' ', '')}.joblib")
        try:
            models[name] = joblib.load(path)
        except Exception as e:
            print(f"Error loading model {name}: {str(e)}")  # Use print for logging in Gradio
    return models

# Load models
models = load_models(MODEL_NAMES)

# Load dataset
data_path = os.path.join(DATA_DIR, DATA_FILE)
df = pd.read_csv(data_path)

# Prepare features and target
X = df.drop(columns=['Salary'])
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

# Pre-defined input choices
input_choices = {
    'MainBranch': df.MainBranch.unique().tolist(),
    'Country': X.Country.unique().tolist(),
    'EducationLevel': X.EducationLevel.unique().tolist(),
    'RemoteWork': df.RemoteWork.unique().tolist(),
}

# Pre-computed statistics for default values
default_comp = float(df.CompTotal.mean())  # Default CompTotal
max_comp = float(df.CompTotal.max() * 1.5)
default_years = 3.0  # Default years of experience
max_years = float(df.YearsOfExperience.max() * 1.5)

def load_and_predict(mainbranch, country, educationlevel, remotework, comptotal, yearsofexperience):
    """Predict salary using loaded models and evaluate statistics."""
    input_data = pd.DataFrame(
        [[mainbranch, country, educationlevel, remotework, comptotal, yearsofexperience]], 
        columns=['MainBranch', 'Country', 'EducationLevel', 'RemoteWork', 'CompTotal', 'YearsOfExperience']
    )
    
    results = []

    for name, model in models.items():
        try:
            salary_pred = model.predict(input_data)[0]
            y_train_pred = model.predict(X_train)
            
            results.append({
                'Model': name,
                'Predicted Salary': salary_pred,
                'R2 Score (%)': r2_score(y_train, y_train_pred) * 100,
                'Mean Absolute Error': mean_absolute_error(y_train, y_train_pred),
                'Mean Squared Error': mean_squared_error(y_train, y_train_pred),
            })
        except Exception as e:
            print(f"Error during prediction with model {name}: {str(e)}")  # Logging
     
    return pd.DataFrame(results).sort_values(by='R2 Score (%)', ascending=False).reset_index(drop=True)

# Gradio interface
inputs = [
    gr.Dropdown(choices=input_choices['MainBranch'], label="Main Branch"),
    gr.Dropdown(choices=input_choices['Country'], label="Country"),
    gr.Dropdown(choices=input_choices['EducationLevel'], label="Education Level"),
    gr.Dropdown(choices=input_choices['RemoteWork'], label="Remote Work"),
    gr.Number(minimum=0.0, maximum=max_comp, value=default_comp, step=0.5, label="CompTotal"),
    gr.Number(minimum=0.0, maximum=50, value=default_years, step=0.5, label="Years of Experience"),
]

output = gr.Dataframe(label="Prediction Results")

gr.Interface(
    fn=load_and_predict,
    inputs=inputs,
    outputs=output,
    title="Developer Salary Prediction App",
    description="This application predicts developer salaries using multiple machine learning models. Provide your details to get salary predictions.",
).launch()