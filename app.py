import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import List, Dict, Any

# Constants for directories and file names
MODEL_DIR = 'models'
DATA_DIR = 'datasets'
DATA_FILE = 'cleaned_survey_results_public_v2.csv'
MODEL_NAMES = [
    'LGBM Regressor',
    'KNeighbors Regressor',
    'Decision Tree Regressor',
]

def load_models(model_names: List[str]) -> Dict[str, Any]:
    """Load machine learning models from disk."""
    models = {}
    for name in model_names:
        path = os.path.join(MODEL_DIR, f"{name.replace(' ', '')}.joblib")
        try:
            models[name] = joblib.load(path)
        except Exception as e:
            st.error(f"Error loading model {name}: {str(e)}")
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

def load_and_predict(sample: pd.DataFrame) -> pd.DataFrame:
    """Predict salary using loaded models and evaluate statistics."""
    results = []

    for name, model in models.items():
        try:
            salary_pred = model.predict(sample)[0]
            y_train_pred = model.predict(X_train)
            
            results.append({
                'Model': name,
                'Predicted Salary': salary_pred,
                'R2 Score (%)': r2_score(y_train, y_train_pred) * 100,
                'Mean Absolute Error': mean_absolute_error(y_train, y_train_pred),
                'Mean Squared Error': mean_squared_error(y_train, y_train_pred),
            })
        except Exception as e:
            st.error(f"Error during prediction with model {name}: {str(e)}")

    return pd.DataFrame(results).sort_values(by='R2 Score (%)', ascending=False).reset_index(drop=True)

# Streamlit UI setup
st.set_page_config(page_title="Developer Salary Prediction App", page_icon="🤑", layout="wide")
st.title("🤑 **Developer Salary Prediction**")

# Sidebar inputs
st.sidebar.header("Input Information")
mainbranch = st.sidebar.selectbox("**MainBranch**", options=input_choices['MainBranch'])
country = st.sidebar.selectbox("**Country**", options=input_choices['Country'])
educationlevel = st.sidebar.selectbox("**Education Level**", options=input_choices['EducationLevel'])
remotework = st.sidebar.selectbox("**Remote Work**", options=input_choices['RemoteWork'])
comptotal = st.sidebar.number_input("**CompTotal**", min_value=0.0, max_value=max_comp, value=default_comp)
yearsofexperience = st.sidebar.number_input("**Years of Experience**", min_value=0.0, max_value=max_years, value=default_years)

# Handling predictions
if st.sidebar.button(label=':rainbow[Predict Salary]'):
    input_data = pd.DataFrame(
        [[mainbranch, country, educationlevel, remotework, comptotal, yearsofexperience]], 
        columns=['MainBranch', 'Country', 'EducationLevel', 'RemoteWork', 'CompTotal', 'YearsOfExperience'])
    
    results_df = load_and_predict(input_data)
    
    if not results_df.empty:
        st.write("### Prediction Results:")
        st.dataframe(results_df)

# Disclaimer Section
st.markdown("---")
st.text('''
    >> Developer Salary Prediction App <<
    This Streamlit application predicts developer salary using multiple machine learning models including LGBM, XGBoost, and Random Forest regressors. 
    Users can input developer information through a user-friendly interface, which includes fields such as country, education level, and years of experience.
        
    > Features:
        **Input Components**: 
        - **MainBranch**: Select your main area of expertise in development, such as software engineering, data science, or web development. This selection may influence salary expectations based on the branch's demand and trends.
        
        - **Country**: Choose your country from the dropdown list. Regions often exhibit varying salary scales due to economic factors, the cost of living, and market demand for tech workers.
        
        - **Education Level**: Indicate the highest level of education you have completed. Higher educational qualifications often correlate with higher earning potential in the tech industry.
        
        - **Remote Work**: Specify whether you primarily work remotely, on-site, or in a hybrid setup. Remote work setups can affect salary offers, especially if hiring companies are based in different geographic areas.
        
        - **CompTotal**: Enter your expected total compensation, which includes salary, bonuses, and other benefits. This field is crucial for setting a base for salary predictions and facilitates comparisons.
        
        - **Years of Experience**: Provide the number of years you've been in a coding-related job. Generally, more years of experience are associated with higher salaries due to skill accumulation and professional development.
        
        **Data Processing**: 
        - The app employs a pre-processed dataset, cleaned and prepared for model training. 
        - It utilizes features including country, education level, and years of experience for predictions.
        - Models are loaded from disk, obtaining predictions based on user-provided input.

        **Prediction**: The app performs predictions with loaded models and calculates performance metrics like R2 score.
        **Results Display**: The predicted salary and model performance metrics are presented in a user-friendly format.
        
    > Usage: 
       Fill out the developer information and click "Predict Salary" to derive insights on anticipated salary and model performance.
       
    > Disclaimer: 
       This application serves educational purposes. Predictions are not guaranteed to be accurate.
''')
