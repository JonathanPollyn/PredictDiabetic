import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the model and preprocessing objects
model = joblib.load('diabetic_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Verify label encoders contain all expected labels
expected_labels = ['0-18', '19-35', '36-50', '51-65', '66-80', '80+']
if not set(expected_labels).issubset(label_encoders['age_range'].classes_):
    label_encoders['age_range'].classes_ = np.append(label_encoders['age_range'].classes_, list(set(expected_labels) - set(label_encoders['age_range'].classes_)))

st.title('Diabetes Prediction App')
st.subheader('An App by Jonathan Ibifubara Pollyn')
st.write('This application is used to make predictions of diabetes based on known features. The purpose of this exercise is to demonstrate the power of Machine learning. The dataset contains a range of health-related attributes collected to aid the development of predictive models to identify any risk of diabetes. The inspiration for this work is due to my passion for working in the healthcare field and my data science knowledge. Below are details of this data and columns, along with a link to the data for more information. If you have any question about this application please contact me via email at j.pollyn@gmail.com ')

st.markdown("""
## Data Description

The dataset contains the following features:

- **Pregnancies**: Number of times pregnant.
- **Glucose**: Plasma glucose concentration over 2 hours in an oral glucose tolerance test.
- **BloodPressure**: Diastolic blood pressure (mm Hg).
- **SkinThickness**: Triceps skinfold thickness (mm).
- **Insulin**: 2-Hour serum insulin (mu U/ml).
- **BMI**: Body mass index (weight in kg / height in m^2).
- **DiabetesPedigreeFunction**: Diabetes pedigree function, a genetic score of diabetes.
- **Age**: Age in years.
- **Outcome**: Binary classification indicating the presence (1) or absence (0) of diabetes.

You can access the dataset [here](https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes).
""")


# Collect user input
st.sidebar.header('User Input Parameters')
pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=100, value=0, step=1, format='%d')
glucose = st.sidebar.number_input('Glucose', min_value=0.0, max_value=100.0, value=0.0)
bloodpressure = st.sidebar.number_input('Blood Pressure', min_value=0.0, max_value=100.0, value=25.0)
skinthickness = st.sidebar.number_input('SkinThickness', min_value=0.0, max_value=100.0, value=25.0)
insulin = st.sidebar.number_input('Insulin', min_value=0.0, max_value=100.0, value=25.0)
bmi = st.sidebar.number_input('BMI', min_value=0.0, max_value=100.0, value=25.0)
diabetespedigreefunction = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=100.0, value=25.0)
age_range = st.sidebar.selectbox('Age Range', ['0-18', '19-35', '36-50', '51-65', '66-80', '80+'])

# Create a DataFrame for the input
input_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [bloodpressure],
    'SkinThickness': [skinthickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [diabetespedigreefunction],
    'age_range': [age_range],
})

# Apply the label encoding for age_range only
input_data['age_range'] = label_encoders['age_range'].transform(input_data['age_range'])

# Define columns to be scaled
columns_to_scale = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'age_range']

# Apply MinMaxScaler
input_data[columns_to_scale] = scaler.transform(input_data[columns_to_scale])

# Predict the model
if st.sidebar.button('Predict'):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.sidebar.write('The model predicts that the patient has diabetes.')
    else:
        st.sidebar.write('The model predicts that the patient does not have diabetes.')
