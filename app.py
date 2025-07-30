import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# Configure the app
st.set_page_config(page_title="Global Disease Mortality Predictor", layout="wide")
st.title("🌍 Global Disease Mortality Predictor")
st.markdown("""
Predict mortality rates by disease and country with health recommendations.
Data source: WHO Global Health Estimates.
""")

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        # Try to load the dataset (replace with your actual path)
        df = pd.read_csv('GHE_FULL_DD.csv')
        
        # Clean and prepare the data
        df = df.rename(columns={
            'DIM_COUNTRY_CODE': 'country',
            'DIM_GHECAUSE_TITLE': 'disease',
            'DIM_SEX_CODE': 'sex',
            'DIM_YEAR_CODE': 'year',
            'VAL_DTHS_RATE100K_NUMERIC': 'death_rate'
        })
        
        # Filter relevant columns
        df = df[['country', 'disease', 'sex', 'year', 'death_rate']]
        
        # Filter recent years and complete cases
        df = df[df['year'] >= 2010]
        df = df.dropna()
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Load the data
data = load_data()

if data.empty:
    st.stop()

# Train the model
@st.cache_resource
def train_model(df):
    try:
        # Encode categorical features
        le_country = LabelEncoder()
        le_disease = LabelEncoder()
        
        df['country_encoded'] = le_country.fit_transform(df['country'])
        df['disease_encoded'] = le_disease.fit_transform(df['disease'])
        
        # Prepare features and target
        X = df[['country_encoded', 'disease_encoded', 'year']]
        y = df['death_rate']
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model, le_country, le_disease
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None, None, None

model, le_country, le_disease = train_model(data)

# Prediction function
def predict_mortality(country, disease, year=2023):
    try:
        # Encode inputs
        country_encoded = le_country.transform([country])[0]
        disease_encoded = le_disease.transform([disease])[0]
        
        # Make prediction
        prediction = model.predict([[country_encoded, disease_encoded, year]])[0]
        return max(0, round(prediction, 1))  # Ensure non-negative
    except ValueError:
        # Handle unknown country/disease
        similar_countries = [c for c in le_country.classes_ if country.lower() in c.lower()]
        similar_diseases = [d for d in le_disease.classes_ if disease.lower() in d.lower()]
        return None, similar_countries, similar_diseases
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, [], []

# Health advice dictionary
HEALTH_ADVICE = {
    "Cardiovascular diseases": [
        "💓 Maintain a heart-healthy diet low in saturated fats",
        "🚭 Avoid tobacco use and secondhand smoke",
        "🏃 Exercise at least 150 minutes per week",
        "🩺 Monitor blood pressure and cholesterol regularly"
    ],
    "Respiratory infections": [
        "💉 Get vaccinated against influenza and pneumonia",
        "😷 Practice good hygiene and wear masks in crowded places",
        "🚬 Avoid smoking and air pollution",
        "🍊 Boost immunity with vitamin C-rich foods"
    ],
    "Neoplasms": [
        "🔬 Get regular cancer screenings as recommended",
        "☀️ Use sun protection to prevent skin cancer",
        "🥦 Eat cruciferous vegetables (broccoli, cauliflower)",
        "🚭 Avoid tobacco and limit alcohol consumption"
    ],
    "Diabetes": [
        "🍽️ Maintain a balanced diet with low glycemic index foods",
        "⚖️ Keep a healthy weight and waist circumference",
        "🩸 Monitor blood sugar levels regularly",
        "🚶 Stay physically active (30 mins/day)"
    ],
    "HIV/AIDS": [
        "🛡️ Practice safe sex and use protection",
        "💊 Adhere to antiretroviral therapy if prescribed",
        "🏥 Get regular medical check-ups",
        "🤝 Seek support groups and counseling"
    ]
}

def get_advice(disease):
    """Get health advice for a specific disease"""
    for key, advice in HEALTH_ADVICE.items():
        if key.lower() in disease.lower():
            return advice
    return [
        "🩺 Consult with a healthcare professional regularly",
        "🥗 Maintain a balanced diet with fruits and vegetables",
        "🏋️ Exercise at least 30 minutes most days",
        "😴 Get 7-9 hours of quality sleep nightly",
        "🚭 Avoid tobacco and limit alcohol consumption"
    ]

# User interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction Inputs")
    
    # Country input with autocomplete
    country_input = st.text_input("Enter country name:", "Rwanda")
    
    # Disease input with autocomplete
    disease_input = st.text_input("Enter disease name:", "HIV/AIDS")
    
    # Year input
    year_input = st.number_input("Select year:", 
                                min_value=2010, 
                                max_value=2030, 
                                value=2023)
    
    # Sex selection
    sex_filter = st.radio("Select sex:", 
                         ("Male", "Female", "Both"), 
                         index=0)

with col2:
    st.subheader("Data Exploration")
    st.write("Top 10 diseases by global mortality (males):")
    
    # Show top diseases
    top_diseases = data[data['sex'] == 'MLE'].groupby('disease')['death_rate'].sum().nlargest(10)
    st.dataframe(top_diseases.sort_values(ascending=False))

# Prediction button
if st.button("Predict Mortality Rate"):
    with st.spinner("Analyzing data..."):
        # Make prediction
        prediction, similar_countries, similar_diseases = predict_mortality(
            country_input, disease_input, year_input
        )
        
        if prediction is not None:
            st.success(f"Predicted mortality rate: {prediction} deaths per 100,000 {sex_filter.lower()} population")
            
            # Show historical trends
            st.subheader("Historical Trends")
            
            # Filter historical data
            history = data[
                (data['country'].str.contains(country_input, case=False)) & 
                (data['disease'].str.contains(disease_input, case=False)) &
                (data['sex'] == ('MLE' if sex_filter == 'Male' else 'FMLE' if sex_filter == 'Female' else 'BTH'))
            ]
            
            if not history.empty:
                # Plot trends
                fig = px.line(
                    history, 
                    x='year', 
                    y='death_rate',
                    title=f"{disease_input} Mortality in {country_input} Over Time",
                    labels={'death_rate': 'Deaths per 100,000', 'year': 'Year'}
                )
                st.plotly_chart(fig)
                
                # Show comparison to global average
                global_avg = data[
                    (data['disease'].str.contains(disease_input, case=False)) &
                    (data['sex'] == ('MLE' if sex_filter == 'Male' else 'FMLE' if sex_filter == 'Female' else 'BTH'))
                ]['death_rate'].mean()
                
                st.metric(
                    label=f"Comparison to Global Average ({sex_filter})",
                    value=f"{prediction} vs {global_avg:.1f}",
                    delta=f"{(prediction - global_avg):.1f}",
                    delta_color="inverse"
                )
            else:
                st.warning("No historical data available for this combination")
            
            # Show health advice
            st.subheader("Health Recommendations")
            for advice in get_advice(disease_input):
                st.info(advice)
            
        else:
            st.error("Could not find exact match for your inputs")
            
            if similar_countries:
                st.write("Did you mean one of these countries?")
                st.write(", ".join(similar_countries))
            
            if similar_diseases:
                st.write("Did you mean one of these diseases?")
                st.write(", ".join(similar_diseases))

# Data visualization section
st.subheader("Global Disease Burden")
viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    # Top countries for selected disease
    if disease_input:
        disease_data = data[
            (data['disease'].str.contains(disease_input, case=False)) &
            (data['sex'] == ('MLE' if sex_filter == 'Male' else 'FMLE' if sex_filter == 'Female' else 'BTH'))
        ]
        
        if not disease_data.empty:
            top_countries = disease_data.groupby('country')['death_rate'].mean().nlargest(10)
            fig = px.bar(
                top_countries,
                title=f"Countries with Highest {disease_input} Mortality",
                labels={'value': 'Deaths per 100,000', 'country': 'Country'}
            )
            st.plotly_chart(fig, use_container_width=True)

with viz_col2:
    # Mortality by disease category
    if country_input:
        country_data = data[
            (data['country'].str.contains(country_input, case=False)) &
            (data['sex'] == ('MLE' if sex_filter == 'Male' else 'FMLE' if sex_filter == 'Female' else 'BTH'))
        ]
        
        if not country_data.empty:
            top_diseases = country_data.groupby('disease')['death_rate'].mean().nlargest(10)
            fig = px.pie(
                top_diseases,
                names=top_diseases.index,
                values=top_diseases.values,
                title=f"Top Diseases in {country_input}"
            )
            st.plotly_chart(fig, use_container_width=True)

# Add disclaimer
st.markdown("---")
st.caption("""
⚠️ **Disclaimer**: Predictions are based on statistical modeling and historical data. 
Actual mortality rates may vary. Always consult healthcare professionals for medical advice.
""")