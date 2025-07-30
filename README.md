# 🌍 Disease Mortality Prediction App (Rwanda WHO Data)

Welcome to the **Disease Mortality Prediction App**, a web-based tool built using **Streamlit** and powered by a **Random Forest Regression model**. This app enables users to explore and predict mortality rates based on disease, country, and year using data from the **World Health Organization (WHO)**.

🔗 **Live App:**  
👉 [https://deathprediction.streamlit.app/](https://deathprediction.streamlit.app/)

---

## 📊 What This App Does

- Loads and explores the **WHO Global Health Estimates dataset**
- Allows users to:
  - Select a country
  - Choose a disease
  - Pick a year
- Predicts the **mortality count** using a machine learning model
- Visualizes trends over time

---

## 🧠 Machine Learning Details

- **Model Used:** `RandomForestRegressor`
- **Input Features:**
  - Country (encoded)
  - Disease (encoded)
  - Year
- **Target:** Predicted number of deaths
- **Model Training:** Happens automatically on app load using cached training for efficiency

---

## 🚀 How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/your-username/disease-mortality-app.git
cd disease-mortality-app
