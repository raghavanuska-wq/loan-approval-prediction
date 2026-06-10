🏦 Loan Approval Prediction System

This project predicts whether a loan application will be approved or rejected using Machine Learning techniques. The model analyzes applicant information such as income, credit history, education, employment status, and property area to make predictions.

The project includes a trained Random Forest model and an interactive Streamlit web application for real-time loan approval prediction.


Features:

* Data Cleaning and Preprocessing
* Missing Value Handling
* Label Encoding
* Feature Scaling using StandardScaler
* Random Forest Classification Model
* Interactive Streamlit Web Application
* Real-Time Loan Approval Prediction
* Prediction Confidence Score


Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Joblib


Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 76%   |
| Precision | 75%   |
| Recall    | 95%   |
| F1 Score  | 84%   |


Input Features

* Gender
* Married
* Dependents
* Education
* Self Employed
* Applicant Income
* Coapplicant Income
* Loan Amount
* Loan Amount Term
* Credit History
* Property Area


🎯 Output

* ✅ Loan Approved
* ❌ Loan Rejected


📂 Project Structure

loan-approval-prediction/

├── app
  └── app.py
├── data
  └── loan_data.csv
├── models
  └── loan_model.pkl
  └── scaler.pkl
├── notebooks
  └── loan_prediction.ipynb
├── README.md
  └── requirements.txt


 Run Locally

pip install -r requirements.txt
streamlit run app.py


🔮 Future Improvements

* Hyperparameter Tuning
* XGBoost Model
* SHAP Explainability
* Cloud Deployment
* Model Monitoring Dashboard


👩‍💻 Author

**Anuska Raghav**

M.Sc. Data Science

Machine Learning & Data Analytics Enthusiast
