import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import os
import missingno as msno
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score, max_error, mean_absolute_error, mean_squared_log_error
from sklearn.metrics import median_absolute_error, r2_score, mean_poisson_deviance, mean_gamma_deviance
from utils import *

def main():

    readme_text = st.markdown(open("README.md").read())

    st.sidebar.title("Dashboard Options")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Home Page", "Data Exploration", "Model Performance", "Model Endpoint"])

    if app_mode == "Home Page":
        st.sidebar.success('To continue select another app mode.')
    elif app_mode == "Data Exploration":
        readme_text.empty()
        run_data_exploration()
    elif app_mode == "Model Performance":
        readme_text.empty()
        run_model_performance()
    elif app_mode == "Model Endpoint":
        readme_text.empty()
        run_model_endpoint()


# To make Streamlit fast, st.cache allows us to reuse computation across runs.
# # In this common pattern, we download data from an endpoint only once.
@st.cache
def load_metadata(url):
    return pd.read_csv(url)

# This is the main app app itself, which appears when the user selects "Run the app".
def run_data_exploration():

    # An amazing property of st.cached functions is that you can pipe them into
    # one another to form a computation DAG (directed acyclic graph). Streamlit
    # recomputes only whatever subset is required to get the right answer!
    df = load_metadata(os.path.join(PATH, "winequality-red.csv"))
    
    st.title("Data Exploration")
    st.markdown("Let us have a look at the data first:")
    st.dataframe(df.head())

    # Summary
    st.header("Summary")
    st.markdown("We can quickly summarise each field to get an overall understanding of the data.")
    st.dataframe(df.describe(include='all'))

    # Histogram
    st.header("Histogram")
    st.markdown("Look through the distribution of each feature by selecting from the dropdown below.")
    hist_col = st.selectbox("Column:", list(df.columns), 0)
    plot_hist(df[hist_col], hist_col)

    # Pairplot
    st.header("Pairplot")
    st.markdown("It is useful to look at a scatterplot between variables to understand any correlations. Select checkbox to generate the plot.")
    cols = list(df.columns)
    st_ms = st.multiselect("Columns", df.columns.tolist(), default=cols)
    
    # toggle for display
    is_check = st.checkbox("Display Pairplot")
    if is_check:
        plot_pairplot(df[st_ms])

    # Correlation Heatmap
    st.header("Correlation Heatmap")
    st.markdown("The correlation between features can be quickly visualised with a correlation heatmap.")
    plot_corr_heatmap(df)

    # missingno
    st.header("Missing Values")
    st.markdown("It is also important to look out for any missing values.")
    st.markdown("We'll start by looking at how many missing values for each feature.")
    plot_missing_bar(df)
    st.markdown("It may also be useful to see the order it is missing based on the dataset.")
    plot_missing_matrix(df)
    st.markdown("Finally, a correlation heatmap of missing values will be useful.")
    plot_missing_heatmap(df)

def run_model_performance():

    df = load_metadata(os.path.join(PATH, "winequality-red.csv"))
    
    X = df.drop(columns='quality')
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    gbt_model = pickle.load(open("gbt_model.pkl", "rb"))
    rf_model = pickle.load(open("rf_model.pkl", "rb"))
    nn_model = pickle.load(open("nn_model.pkl", "rb"))
    lr_model = pickle.load(open("lr_model.pkl", "rb"))

    st.title("Model Perfomance")
    st.markdown("We can view the performance of different models for predicting wine quality.")

    # Model metrics
    st.header("Metrics")
    st.markdown("Most important is to look at the key metrics to evaluate each model, particularly for the test set.")
    model_selected = st.selectbox('Select Model'
        ,('Gradient Boosting Tree', 'Random Forest', 'Neural Network', 'Linear Regression'))

    if model_selected=="Gradient Boosting Tree":
        model = gbt_model
    elif model_selected=="Random Forest":
        model = rf_model
    elif model_selected=="Neural Network":
        model = nn_model
    elif model_selected=="Linear Regression":
        model = lr_model

    st.text(loss_metric_regression(model, X_train, y_train, X_test, y_test))
    
    # Plot prediction vs actual
    st.header("Prediction vs Actual (test set)")
    st.markdown("Let us also visualise the predicted vs actual values for each model.")
    models = np.array([(gbt_model,'Gradient Boosting Tree'), (rf_model,'Random Forest')
        ,(nn_model,'Neural Network'), (lr_model, 'Linear Regression')])
    error_compare_models(models, X_test, y_test)

    # Plot residuals
    st.header("Residuals (test set)")
    st.markdown("Finally, we can look at the residuals, which would ideally be around 0.")
    resids_compare_models(models, X_test, y_test)

def run_model_endpoint():
    
    st.title("Model Endpoint")
    st.markdown("We can then deploy our models and accept predictions. First we can choose which model to deploy.")
    
    gbt_model = pickle.load(open("gbt_model.pkl", "rb"))
    rf_model = pickle.load(open("rf_model.pkl", "rb"))
    nn_model = pickle.load(open("nn_model.pkl", "rb"))
    lr_model = pickle.load(open("lr_model.pkl", "rb"))

    st.header("Select Model for prediction")
    model_selected = st.selectbox('Select Model'
        ,('Gradient Boosting Tree', 'Random Forest', 'Neural Network', 'Linear Regression'))

    if model_selected=="Gradient Boosting Tree":
        model = gbt_model
    elif model_selected=="Random Forest":
        model = rf_model
    elif model_selected=="Neural Network":
        model = nn_model
    elif model_selected=="Linear Regression":
        model = lr_model

    st.header("Feature values")
    st.markdown("Set values for input to model.")
    df = load_metadata(os.path.join(PATH, "winequality-red.csv")).drop(columns='quality')
    value_list = []
    for col in df.columns:
        value_list.append(st.number_input(str(col)))
    
    st.header("Prediction")
    st.markdown("The predicted value is:")
    st.text(model.predict(np.array(value_list).reshape(-1,df.shape[1])))



PATH = ""

if __name__ == "__main__":
    main()