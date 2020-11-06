import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import missingno as msno
from sklearn.metrics import mean_squared_error, explained_variance_score, max_error, mean_absolute_error, mean_squared_log_error
from sklearn.metrics import median_absolute_error, r2_score, mean_poisson_deviance, mean_gamma_deviance

def plot_hist(df, col):
    '''plot histogram'''

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8, forward=True)
    histogram = sns.distplot(df, ax=ax)
    histogram.set_title('Histogram of ' + col, fontdict={'fontsize':12}, pad=12)
    st.pyplot(fig) 

def plot_pairplot(df):
    '''plot pairplot'''

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8, forward=True)
    pairplot = sns.pairplot(df)
    st.pyplot(pairplot) 

def plot_corr_heatmap(df):
    '''plot correlation heatmap'''

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8, forward=True)
    heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, ax=ax)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
    st.pyplot(fig) 

def plot_missing_matrix(df):
    '''plot missing values matrix'''

    fig, ax = plt.subplots()
    fig.set_size_inches(3, 2, forward=True)
    msno.matrix(df, ax=ax)
    st.pyplot(fig)

def plot_missing_bar(df):
    '''plot missing values bar'''

    fig, ax = plt.subplots()
    fig.set_size_inches(3, 2, forward=True)
    msno.bar(df, ax=ax)
    st.pyplot(fig)

def plot_missing_heatmap(df):
    '''plot missing values heatmap'''

    fig, ax = plt.subplots()
    fig.set_size_inches(3, 2, forward=True)
    msno.heatmap(df, ax=ax)
    st.pyplot(fig)

def loss_metric_regression(model, X_train, y_train, X_test, y_test):
    '''Determine loss metrics for regression models'''
   
    loss_metric_list = [
        mean_squared_error
        ,explained_variance_score
        ,max_error
        ,mean_absolute_error
        ,mean_squared_log_error
        ,median_absolute_error
        ,r2_score
        ,mean_poisson_deviance
        ,mean_gamma_deviance
    ]

    loss_metric_name_list = [
        'Mean Squared Error:'
        ,'Explained Variance Score:'
        ,'Max Error:'
        ,'Mean Absolute Error:'
        ,'Mean Squared Log Error:'
        ,'Median Absolute Error:'
        ,'R2 Score:'
        ,'Mean Poisson Deviance:'
        ,'Mean Gamma Deviance:'
    ]
   
    loss_metric_log_list = [
        'mean_squared_error'
        ,'explained_variance_score'
        ,'max_error'
        ,'mean_absolute_error'
        ,'mean_squared_log_error'
        ,'median_absolute_error'
        ,'r2_score'
        ,'mean_poisson_deviance'
        ,'mean_gamma_deviance'
    ]

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
   
    output = "."

    output += "{0:29} {1:25} {2:25}\n".format("","Train set error","Test set error")
    for loss_metric in list(zip(loss_metric_list, loss_metric_name_list)):
        output += "{0:30} {1:<25} {2:<25}\n".format(
            loss_metric[1]
            ,loss_metric[0](y_train, pred_train)
            ,loss_metric[0](y_test, pred_test))

    return output

# code edited from: https://medium.com/district-data-labs/visual-diagnostics-for-more-informed-machine-learning-7ec92960c96b
def error_compare_models(mods,X,y):
    '''plot predicted vs actual'''

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True)
    for mod, ax in ((mods[0], ax1),(mods[1], ax2),(mods[2], ax3),(mods[3], ax4)):
        predicted = mod[0].predict(X)
        ax.scatter(y, predicted, c='#F2BE2C')
        ax.set_title('Prediction Error for %s' % mod[1])
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4, c='#2B94E9')
        ax.set_ylabel('Predicted')
    plt.xlabel('Measured')
    fig.set_size_inches(10, 8, forward=True)
    st.pyplot(fig)

# code edited from: https://medium.com/district-data-labs/visual-diagnostics-for-more-informed-machine-learning-7ec92960c96b
def resids_compare_models(mods,X,y):
    '''compared residuals of all models'''

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True)
    plt.title('Plotting residuals using training (blue) and test (green) data')
    for m, ax in ((mods[0], ax1),(mods[1], ax2),(mods[2], ax3),(mods[3], ax4)):
        ax.scatter(m[0].predict(X),m[0].predict(X)-y,c='#2B94E9',s=40,alpha=0.5)
        # ax.scatter(m[0].predict(X_tt), m[0].predict(X_tt)-y_tt,c='#94BA65',s=40)
        ax.hlines(y=0, xmin=3, xmax=9)
        ax.set_title(m[1])
        ax.set_ylabel('Residuals')
    plt.xlim([4,8])        # Adjust according to your dataset
    plt.ylim([-5,5])  
    fig.set_size_inches(10, 8, forward=True)
    st.pyplot(fig)