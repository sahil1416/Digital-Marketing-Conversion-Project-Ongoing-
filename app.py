#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:17:23 2024

@author: sahil
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)

# Load the data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('digital_marketing_campaign_dataset.csv', skip_blank_lines=True, on_bad_lines='skip')
        print(f"Loaded data shape: {data.shape}")
        logging.info("Data loaded successfully")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

# Preprocess the data
def preprocess_data(data):
    try:
        # Print column names and their lengths
        for col in data.columns:
            print(f"Column: {col}, Length: {len(data[col])}")

        # Drop unnecessary columns
        columns_to_drop = ['CustomerID', 'AdvertisingPlatform', 'AdvertisingTool']
        data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
        
        # Convert categorical variables to numerical
        if 'Gender' in data.columns:
            data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
        if 'CampaignChannel' in data.columns:
            data = pd.get_dummies(data, columns=['CampaignChannel'], drop_first=True)
        if 'CampaignType' in data.columns:
            data = pd.get_dummies(data, columns=['CampaignType'], drop_first=True)
        
        # Ensure all columns are numeric
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Split features and target
        target_column = 'Conversion' if 'Conversion' in data.columns else 'Converted'
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        print("Preprocessing completed successfully")
        logging.info("Data preprocessed successfully")
        return X, y
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        logging.error(f"Error preprocessing data: {e}")
        return None, None

# Train the model
@st.cache_resource
def train_model(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        param_grid_xgb = {
            'n_estimators': [100, 250, 400],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2]
        }

        xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        random_search_xgb = RandomizedSearchCV(xgb, param_grid_xgb, cv=5, scoring='f1', n_jobs=-1, n_iter=20, verbose=1)
        random_search_xgb.fit(X_train_scaled, y_train)
        
        y_pred = random_search_xgb.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logging.info(f"Model trained successfully. Accuracy: {accuracy:.2f}")
        return random_search_xgb, scaler, accuracy
    except Exception as e:
        logging.error(f"Error training model: {e}")
        return None, None, None

# Make predictions
def predict(model, scaler, input_data):
    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        return prediction
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        return None

# Main Streamlit app
def main():
    st.title('Digital Marketing Campaign Conversion Predictor')
    
    # Load and preprocess data
    data = load_data()
    if data is None:
        st.error("Failed to load data. Please check the CSV file.")
        return

    X, y = preprocess_data(data)
    if X is None or y is None:
        st.error("Failed to preprocess data. Please check the console for more information.")
        return

    # Train model
    model, scaler, accuracy = train_model(X, y)
    if model is None or scaler is None:
        st.error("Failed to train model.")
        return

    st.write(f'Model Accuracy: {accuracy:.2f}')
    
    # Input form for user
    st.header('Enter Customer Information')
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input('Age', min_value=18, max_value=100, value=30)
        gender = st.selectbox('Gender', ['Male', 'Female'])
        income = st.number_input('Annual Income', min_value=0, value=50000)
        campaign_channel = st.selectbox('Campaign Channel', ['Email', 'Social Media', 'PPC', 'Referral', 'SEO'])
        campaign_type = st.selectbox('Campaign Type', ['Awareness', 'Consideration', 'Conversion', 'Retention'])
        ad_spend = st.number_input('Ad Spend', min_value=0.0, value=1000.0)
        ctr = st.number_input('Click Through Rate', min_value=0.0, max_value=1.0, value=0.1)
        conversion_rate = st.number_input('Conversion Rate', min_value=0.0, max_value=1.0, value=0.05)

    with col2:
        website_visits = st.number_input('Website Visits', min_value=0, value=1000)
        pages_per_visit = st.number_input('Pages Per Visit', min_value=0.0, value=2.0)
        time_on_site = st.number_input('Time on Site', min_value=0.0, value=5.0)
        social_shares = st.number_input('Social Shares', min_value=0, value=10)
        email_opens = st.number_input('Email Opens', min_value=0, value=50)
        email_clicks = st.number_input('Email Clicks', min_value=0, value=10)
        previous_purchases = st.number_input('Previous Purchases', min_value=0, value=2)
        loyalty_points = st.number_input('Loyalty Points', min_value=0, value=100)

    # Create input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [0 if gender == 'Male' else 1],
        'Income': [income],
        'AdSpend': [ad_spend],
        'ClickThroughRate': [ctr],
        'ConversionRate': [conversion_rate],
        'WebsiteVisits': [website_visits],
        'PagesPerVisit': [pages_per_visit],
        'TimeOnSite': [time_on_site],
        'SocialShares': [social_shares],
        'EmailOpens': [email_opens],
        'EmailClicks': [email_clicks],
        'PreviousPurchases': [previous_purchases],
        'LoyaltyPoints': [loyalty_points],
        'CampaignChannel_PPC': [1 if campaign_channel == 'PPC' else 0],
        'CampaignChannel_Referral': [1 if campaign_channel == 'Referral' else 0],
        'CampaignChannel_SEO': [1 if campaign_channel == 'SEO' else 0],
        'CampaignChannel_Social Media': [1 if campaign_channel == 'Social Media' else 0],
        'CampaignType_Consideration': [1 if campaign_type == 'Consideration' else 0],
        'CampaignType_Conversion': [1 if campaign_type == 'Conversion' else 0],
        'CampaignType_Retention': [1 if campaign_type == 'Retention' else 0]
    })
    
    # Make prediction
    if st.button('Predict'):
        prediction = predict(model, scaler, input_data)
        if prediction is not None:
            st.write('Prediction:')
            if prediction[0] == 1:
                st.success('The customer is likely to convert!')
            else:
                st.error('The customer is unlikely to convert.')
        else:
            st.error("Failed to make prediction.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logging.error(f"An error occurred: {e}")