from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the CSV files
mortality_data = pd.read_csv('death.csv')
incidence_data = pd.read_csv('incd.csv')


# Preprocess the data
def preprocess_data():
    # Preprocess mortality_data
    # Drop any rows with missing values
    mortality_data.dropna(inplace=True)

    # Perform feature scaling for numerical columns using StandardScaler
    scaler = StandardScaler()
    numerical_cols = ['Met Objective of 45.5', 'Age-Adjusted Death Rate',
                      'Lower 95% Confidence Interval for Death Rate', 'Upper 95% Confidence Interval for Death Rate',
                      'Average Deaths per Year', 'Recent Trend (2)', 'Recent 5-Year Trend (2) in Death Rates',
                      'Lower 95% Confidence Interval for Trend', 'Upper 95% Confidence Interval for Trend']
    mortality_data[numerical_cols] = scaler.fit_transform(mortality_data[numerical_cols])

    # Preprocess incidence_data
    # Drop any rows with missing values
    incidence_data.dropna(inplace=True)

    # Perform feature scaling for numerical columns using StandardScaler
    incidence_data[numerical_cols] = scaler.transform(incidence_data[numerical_cols])
    # Return preprocessed data
    return mortality_data, incidence_data


# Train the prediction models
def train_models(mortality_data, incidence_data):
    # Extract features and target variables for mortality prediction
    mortality_features = mortality_data[
        ['Met Objective of 45.5', 'Age-Adjusted Death Rate', 'Lower 95% Confidence Interval for Death Rate',
         'Upper 95% Confidence Interval for Death Rate', 'Average Deaths per Year', 'Recent Trend (2)',
         'Lower 95% Confidence Interval for Trend', 'Upper 95% Confidence Interval for Trend']]
    mortality_target = mortality_data['Recent 5-Year Trend (2) in Death Rates']

    # Perform preprocessing and feature scaling for mortality_features
    scaler = StandardScaler()
    mortality_features = scaler.fit_transform(mortality_features)

    # Train the mortality prediction model
    mortality_model = LinearRegression()
    mortality_model.fit(mortality_features, mortality_target)

    # Extract features and target variables for incidence prediction
    incidence_features = incidence_data[
        ['Met Objective of 45.5', 'Age-Adjusted Incidence Rate', 'Lower 95% Confidence Interval for Incidence Rate',
         'Upper 95% Confidence Interval for Incidence Rate', 'Average Incidence per Year', 'Recent Trend (2)',
         'Lower 95% Confidence Interval for Trend', 'Upper 95% Confidence Interval for Trend']]
    incidence_target = incidence_data['Recent 5-Year Trend (2) in Incidence Rates']

    # Perform preprocessing and feature scaling for incidence_features
    incidence_features = scaler.transform(incidence_features)

    # Train the incidence prediction model
    incidence_model = LinearRegression()
    incidence_model.fit(incidence_features, incidence_target)

    # Return trained models
    return mortality_model, incidence_model


# Define the route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        location = request.form['location']

        # Find the corresponding row in the mortality_data and incidence_data DataFrames
        mortality_row = mortality_data[mortality_data['County'][1] == location]
        incidence_row = incidence_data[incidence_data['County'][1] == location]
        print(mortality_row)
        if not mortality_row.empty and not incidence_row.empty:
            # Get the features for prediction
            mortality_features = mortality_row[
                ['Met Objective of 45.5', 'Age-Adjusted Death Rate', 'Lower 95% Confidence Interval for Death Rate',
                 'Upper 95% Confidence Interval for Death Rate', 'Average Deaths per Year', 'Recent Trend (2)',
                 'Recent 5-Year Trend (2) in Death Rates', 'Lower 95% Confidence Interval for Trend',
                 'Upper 95% Confidence Interval for Trend']]
            incidence_features = incidence_row[['Met Objective of 45.5', 'Age-Adjusted Incidence Rate',
                                                'Lower 95% Confidence Interval for Incidence Rate',
                                                'Upper 95% Confidence Interval for Incidence Rate',
                                                'Average Incidence per Year', 'Recent Trend (2)',
                                                'Recent 5-Year Trend (2) in Incidence Rates',
                                                'Lower 95% Confidence Interval for Trend',
                                                'Upper 95% Confidence Interval for Trend']]

            # Perform prediction using the trained models
            mortality_prediction = mortality_model.predict(mortality_features)
            incidence_prediction = incidence_model.predict(incidence_features)

            # Calculate percentage based on prediction and round to one decimal place
            mortality_percentage = round(mortality_prediction[0] * 100, 1)
            incidence_percentage = round(incidence_prediction[0] * 100, 1)

            return render_template('index.html', mortality=mortality_percentage, incidence=incidence_percentage)
        else:
            error_message = "Location not found!"
            return render_template('index.html', error=error_message)

    return render_template('index.html')


if __name__ == '__main__':
    # Preprocess the data
    mortality_data, incidence_data = preprocess_data()

    # Train the prediction models
    mortality_model, incidence_model = train_models(mortality_data, incidence_data)

    app.run(debug=True,host='0.0.0.0',port=80)
