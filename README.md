# Health-Diagnosis-Chatbot-using-Machine-Learning

This Python code implements a Health Diagnosis Chatbot that uses Machine Learning to predict health conditions based on symptoms provided by the user. The chatbot uses a Decision Tree Classifier and Support Vector Machine (SVM) to make predictions.

## Prerequisites

Before running the code, make sure you have the following:

Python (version 3.x) installed on your system.
Required Python packages: pandas, scikit-learn.

## Dataset

The code uses two CSV files for training and testing data:

1. Data/Training.csv: This file contains the training data with symptom data and corresponding health conditions (prognosis).
2. Data/Testing.csv: This file contains the testing data for evaluating the model's performance.

## Running the code

1. clone the repo
2. Open the file
3. Run the command `pip install -r requirements.txt`.
4. Now run the python file .

## Machine Learning Models

The code uses two machine learning models:

1. Decision Tree Classifier: The Decision Tree model is trained on the training data to classify symptoms and predict health conditions.

2. Support Vector Machine (SVM): The SVM model is used as an alternative classifier for predicting health conditions based on symptoms.

## Feature Importance

The code calculates feature importance using the Decision Tree model to identify the most crucial symptoms for predicting health conditions.


## Usage

1. Make sure you have the necessary prerequisites installed.
2. Place the training and testing CSV files (Training.csv and Testing.csv) in the Data folder.
3. Run the Python script to start the Health Diagnosis ChatBot.
4. The ChatBot will prompt you for your name and symptoms you are experiencing.
5. Based on your symptoms, the ChatBot will suggest possible health conditions and provide precautionary measures.
6. The ChatBot will also calculate the severity of your symptoms and advise whether to consult a doctor or take necessary precautions.

## Improvements and Enhancements

This Health Diagnosis ChatBot is a basic implementation and can be further improved and enhanced in the following ways:

1. Include more sophisticated machine learning models to improve accuracy.
2. Implement a user interface (UI) for a better user experience.
3. Utilize a larger and more diverse dataset to improve prediction capabilities.
4. Incorporate natural language processing (NLP) techniques to better understand user inputs.

