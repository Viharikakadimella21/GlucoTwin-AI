# GlucoTwin AI

## A Personalized Digital Twin for Early Prediction of Blood-Glucose Spikes

GlucoTwin AI is an AI-powered healthcare Digital Twin prototype designed to predict a patient's future blood-glucose level using historical patient data and dynamic lifestyle/wearable signals.

The system creates a patient-specific digital representation from historical glucose and physiological data, continuously updates the twin state using current inputs, predicts glucose approximately 2 hours ahead, identifies a glucose-risk level, and provides model-based what-if simulations.

---

## Problem Statement

Blood glucose can change significantly depending on factors such as carbohydrate intake, physical activity, heart rate, insulin delivery, and previous glucose trends.

Traditional monitoring mainly focuses on current or past glucose values. GlucoTwin AI aims to provide an early view of the patient's expected glucose state by combining historical patterns with dynamic physiological and lifestyle information.

---

## Proposed Solution

GlucoTwin AI uses machine learning to:

- Analyze historical glucose patterns
- Incorporate dynamic patient signals
- Create a patient-specific Digital Twin state
- Predict glucose approximately 2 hours ahead
- Identify the predicted glucose risk range
- Simulate possible changes in lifestyle inputs
- Display the results through an interactive dashboard

---

## Key Features

### 1. Patient-Specific Digital Twin
Historical patient information is used to build a personalized baseline for each patient.

### 2. Dynamic State Updating
The Digital Twin uses current glucose, heart rate, steps, carbohydrate intake, insulin-related signals, and recent glucose trends.

### 3. 2-Hour Glucose Prediction
The machine learning model predicts the patient's glucose level approximately two hours into the future.

### 4. Risk Identification
The predicted glucose value is mapped into prototype risk categories:

- Low Glucose Risk
- Normal Range
- High Glucose Risk

### 5. What-If Simulation
Users can modify inputs and observe how the trained model's prediction changes.

Example scenarios include:

- Additional carbohydrate intake
- Additional physical activity

### 6. Interactive Dashboard
A Streamlit dashboard provides:

- Patient selection
- Current patient state
- Glucose prediction
- Risk status
- Recent glucose trend
- What-if simulation

---

## System Architecture

```text
Historical Patient Data
        |
        v
Data Preprocessing
        |
        v
Feature Engineering
        |
        +----------------------+
        |                      |
        v                      v
Patient Profile       Dynamic Patient Signals
        |                      |
        +----------+-----------+
                   |
                   v
          Digital Twin State
                   |
                   v
        Machine Learning Model
          (Random Forest)
                   |
                   v
       2-Hour Glucose Prediction
                   |
          +--------+--------+
          |                 |
          v                 v
     Risk Indicator    What-If Simulation
          |                 |
          +--------+--------+
                   |
                   v
          Streamlit Dashboard