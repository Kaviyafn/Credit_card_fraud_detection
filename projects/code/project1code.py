import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st


# Load the dataset
data = pd.read_csv('creditcard.csv')

# Split the data into legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split the data into features (X) and target (y)
x = data.drop(columns="Class", axis=1)
y = data["Class"]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)  # Increased max_iter to avoid convergence warnings
model.fit(x_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(x_train), y_train)
test_acc = accuracy_score(model.predict(x_test), y_test)

# Streamlit web app
st.title("Credit Card Fraud Detection Model")
st.write(f"Training Accuracy: {train_acc:.2f}")
st.write(f"Test Accuracy: {test_acc:.2f}")

# Get user input from Streamlit
input_df = st.text_input("Enter All Required Features Values (comma-separated)")

submit = st.button("Submit")

if submit:
    if input_df:  # Check if input is not empty
        try:
            # Split input values by comma and convert to float
            input_df_splited = list(map(float, input_df.split(',')))
            
            # Check if the input has the correct number of features (equal to the number of columns in the model)
            if len(input_df_splited) != x_train.shape[1]:
                st.write(f"Error: Expected {x_train.shape[1]} feature values. Please enter the correct number of values.")
            else:
                # Convert input to a numpy array and reshape for prediction
                Features = np.asarray(input_df_splited, dtype=np.float64).reshape(1, -1)
                prediction = model.predict(Features)

                # Display the prediction
                if prediction[0] == 0:
                    st.write("Legitimate Transaction")
                else:
                    st.write("Fraudulent Transaction")
        except ValueError:
            st.write("Error: Please enter valid numerical values.")
    else:
        st.write("Error: Please provide input values.")
