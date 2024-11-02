from flask import Flask, request, jsonify
import pickle
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

with open("intrusion_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Welcome to the Secured Cloud Application!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = process_input(data)
    prediction = model.predict([input_data])
    return jsonify({"prediction": prediction[0]})

def process_input(data):
    return [data['feature1'], data['feature2'], data['feature3']]

def train_model():
    data = pd.read_csv("kddcup.data_10_percent.gz", header=None)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    with open("intrusion_model.pkl", "wb") as file:
        pickle.dump(model, file)

if __name__ == '__main__':
    app.run(debug=True)
