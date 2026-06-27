from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("crop_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    nitrogen = float(request.form["nitrogen"])
    phosphorus = float(request.form["phosphorus"])
    potassium = float(request.form["potassium"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    rainfall = float(request.form["rainfall"])
    ph = float(request.form["ph"])

    input_data = np.array([[nitrogen, phosphorus, potassium,
                            temperature, humidity, rainfall, ph]])

    prediction = model.predict(input_data)

    return render_template("index.html", prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)