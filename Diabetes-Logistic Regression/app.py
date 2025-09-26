from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["Pregnancies"]),
            float(request.form["Glucose"]),
            float(request.form["BloodPressure"]), 
            float(request.form["SkinThickness"]), 
            float(request.form["Insulin"]),
            float(request.form["BMI"]),
            float(request.form["DiabetesPedigreeFunction"]),
            float(request.form["Age"]),
        ]
        prediction = model.predict([np.array(features)])
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    except:
        result = "Invalid Input!"

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)

