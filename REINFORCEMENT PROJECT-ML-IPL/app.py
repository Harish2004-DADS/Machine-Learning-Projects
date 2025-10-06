from flask import Flask, render_template, request
import pickle
import pandas as pd

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
encoders, le_y = pickle.load(open("encoders.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get input from form
        user_input = {
            "City": request.form["City"],
            "Team1": request.form["Team1"],
            "Team2": request.form["Team2"],
            "Toss Winner": request.form["Toss_Winner"],
            "Toss Decision": request.form["Toss_Decision"]
        }

        # Convert to DataFrame
        user_df = pd.DataFrame([user_input])

        # Encode features using stored encoders
        for col in user_df.columns:
            user_df[col] = encoders[col].transform(user_df[col])

        # Predict
        prediction_encoded = model.predict(user_df)
        prediction = le_y.inverse_transform(prediction_encoded)[0]

        return render_template("result.html", prediction=prediction, data=user_input)

if __name__ == "__main__":
    app.run(debug=True)
