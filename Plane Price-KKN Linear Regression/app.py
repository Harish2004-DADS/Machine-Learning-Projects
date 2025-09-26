from flask import Flask, render_template, request
import pickle
import pandas as pd

# Load saved model and preprocessors
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
imputer = pickle.load(open("imputer.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect user inputs
        user_input = {
            "Engine Type": request.form["engine_type"],
            "HP or lbs thr ea engine": float(request.form["hp"]),
            "Max speed Knots": float(request.form["max_speed"]),
            "Rcmnd cruise Knots": float(request.form["cruise"]),
            "Stall Knots dirty": float(request.form["stall"]),
            "Fuel gal/lbs": float(request.form["fuel"]),
            "All eng rate of climb": float(request.form["all_climb"]),
            "Eng out rate of climb": float(request.form["eng_out"]),
            "Takeoff over 50ft": float(request.form["takeoff"]),
            "Landing over 50ft": float(request.form["landing"]),
            "Empty weight lbs": float(request.form["empty_weight"]),
            "Length ft/in": float(request.form["length"]),
            "Wing span ft/in": float(request.form["wingspan"]),
            "Range N.M.": float(request.form["range_nm"])
        }

        # Encode engine type
        user_input["Engine Type"] = encoder.transform([user_input["Engine Type"]])[0]

        # Convert to DataFrame
        df_input = pd.DataFrame([user_input])

        # Handle missing, scale
        df_input = imputer.transform(df_input)
        df_input = scaler.transform(df_input)

        # Predict
        prediction = model.predict(df_input)[0]

        return render_template("result.html", prediction=round(prediction, 2))

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
