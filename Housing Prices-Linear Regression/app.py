from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result')
def result():
    bedroom = request.args.get('bedroom', type=float)
    net_sqm = request.args.get('net_sqm', type=float)
    distance_to_center_km = request.args.get('distance_to_center_km', type=float)

    data = np.array([[bedroom, net_sqm, distance_to_center_km]])
    prediction = model.predict(data)[0]
    price = round(prediction, 2)

    return render_template('result.html', price=price)

if __name__ == "__main__":
    app.run(debug=True)
