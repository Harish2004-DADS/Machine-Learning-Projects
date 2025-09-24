from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
tfidf_transformer = pickle.load(open("tfidf_transformer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email_text = request.form['email_text']
        X_counts = vectorizer.transform([email_text])
        X_tfidf = tfidf_transformer.transform(X_counts)
        prediction = model.predict(X_tfidf)[0]
        return render_template('result.html', prediction=prediction, email_text=email_text)

if __name__ == '__main__':
    app.run(debug=True)

