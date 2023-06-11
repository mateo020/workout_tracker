import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib 
#create flask app
app = Flask(__name__)
model , ref_cols, target = joblib.load("/models/model.pkl")

@app.route("/")
def Home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    prediction = model
    
if __name__ == "__main__":
    app.run(debug=True)
    