from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load trained model
model = joblib.load(os.path.join("models", "house_price_xgb_model.pkl"))

@app.route("/", methods=["GET", "POST"])
def home():
    predicted_price = None
    if request.method == "POST":
        try:
            # Get user inputs
            input_data = {
                'OverallQual': int(request.form['OverallQual']),
                'GrLivArea': float(request.form['GrLivArea']),
                'GarageCars': int(request.form['GarageCars']),
                'TotalBsmtSF': float(request.form['TotalBsmtSF']),
                '1stFlrSF': float(request.form['1stFlrSF']),
                'FullBath': int(request.form['FullBath']),
                'GarageArea': float(request.form['GarageArea']),
                'YearBuilt': int(request.form['YearBuilt']),
                'YearRemodAdd': int(request.form['YearRemodAdd']),
                'MasVnrArea': float(request.form['MasVnrArea'])
            }

            df_input = pd.DataFrame([input_data])
            pred = model.predict(df_input)[0]
            predicted_price = f"${pred:,.2f}"

        except Exception as e:
            predicted_price = f"Error: {e}"

    return render_template("index.html", predicted_price=predicted_price)

if __name__ == "__main__":
    app.run(debug=True)
