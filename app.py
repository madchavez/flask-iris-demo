from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

model_data = joblib.load("model.pkl")
model = model_data["model"]
species_names = model_data["species_names"]


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        sepal_length = float(request.form.get("sepal_length"))
        sepal_width = float(request.form.get("sepal_width"))
        petal_length = float(request.form.get("petal_length"))
        petal_width = float(request.form.get("petal_width"))

        features = np.array([[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]])

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        predicted_species = species_names[prediction]
        confidence = probabilities[prediction]

        return render_template(
            "index.html",
            result=predicted_species,
            confidence=round(float(confidence), 2),
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width
        )

    except Exception as e:
        return render_template("index.html", error=str(e))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()

        features = np.array([[
            float(data["sepal_length"]),
            float(data["sepal_width"]),
            float(data["petal_length"]),
            float(data["petal_width"])
        ]])

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        predicted_species = species_names[prediction]
        confidence = probabilities[prediction]

        return jsonify({
            "prediction": str(predicted_species),
            "confidence": round(float(confidence), 2),
            "probabilities": {
                str(species_names[i]): round(float(probabilities[i]), 2)
                for i in range(len(species_names))
            }
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)