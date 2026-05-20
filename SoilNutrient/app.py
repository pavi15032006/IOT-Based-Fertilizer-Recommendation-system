from flask import Flask, request, render_template
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# Ideal nutrient values
IDEAL_N = 50
IDEAL_P = 40
IDEAL_K = 40

app = Flask(__name__)

# Load model
model_data = pickle.load(open("model.pkl", "rb"))

model2 = model_data["model_nutrient"]
soil_enc = model_data["soil_enc"]
season_enc = model_data["season_enc"]
crop_enc = model_data["crop_enc"]
acc2 = model_data["acc_nutrient"]

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":
        try:
            # ---------------- INPUT ----------------
            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])

            soil = request.form["soil"].lower().strip()
            season = request.form["season"].lower().strip()
            crop = request.form["crop"].lower().strip()

            # ---------------- VALIDATION ----------------
            if crop not in crop_enc.classes_:
                return render_template("index.html", error=f"Invalid crop! Available: {', '.join(crop_enc.classes_)}")

            if soil not in soil_enc.classes_:
                return render_template("index.html", error=f"Invalid soil type! Available: {', '.join(soil_enc.classes_)}")

            if season not in season_enc.classes_:
                return render_template("index.html", error=f"Invalid season! Available: {', '.join(season_enc.classes_)}")

            # ---------------- ENCODING ----------------
            soil_encoded = soil_enc.transform([soil])[0]
            season_encoded = season_enc.transform([season])[0]
            crop_encoded = crop_enc.transform([crop])[0]

            # ---------------- ML PREDICTION ----------------
            data = np.array([[N, P, K, soil_encoded, season_encoded, crop_encoded]])

            probabilities = model2.predict_proba(data)[0]
            nutrient_index = np.argmax(probabilities)
            predicted_nutrient = model2.classes_[nutrient_index]
            nutrient_confidence = probabilities[nutrient_index] * 100

            # ---------------- MULTI-NUTRIENT LOGIC ----------------
            deficit_N = max(0, IDEAL_N - N)
            deficit_P = max(0, IDEAL_P - P)
            deficit_K = max(0, IDEAL_K - K)

            status_N = "Deficient" if deficit_N > 0 else "Sufficient"
            status_P = "Deficient" if deficit_P > 0 else "Sufficient"
            status_K = "Deficient" if deficit_K > 0 else "Sufficient"

            # ---------------- FERTILIZER ----------------
            fertilizer_data = []

            if status_N == "Deficient":
                fertilizer_data.append(("Urea", f"{round(deficit_N / 0.46, 2)} kg/acre"))

            if status_P == "Deficient":
                fertilizer_data.append(("DAP", f"{round(deficit_P / 0.46, 2)} kg/acre"))

            if status_K == "Deficient":
                fertilizer_data.append(("MOP", f"{round(deficit_K / 0.60, 2)} kg/acre"))

            if not fertilizer_data:
                fertilizer_data.append(("No fertilizer required", "0 kg/acre"))

            # ---------------- SOIL HEALTH ----------------
            health = round((N + P + K) / 3, 2)

            if health < 30:
                health_status = "Poor"
            elif health < 50:
                health_status = "Medium"
            else:
                health_status = "Good"

            # ---------------- GRAPH ----------------
            if not os.path.exists("static"):
                os.makedirs("static")

            graph_path = "static/graph.png"

            plt.figure()
            plt.bar(["Nitrogen", "Phosphorus", "Potassium"], [N, P, K])
            plt.title("Soil Nutrient Levels")
            plt.savefig(graph_path)
            plt.close()

            # ---------------- OUTPUT ----------------
            return render_template(
                "index.html",
                crop_name=crop,

                status_N=status_N,
                status_P=status_P,
                status_K=status_K,

                fertilizer_data=fertilizer_data,  # ✅ FIXED

                predicted_nutrient=predicted_nutrient,
                nutrient_confidence=round(nutrient_confidence, 2),

                health=health,
                health_status=health_status,
                graph=graph_path,
                acc2=round(acc2 * 100, 2)
            )

        except Exception as e:
            return render_template("index.html", error=f"Error: {str(e)}")

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)