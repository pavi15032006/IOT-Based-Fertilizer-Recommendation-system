from flask import Flask, request, render_template
import pickle
import numpy as np
import os

# ✅ FIX FOR RENDER / SERVER
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# ---------------- IDEAL VALUES ----------------
IDEAL_N = 50
IDEAL_P = 40
IDEAL_K = 40

# ---------------- FLASK APP ----------------
app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
model_data = pickle.load(open("model.pkl", "rb"))

model2 = model_data["model_nutrient"]
soil_enc = model_data["soil_enc"]
season_enc = model_data["season_enc"]
crop_enc = model_data["crop_enc"]
acc2 = model_data["acc_nutrient"]

# ---------------- HOME PAGE ----------------
@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        try:

            # ---------------- USER INPUT ----------------
            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])

            soil = request.form["soil"].lower().strip()
            season = request.form["season"].lower().strip()
            crop = request.form["crop"].lower().strip()

            # ---------------- VALIDATION ----------------
            if crop not in crop_enc.classes_:
                return render_template(
                    "index.html",
                    error=f"Invalid crop! Available crops: {', '.join(crop_enc.classes_)}"
                )

            if soil not in soil_enc.classes_:
                return render_template(
                    "index.html",
                    error=f"Invalid soil type! Available: {', '.join(soil_enc.classes_)}"
                )

            if season not in season_enc.classes_:
                return render_template(
                    "index.html",
                    error=f"Invalid season! Available: {', '.join(season_enc.classes_)}"
                )

            # ---------------- ENCODING ----------------
            soil_encoded = soil_enc.transform([soil])[0]
            season_encoded = season_enc.transform([season])[0]
            crop_encoded = crop_enc.transform([crop])[0]

            # ---------------- MODEL INPUT ----------------
            data = np.array([
                [N, P, K, soil_encoded, season_encoded, crop_encoded]
            ])

            # ---------------- ML PREDICTION ----------------
            probabilities = model2.predict_proba(data)[0]

            nutrient_index = np.argmax(probabilities)

            predicted_nutrient = model2.classes_[nutrient_index]

            nutrient_confidence = round(
                probabilities[nutrient_index] * 100,
                2
            )

            # ---------------- DEFICIENCY CHECK ----------------
            deficit_N = max(0, IDEAL_N - N)
            deficit_P = max(0, IDEAL_P - P)
            deficit_K = max(0, IDEAL_K - K)

            # ---------------- NUTRIENT STATUS ----------------
            status_N = "Deficient" if deficit_N > 0 else "Sufficient"
            status_P = "Deficient" if deficit_P > 0 else "Sufficient"
            status_K = "Deficient" if deficit_K > 0 else "Sufficient"

            # ---------------- FERTILIZER RECOMMENDATION ----------------
            fertilizer_data = []

            # Nitrogen
            if deficit_N > 0:
                urea_amount = round(deficit_N / 0.46, 2)

                fertilizer_data.append(
                    ("Urea", f"{urea_amount} kg/acre")
                )

            # Phosphorus
            if deficit_P > 0:
                dap_amount = round(deficit_P / 0.46, 2)

                fertilizer_data.append(
                    ("DAP", f"{dap_amount} kg/acre")
                )

            # Potassium
            if deficit_K > 0:
                mop_amount = round(deficit_K / 0.60, 2)

                fertilizer_data.append(
                    ("MOP", f"{mop_amount} kg/acre")
                )

            # If all nutrients sufficient
            if len(fertilizer_data) == 0:
                fertilizer_data.append(
                    ("No fertilizer required", "0 kg/acre")
                )

            # ---------------- SOIL HEALTH ----------------
            health = round((N + P + K) / 3, 2)

            if health < 30:
                health_status = "Poor"

            elif health < 50:
                health_status = "Medium"

            else:
                health_status = "Good"

            # ---------------- CREATE STATIC FOLDER ----------------
            if not os.path.exists("static"):
                os.makedirs("static")

            # ---------------- GRAPH ----------------
            graph_path = os.path.join("static", "graph.png")

            plt.figure(figsize=(6, 4))

            plt.bar(
                ["Nitrogen", "Phosphorus", "Potassium"],
                [N, P, K]
            )

            plt.title("Soil Nutrient Levels")

            plt.xlabel("Nutrients")

            plt.ylabel("Values")

            plt.tight_layout()

            plt.savefig(graph_path)

            plt.close()

            # ---------------- SEND RESULTS ----------------
            return render_template(

                "index.html",

                crop_name=crop,

                status_N=status_N,
                status_P=status_P,
                status_K=status_K,

                fertilizer_data=fertilizer_data,

                predicted_nutrient=predicted_nutrient,

                nutrient_confidence=nutrient_confidence,

                health=health,

                health_status=health_status,

                graph=graph_path,

                acc2=round(acc2 * 100, 2)
            )

        except Exception as e:

            return render_template(
                "index.html",
                error=f"Error: {str(e)}"
            )

    return render_template("index.html")


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)