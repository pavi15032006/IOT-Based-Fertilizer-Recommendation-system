import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pickle
import os

# ---------------- LOAD DATA ----------------
data = pd.read_csv("Crop_recommendation.csv")
data.columns = data.columns.str.strip()

# Remove missing values
data = data.dropna()

# ---------------- LOWERCASE FIX ----------------
for col in ["SoilType", "Season", "PreviousCrop", "Crop"]:
    data[col] = data[col].astype(str).str.lower().str.strip()

print("Nutrient Distribution:")
print(data["RequiredNutrient"].value_counts())

# ---------------- ENCODING ----------------
soil_enc = LabelEncoder()
season_enc = LabelEncoder()
crop_enc = LabelEncoder()

# Encode Soil and Season
data["SoilType"] = soil_enc.fit_transform(data["SoilType"])
data["Season"] = season_enc.fit_transform(data["Season"])

# Fit crop encoder using BOTH Crop + PreviousCrop
all_crops = pd.concat([data["Crop"], data["PreviousCrop"]])
crop_enc.fit(all_crops)

# Transform crop columns
data["Crop"] = crop_enc.transform(data["Crop"])
data["PreviousCrop"] = crop_enc.transform(data["PreviousCrop"])

# ---------------- FEATURES ----------------

# Crop prediction
X_crop = data[["N", "P", "K", "SoilType", "Season", "PreviousCrop"]]
y_crop = data["Crop"]

# Nutrient prediction
X_nutrient = data[["N", "P", "K", "SoilType", "Season", "Crop"]]
y_nutrient = data["RequiredNutrient"]

# ---------------- TRAIN TEST SPLIT ----------------
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_crop, y_crop,
    test_size=0.2,
    random_state=42,
    stratify=y_crop
)

Xn_train, Xn_test, yn_train, yn_test = train_test_split(
    X_nutrient, y_nutrient,
    test_size=0.2,
    random_state=42,
    stratify=y_nutrient
)

# ---------------- MODELS ----------------
model1 = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42
)

model2 = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42
)

# ---------------- TRAIN ----------------
model1.fit(Xc_train, yc_train)
model2.fit(Xn_train, yn_train)

# ---------------- EVALUATION ----------------
yc_pred = model1.predict(Xc_test)
yn_pred = model2.predict(Xn_test)

acc1 = accuracy_score(yc_test, yc_pred)
acc2 = accuracy_score(yn_test, yn_pred)

print("\nCrop Prediction Accuracy:", round(acc1, 4))
print("Nutrient Prediction Accuracy:", round(acc2, 4))

# ---------------- CROSS VALIDATION ----------------
crop_cv = cross_val_score(model1, X_crop, y_crop, cv=5)
nutrient_cv = cross_val_score(model2, X_nutrient, y_nutrient, cv=5)

print("Average Crop Accuracy (CV):", round(crop_cv.mean(), 4))
print("Average Nutrient Accuracy (CV):", round(nutrient_cv.mean(), 4))

# ---------------- CROP-WISE ACCURACY ----------------
report = classification_report(yc_test, yc_pred, output_dict=True)
crop_accuracy = {}

for crop in report:
    if crop not in ["accuracy", "macro avg", "weighted avg"]:
        crop_accuracy[crop] = report[crop]["recall"]

# ---------------- CONFUSION MATRIX ----------------
if not os.path.exists("static"):
    os.makedirs("static")

cm = confusion_matrix(yc_test, yc_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues")
plt.title("Crop Prediction Confusion Matrix")
plt.colorbar()

plt.xticks(range(len(crop_enc.classes_)), crop_enc.classes_, rotation=90)
plt.yticks(range(len(crop_enc.classes_)), crop_enc.classes_)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("static/confusion_matrix.png")
plt.close()

# ---------------- SAVE MODEL ----------------
pickle.dump(
    {
        "model_crop": model1,
        "model_nutrient": model2,
        "soil_enc": soil_enc,
        "season_enc": season_enc,
        "crop_enc": crop_enc,
        "acc_crop": acc1,
        "acc_nutrient": acc2,
        "crop_accuracy": crop_accuracy
    },
    open("model.pkl", "wb")
)

print("\nModel trained and saved successfully!")