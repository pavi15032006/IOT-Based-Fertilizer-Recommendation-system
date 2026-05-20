import pandas as pd
import random
import os

rows = []

soil_types = ["Red", "Black", "Sandy", "Clay"]
seasons = ["Kharif", "Rabi", "Summer"]

# ✅ Added new crops
crops = [
    "Rice", "Maize", "Wheat", "Cotton", "Soybean",
    "Pulses", "Groundnut", "Barley", "Chickpea",
    "Sugarcane", "Sunflower", "Tomato",
    "Potato", "Onion", "Millet", "Sorghum"
]

for i in range(3000):
    N = random.randint(10, 80)
    P = random.randint(10, 80)
    K = random.randint(5, 60)

    soil = random.choice(soil_types)
    season = random.choice(seasons)
    crop = random.choice(crops)

    # Nutrient logic
        
    min_value = min(N, P, K)

    if N > 50 and P > 50 and K > 40:
        nutrient = "None"
    elif min_value == N:
        nutrient = "Nitrogen"
    elif min_value == P:
        nutrient = "Phosphorus"
    else:
        nutrient = "Potassium"

    rows.append([N, P, K, soil, season, crop, crop, nutrient])

df = pd.DataFrame(rows, columns=[
    "N", "P", "K", "SoilType", "Season",
    "PreviousCrop", "Crop", "RequiredNutrient"
])

print("File will be saved at:", os.getcwd())

df.to_csv("Crop_recommendation.csv", index=False)

print("Dataset generated successfully with 16 crops ✅")