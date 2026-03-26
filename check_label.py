import joblib

# Load the encoder you just saved from your training script
le = joblib.load('models/label_encoder.pkl')

# Generate the dictionary automatically
mapping = {i: label for i, label in enumerate(le.classes_)}

print("\n--- COPY THIS INTO VANGUARD.PY ---")
print(f"attack_map = {mapping}")
print("----------------------------------")