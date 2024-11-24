import tkinter as tk
from tkinter import ttk
import random
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('best_acc_model_random_forest.joblib')

# Create the main window
root = tk.Tk()
root.title("Accident Severity Prediction")

# Function to handle prediction
def predict_severity():
    num_predictions = int(prediction_count_entry.get())
    predictions = []
    
    for _ in range(num_predictions):
        # Generate random input values (or use fixed values for demonstration)
        input_data = {
            'Tienpit': random.randint(1, 3),
            'Tie': random.randint(1, 300),
            'Aosa': random.randint(1, 100),
            'Aet': random.randint(1, 1000),
            'Ajr': random.randint(0, 2),
            'ELY': random.randint(1, 14),
            'Ontyyppi': random.randint(0, 10),
            'Nopraj': int(speed_dropdown.get()),
            'Taajmerk': 0 if int(speed_dropdown.get()) > 50 else random.choice([0, 1]),
            'Pinta': random.randint(1, 7),
            'Maakunta': random.randint(1, 19),
            'Kunta': random.randint(1, 800),
            'Noplaji': random.randint(1, 6),
            'Toimluokka': random.randint(1, 4),
            'Kk': int(month_dropdown.get()),
            'Valoisuus': int(brightness_dropdown.get()),
            'Weather': int(weather_dropdown.get()),
            'Day': int(day_dropdown.get()),
            'Kvl': int(kvl_spinbox.get()),
            'Raskaskvl': int(raskaskvl_spinbox.get()),
            'Lampotila': int(temp_entry.get()),
            # Add more fields if needed
        }
        
        # Convert to DataFrame and align with model features
        df = pd.DataFrame([input_data])
        for col in model.feature_names_in_:
            if col not in df:
                df[col] = 0
        df = df[model.feature_names_in_]
        
        prediction = model.predict(df)[0]
        predictions.append(prediction)
        
    label_result.config(text=f"Predicted Severities: {predictions}")
    print(predictions)

# Dropdowns for discrete options
def create_dropdown(label_text, options):
    label = tk.Label(root, text=label_text)
    label.pack()
    var = tk.StringVar(value=options[0])
    dropdown = ttk.Combobox(root, textvariable=var, values=options)
    dropdown.pack()
    return dropdown

# Create dropdowns for categorical inputs
month_dropdown = create_dropdown("Select Month:", list(range(1, 13)))
brightness_dropdown = create_dropdown("Brightness (1-4):", [1, 2, 3, 4])
weather_dropdown = create_dropdown("Weather Condition (1-6):", [1, 2, 3, 4, 5, 6])
day_dropdown = create_dropdown("Day of the Week (1-7):", [1, 2, 3, 4, 5, 6, 7])
speed_dropdown = create_dropdown("Speed Limit:", [40, 60, 70, 80, 100, 120])

# Spinboxes for ranged values
def create_spinbox(label_text, min_val, max_val, increment=1):
    label = tk.Label(root, text=label_text)
    label.pack()
    spinbox = tk.Spinbox(root, from_=min_val, to=max_val, increment=increment)
    spinbox.pack()
    return spinbox

kvl_spinbox = create_spinbox("Vehicle Weight Limit:", 200, 8000, 100)
raskaskvl_spinbox = create_spinbox("Heavy Vehicle Limit:", 200, 8000, 100)

# Entry for temperature with linked slider
def update_temp_entry(val):
    temp_entry.delete(0, tk.END)
    temp_entry.insert(0, val)

temp_label = tk.Label(root, text="Temperature (-40 to 35):")
temp_label.pack()

temp_slider = tk.Scale(root, from_=-40, to=35, orient=tk.HORIZONTAL, command=update_temp_entry)
temp_slider.pack()

temp_entry = tk.Entry(root)
temp_entry.pack()
temp_entry.insert(0, "0")

# Prediction count entry for batch predictions
prediction_count_label = tk.Label(root, text="Number of Predictions:")
prediction_count_label.pack()
prediction_count_entry = tk.Entry(root)
prediction_count_entry.pack()
prediction_count_entry.insert(0, "10")

# Advanced Settings Window
def open_advanced_settings():
    settings_window = tk.Toplevel(root)
    settings_window.title("Advanced Settings")

    # Add sliders or inputs for advanced fields, e.g., road condition or intersection type
    tk.Label(settings_window, text="Road Condition (1-5):").pack()
    road_condition_spinbox = tk.Spinbox(settings_window, from_=1, to=5)
    road_condition_spinbox.pack()

    tk.Label(settings_window, text="Intersection Type (1-10):").pack()
    intersection_type_spinbox = tk.Spinbox(settings_window, from_=1, to=10)
    intersection_type_spinbox.pack()

advanced_button = tk.Button(root, text="Advanced Settings", command=open_advanced_settings)
advanced_button.pack()

# Button to make the prediction
predict_button = tk.Button(root, text="Predict", command=predict_severity)
predict_button.pack()

# Label to display result
label_result = tk.Label(root, text="")
label_result.pack()

# Run the GUI event loop
root.mainloop()
