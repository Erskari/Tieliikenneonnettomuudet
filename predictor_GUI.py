import tkinter as tk
import joblib
import pandas as pd
import random

# Load the trained model
model = joblib.load('best_acc_model_random_forest.joblib')

def predict_severity():
    
    num_predictions = prediction_count_slider.get()  # Number of predictions to generate
    
    predictions = []
    
    for _ in range(num_predictions):
        # Randomly generate values for specific fields
        Tienpit = random.randint(1, 3)
        Tie = random.randint(1, 300)
        Aosa = random.randint(1, 100)
        Aet = random.randint(1, 1000)
        Ajr = random.randint(0, 2)
        ELY = random.randint(1, 14)
        Ontyyppi = random.randint(0, 100)
        Nopraj = random.choice([40, 60, 70, 80, 100, 120])
        Taajmerk = 0 if Nopraj > 50 else random.choice([0, 1])  # Ensure Taajmerk is 0 if Nopraj > 50
        Pinta = random.randint(1, 7)
        Maakunta = random.randint(1, 19)
        Kunta = random.randint(1, 800)
        Noplaji = random.randint(1, 6)
        Nopsuunvas = Nopraj  # Same as Nopraj
        Nopsuunoik = Nopraj  # Same as Nopraj
        Toimluokka = random.randint(1, 4)
        
        # User-defined inputs from sliders
        Kk = month_slider.get()  # Month
        Valoisuus = valoisuus_slider.get()  # Brightness
        Kvl = kvl_slider.get()  # Vehicle weight limit
        Raskaskvl = raskaskvl_slider.get()  # Heavy vehicle limit
        Lampotila = lampotila_slider.get()  # Temperature
        Saa = weather_slider.get()  # Weather condition (1-6)
        Vkpv = day_slider.get()  # Day of the week (1-7)
        
        
        # Prepare one-hot encoding for weather condition
        weather_dummies = {f'Saa_{i}': 1 if i == Saa else 0 for i in range(1, 7)}
        
        
        # Prepare one-hot encoding for Season
        season_mapping = {
            1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 
            5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'
        }
        Season = season_mapping[Kk]
        season_dummies = {
            'Season_Autumn': 1 if Season == 'Autumn' else 0,
            'Season_Spring': 1 if Season == 'Spring' else 0,
            'Season_Summer': 1 if Season == 'Summer' else 0,
            'Season_Winter': 1 if Season == 'Winter' else 0,
        }
        
        # Prepare one-hot encoding for Vkpv (Days of the week)
        vkpv_mapping = {
            1: 'Maanantai', 2: 'Tiistai', 3: 'Keskiviikko',
            4: 'Torstai', 5: 'Perjantai', 6: 'Lauantai', 7: 'Sunnuntai'
        }
        Vkpv = vkpv_mapping[Vkpv]
        vkpv_dummies = {
            'Vkpv_Lauantai': 1 if Vkpv == 'Lauantai' else 0,
            'Vkpv_Keskiviikko': 1 if Vkpv == 'Keskiviikko' else 0,
            'Vkpv_Maanantai': 1 if Vkpv == 'Maanantai' else 0,
            'Vkpv_Perjantai': 1 if Vkpv == 'Perjantai' else 0,
            'Vkpv_Sunnuntai': 1 if Vkpv == 'Sunnuntai' else 0,
            'Vkpv_Tiistai': 1 if Vkpv == 'Tiistai' else 0,
            'Vkpv_Torstai': 1 if Vkpv == 'Torstai' else 0,
        }
        
        
        # Fixed values
        Tienlev = 120
        Paallyslev = Tienlev
        Tietyo = random.randint(0, 1)  # Randomly generate between 0 and 1
        Risteys = random.randint(0, 8)
        Poikkileik = random.choice([1, 2])
        Paallystlk = random.choice([10, 20])  # 10 or 20
        Nakos150 = random.randint(90, 110)
        Nakos300 = random.randint(50, 90)
        Nakos460 = random.randint(10, 80)
        Runkotie = 0  # Fixed value
        Vuosi = 2024
    
        # Prepare the input data for prediction
        input_data = {
            'Tienpit': Tienpit,
            'Tie': Tie,
            'Aosa': Aosa,
            'Aet': Aet,
            'Ajr': Ajr,
            'Vuosi': Vuosi,
            'Kk': Kk,
            'ELY': ELY,
            'Ontyyppi': Ontyyppi,
            'Nopraj': Nopraj,
            'Taajmerk': Taajmerk,
            'Pinta': Pinta,
            'Valoisuus': Valoisuus,
            'Onnpaikka': 1,  # Fixed value
            'Maakunta': Maakunta,
            'Kunta': Kunta,
            'Noplaji': Noplaji,
            'Nopsuunvas': Nopsuunvas,
            'Nopsuunoik': Nopsuunoik,
            'Toimluokka': Toimluokka,
            'Kvl': Kvl,
            'Raskaskvl': Raskaskvl,
            'Tienlev': Tienlev,
            'Tietyo': Tietyo,
            'Paallyste': random.choice([1, 2]),  # Randomly choose between 1 or 2
            'Lampotila': Lampotila,
            'Risteys': Risteys,
            'Poikkileik': Poikkileik,
            'Paallyslev': Paallyslev,
            'Paallystlk': Paallystlk,
            'Nakos150': Nakos150,
            'Nakos300': Nakos300,
            'Nakos460': Nakos460,
            'Runkotie': Runkotie
        }
        
        # Merge the weather dummies into the input data
        input_data.update(weather_dummies)
        input_data.update(season_dummies)
        input_data.update(vkpv_dummies)
        
    
        # Convert input data to DataFrame
        df = pd.DataFrame([input_data])
        
        # Convert input data to DataFrame and align with model's features
        df = pd.DataFrame([input_data])
        for col in model.feature_names_in_:
            if col not in df:
                df[col] = 0
        df = df[model.feature_names_in_]    
    
        # Predict using the model
        prediction = model.predict(df)[0]
        predictions.append(prediction)
        
    
    # Display the predictions
    label_result.config(text=f"Predicted Severities: {predictions}")
    
    # Print the predictions to the console
    print(predictions)
    
def close_program():
    root.destroy()
    



# Create the main window
root = tk.Tk()
root.title("Accident Severity Prediction")

# Create sliders for user inputs
tk.Label(root, text="Select Month (1-12):").pack()
month_slider = tk.Scale(root, from_=1, to=12, orient=tk.HORIZONTAL)
month_slider.pack()

tk.Label(root, text="Select Brightness (1-4):").pack()
valoisuus_slider = tk.Scale(root, from_=1, to=4, orient=tk.HORIZONTAL)
valoisuus_slider.pack()

tk.Label(root, text="Weather (1-6):").pack()
weather_slider = tk.Scale(root, from_=1, to=6, orient='horizontal')
weather_slider.pack()

tk.Label(root, text="Day (1-7):").pack()
day_slider = tk.Scale(root, from_=1, to=7, orient='horizontal')
day_slider.pack()


tk.Label(root, text="Set Vehicle Weight Limit (200-8000):").pack()
kvl_slider = tk.Scale(root, from_=200, to=8000, orient=tk.HORIZONTAL)
kvl_slider.pack()

tk.Label(root, text="Set Heavy Vehicle Limit (200-8000):").pack()
raskaskvl_slider = tk.Scale(root, from_=200, to=8000, orient=tk.HORIZONTAL)
raskaskvl_slider.pack()

tk.Label(root, text="Set Temperature (-40 to +35):").pack()
lampotila_slider = tk.Scale(root, from_=-40, to=35, orient=tk.HORIZONTAL)
lampotila_slider.pack()

# Add a slider to select the number of predictions
tk.Label(root, text="Number of Predictions:").pack()
prediction_count_slider = tk.Scale(root, from_=1, to=1000, orient=tk.HORIZONTAL)
prediction_count_slider.pack()

# Create a button to make the prediction
button_predict = tk.Button(root, text="Predict", command=predict_severity)
button_predict.pack()

# Label to display the result
label_result = tk.Label(root, text="")
label_result.pack()

# Create an Exit button
button_exit = tk.Button(root, text="Exit", command=close_program)
button_exit.pack()

# Run the GUI event loop
root.mainloop()
