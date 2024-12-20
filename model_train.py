import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import joblib
from imblearn.under_sampling import RandomUnderSampler

years = ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']  
data_frames = []


for year in years:
    file_path = f'data/tieliikenneonnettomuudet_{year}/tieliikenneonnettomuudet_{year}_onnettomuus.csv'
    df = pd.read_csv(file_path, delimiter=';')
    data_frames.append(df)

# Combine all data into a single DataFrame
data = pd.concat(data_frames, ignore_index=True)

# Strip whitespace from column names
data.columns = data.columns.str.strip()

# Convert 'Tietyo' column to numerical values
data['Tietyo'] = data['Tietyo'].map({'E': 0, 'K': 1})

# Convert month to categorical season
data['Season'] = data['Kk'].map({
    1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 
    5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'
})

# Check for any NaN values before proceeding
if data.isnull().sum().any():
    print("NaN values found in the dataset. Please address these before proceeding.")
    print(data.isnull().sum())
    
    # Fill NaN values in numerical columns with mean 
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())

    # Fill NaN values in categorical columns with mode
    categorical_cols = data.select_dtypes(include=[object]).columns
    for col in categorical_cols:
        data[col].fillna(data[col].mode()[0])

else:
    print("No NaN values in the dataset.")
    
    
saa_categories = ['1', '2', '3', '4', '5', '6']
Vkpv_categories = ['Maanantai', 'Tiistai', 'Keskiviikko', 'Torstai', 'Perjantai', 'Lauantai', 'Sunnuntai']

data['Saa'] = pd.Categorical(data['Saa'], categories=saa_categories)
data['Vkpv'] = pd.Categorical(data['Vkpv'], categories=Vkpv_categories)

# Convert 'Saa' and other categorical variables to numerical
data = pd.get_dummies(data, columns=['Saa', 'Vkpv', 'Season'], drop_first=True)

# Select features and target variable
features = data.drop(columns=['Vakavuusko', 'X', 'Y', 'Onluokka', 'Osallkm'])
target = data['Vakavuusko']

# Make sure target variable is categorical
target = target.astype('category')

# Check the types of features and target
print("Features Data Types:\n", features.dtypes)
print("Target Data Type:\n", target.dtypes)

# Down-sample class 0
down_sampler = RandomUnderSampler(sampling_strategy={0: 25000}, random_state=42)
X_down, y_down = down_sampler.fit_resample(features, target)

# Apply SMOTE on the minority classes (1 and 2)
smote = SMOTE(sampling_strategy={1: 10000, 2: 32706}, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_down, y_down)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


print("begin model training")

# Train a Random Forest Classifier
model = RandomForestClassifier(class_weight={0:1, 1: 49.84, 2: 7.55}, n_estimators=5 , random_state=42)
model.fit(X_train, y_train)

print("start predicting")

# Predict on the test set
y_pred = model.predict(X_test)

print("evaluate model")

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification = classification_report(y_test, y_pred)

# Print the results
print(f'Accuracy: {accuracy:.4f}')
print('Confusion Matrix:')
print(confusion)
print('Classification Report:')
print(classification)

# Visualizing the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(data['Vakavuusko'].unique()), yticklabels=sorted(data['Vakavuusko'].unique()))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Save the trained model to a file
joblib.dump(model, 'best_acc_model_random_forest_1_12.joblib')

print("Model saved successfully!")
