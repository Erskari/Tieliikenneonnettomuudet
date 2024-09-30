import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('data/tieliikenneonnettomuudet_2014/tieliikenneonnettomuudet_2014_onnettomuus.csv', delimiter=';')


data.fillna(method='ffill', inplace=True)

data = pd.get_dummies(data,drop_first=True)

scaler = StandardScaler()

datacolumn = data.columns


X = data.drop('Vakavuusko', axis=1)
y = data['Vakavuusko']

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = RandomForestClassifier()

model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Display confusion matrix
print(confusion_matrix(y_test, y_pred))

