import xgboost as xgb
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
# read data
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# getting the data
data = pd.read_csv('RH_train.csv')
# getting the target labels (whether the leg is normal (0) or lame (1))
target = data.loc[:, "RH"]

features = data.drop(["id", "RH", "dob", "forceplate_date",
                     "gait", "speed", "Gait", "Speed"], axis=1)

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=.2)

# create model instance
bst = XGBClassifier(n_estimators=5000, max_depth=2,
                    learning_rate=0.05, objective='binary:logistic', tree_method="approx", enable_categorical=True)
# fit model
bst.fit(X_train, y_train)
# make predictions
preds = bst.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print("Accuracy:", accuracy)

# load the test data
test_data = pd.read_csv('RH_test.csv')
test_features = test_data.drop(
    ["id", "dob", "forceplate_date", "gait", "speed", "Gait", "Speed"], axis=1)
test_preds = bst.predict(test_features)

# creating csv file
np.savetxt("RH_test_labels.csv", test_preds, delimiter=",")
