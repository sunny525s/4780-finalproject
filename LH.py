import xgboost as xgb
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
# read data
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# learning rate =0.1??
# number of estimators =50?

# getting the data
data = pd.read_csv('LH_train.csv')
# data = data.set_index("id")
# getting the target labels (whether the leg is normal (0) or lame (1))
target = data.loc[:, "LH"]
# remove labels from the dataset
features = data.drop(["id", "LH", "dob", "forceplate_date", "gait", "speed", "Gait", "Speed"], axis=1)
#features["dob", "forceplate_date","gait", "speed", "Gait", "Speed"] = features["dob", "forceplate_date","gait", "speed", "Gait", "Speed"].astype("category")


# cat_features = data.loc[:, ["dob", "forceplate_date",
#                             "gait", "speed", "Gait", "Speed"]]
# encoded_cat_features = pd.get_dummies(cat_features)
# # get the features
# features = pd.concat([data.drop(["id", "LH", "dob", "forceplate_date", "gait", "speed", "Gait", "Speed"], axis=1),
#                      encoded_cat_features], axis=1)



# dtrain = xgb.DMatrix(features, label=target, enable_categorial=True)


# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=.2)

print(data.loc[0])
print("__________________________________________\n")

print("________________________\n")
print(features.loc[0])

# create model instance
bst = XGBClassifier(n_estimators=5000, max_depth=2,
                    learning_rate=0.05, objective='binary:logistic', tree_method="approx", enable_categorical=True)
# fit model
bst.fit(X_train, y_train)
# make predictions
preds = bst.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print("Accuracy:", accuracy)

test_data = pd.read_csv('LH_test.csv')
test_features = test_data.drop(["id", "dob", "forceplate_date","gait", "speed", "Gait", "Speed"], axis=1)

# remove labels from the dataset
# test_cf = test_data.loc[:, ["dob", "forceplate_date",
#                             "gait", "speed", "Gait", "Speed"]]
# encoded_test_cf = pd.get_dummies(test_cf)
# # get the features
# test_features = pd.concat([test_data.drop(["id", "dob", "forceplate_date", "gait", "speed", "Gait", "Speed"], axis=1),
#                      encoded_test_cf], axis=1)
print(test_data.loc[0])
print(test_features.shape)
test_preds = bst.predict(test_features)

# creating csv file
np.savetxt("LH_test_labels.csv", test_preds, delimiter=",")

