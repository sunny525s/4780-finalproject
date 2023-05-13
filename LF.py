import xgboost as xgb
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
# read data
from sklearn.model_selection import train_test_split

# getting the data
data = pd.read_csv('LF_train.csv')
data.fillna(data.median())
# data = data.set_index("id")

# getting the target labels (whether the leg is normal (0) or lame (1))
target = data.loc[:, "LF"]
data = data.drop(["id", "LF"], axis=1)
# print(data.loc[0])

cat_cols = ["dob", "forceplate_date", "gait",
            "speed", "Gait", "Speed", "weight"]
data[cat_cols] = data[cat_cols].astype('category')

# features = data["dob", "forceplate_date", "gait", "speed", "Gait", "Speed"].astype("category")

# print("FEATURES")
# print(features.loc[0])

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=.2)


# print(X_train.shape)
# print(X_train.shape)
# create model instance
bst = XGBClassifier(n_estimators=100, max_depth=2,
                    learning_rate=0.05, objective='binary:logistic', tree_method="approx", enable_categorical=True)
# fit model
bst.fit(X_train, y_train)
# make predictions
preds = bst.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print("Accuracy:", accuracy)


# load the test data
test_data = pd.read_csv('LF_test.csv')
ids = (test_data.loc[:, "id"]).to_numpy(np.int32)
# print(ids)
test_data = test_data.drop(["id"], axis=1)

cat_cols = ["dob", "forceplate_date", "gait",
            "speed", "Gait", "Speed", "weight"]
test_data[cat_cols] = test_data[cat_cols].astype('category')
print(test_data.shape)
# remove labels from the dataset
# test_cf = data.loc[:, ["dob", "forceplate_date",
#                        "gait", "speed", "Gait", "Speed"]]
# encoded_test_cf = pd.get_dummies(test_cf)
# # get the features
# test_features = pd.concat([data.drop(["id", "dob", "forceplate_date", "gait", "speed", "Gait", "Speed"], axis=1),
#                            encoded_test_cf], axis=1)
# print(test_features.shape)
test_preds = (bst.predict(test_data)).astype(int)
final = np.transpose(np.vstack((ids, test_preds)))
# print(final)

# creating csv file
# np.savetxt("LF_test_labels.csv", test_preds, header="id,label", delimiter=",")
df = pd.DataFrame(final, columns=['id', 'LF'])
df.to_csv('LF_test_labels.csv', index=False)
