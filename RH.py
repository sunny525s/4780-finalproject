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
data.fillna(data.median())

# getting the target labels (whether the leg is normal (0) or lame (1))
target = data.loc[:, "RH"]
data = data.drop(
    ["id", "RH"], axis=1)

cat_cols = ["speed", "Speed", "weight", "age",
            "gait", "Gait", "dob", "forceplate_date"]
data[cat_cols] = data[cat_cols].astype('category')

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=.2)

# create model instance
bst = XGBClassifier(n_estimators=100, max_depth=3,
                    learning_rate=0.2, objective='binary:logistic', tree_method="approx", enable_categorical=True)
# fit model
bst.fit(X_train, y_train)
# make predictions
preds = bst.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print("Accuracy:", accuracy)


# load the test data
test_data = pd.read_csv('RH_test.csv')
ids = (test_data.loc[:, "id"]).to_numpy(np.int32)
test_data = test_data.drop(
    ["id"], axis=1)

cat_cols = ["speed", "Speed", "weight", "age",
            "gait", "Gait", "dob", "forceplate_date"]
test_data[cat_cols] = test_data[cat_cols].astype('category')
test_preds = (bst.predict(test_data)).astype(int)
final = np.transpose(np.vstack((ids, test_preds)))

# creating csv file
df = pd.DataFrame(final, columns=['id', 'RH'])
df.to_csv('RH_test_labels.csv', index=False)
