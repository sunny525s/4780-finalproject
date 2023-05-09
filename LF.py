import xgboost as xgb
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
# read data
from sklearn.model_selection import train_test_split

# getting the data
data = pd.read_csv('LF_train.csv')
# data2 = pd.read_csv('RF_train.csv')

# data.fillna(data.median())
# print(data.shape)
# data2.fillna(data.median())
# print(data2.shape)

# frames = [data, data2]
# data = pd.concat(frames)
# print(data.shape)

# getting the target labels (whether the leg is normal (0) or lame (1))

data = data.drop(["id"], axis=1)
target = data.loc[:, "LF"]


# turn variables into
cat_cols = ["dob", "forceplate_date", "gait", "speed", "Gait", "Speed"]
data[cat_cols] = data[cat_cols].astype('category')


corr_matrix = data.corr()

# choose number of features to select
n = 10
features = (corr_matrix.nlargest(n, "LF")["LF"].index).drop("LF")

features = features.union(["weight", "age", "speed"])

data = data[features]

# X_train, X_test, y_train, y_test = train_test_split(
#     data, target, test_size=.2)
# # create model instance
# bst = XGBClassifier(n_estimators=5000, max_depth=2,
#                     learning_rate=0.05, objective='binary:logistic', tree_method="approx", enable_categorical=True)

# # train model
# bst.fit(X_train, y_train)
# # make predictions
# preds = bst.predict(X_test)
# accuracy = accuracy_score(y_test, preds)
# print("Accuracy:", accuracy)


# split data into training and testing sets
n_splits = 20
kf = KFold(n_splits=n_splits, shuffle=True)
best_model = None
models = {}
best_accuracy = 0
avg_accuracy = 0
for train_index, test_index in kf.split(data):
    X_train, X_test = data.iloc[train_index], data.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]

    bst = XGBClassifier(n_estimators=100, max_depth=2,
                        learning_rate=0.3, objective='binary:logistic', tree_method="approx", enable_categorical=True)
    bst.fit(X_train, y_train)
    preds = bst.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    if accuracy > best_accuracy:
        best_model = bst
        best_accuracy = accuracy
    print("fold accuracy: " + str(accuracy))
    avg_accuracy += accuracy
    models[bst] = accuracy

print("Avg accuracy: " + str(avg_accuracy/n_splits))
print("Accuracy: " + str(best_accuracy))

# predictions = np.zeros((len(data), 2))
# for model in models:
#     pred_probs = model.predict_proba(data)
#     predictions += pred_probs
# predictions /= len(models)

# y_pred = np.argmax(predictions, axis=1)
# new_acc = accuracy_score(y_pred, target)
# print("average over classifiers: " + str(new_acc))


# load the test data
test_data = pd.read_csv('LF_test.csv')
ids = (test_data.loc[:, "id"]).to_numpy(np.int32)
test_data = test_data.drop(["id"], axis=1)

cat_cols = ["dob", "forceplate_date", "gait", "speed", "Gait", "Speed"]
test_data[cat_cols] = test_data[cat_cols].astype('category')

test_data = test_data[features]

#test_preds = (bst.predict(test_data)).astype(int)


models = dict(sorted(models.items(), key=lambda x: x[1], reverse=True))
models = list(models.keys())[:5]


predictions = np.zeros((len(test_data), 2))
for model in models:
    pred_probs = model.predict_proba(test_data)
    predictions += pred_probs
predictions /= len(models)

test_preds = np.argmax(predictions, axis=1)


final = np.transpose(np.vstack((ids, test_preds)))


# creating csv file
# np.savetxt("LF_test_labels.csv", test_preds, header="id,label", delimiter=",")
df = pd.DataFrame(final, columns=['id', 'LF'])
df.to_csv('LF_test_labels.csv', index=False)
