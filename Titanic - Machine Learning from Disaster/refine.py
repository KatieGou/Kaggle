import pandas as pd

features=pd.read_csv('train.csv', nrows=1)
features=list(features)
print(features)
label=[features[1]]
useful_features=[features[2], features[4], features[6], features[7]] # features[9], features[11]]
print(useful_features)

data=pd.read_csv('train.csv', usecols=useful_features)
# print(data.dtypes)
data.Sex=data.Sex.replace({'male': 1, 'female': 2})
# print(data.mean())
# data.Age=data.Age.fillna(-1)
# print(data.dtypes)
# print(data.isnull().sum())
# print(data)
data.to_csv('training_data.csv', index=False)

labels=pd.read_csv('train.csv', usecols=label)
# print(labels)
# print(labels.dtypes)
labels.to_csv('training_labels.csv', index=False)

# test part
test_data=pd.read_csv('test.csv', usecols=useful_features)
# print('Embarked num:', test_data[data.columns[-1]].count())
test_data.Sex=test_data.Sex.replace({'male': 1, 'female': 2})
# test_data.Age=test_data.Age.fillna(-1)
print(test_data.isnull().sum())
# print(test_data.dtypes)
test_data.to_csv('test_data.csv', index=False)