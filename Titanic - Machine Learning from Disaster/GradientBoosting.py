from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv
import numpy as np

data=list()
labels=list()
with open("training_data.csv", 'r') as file:
    reader=csv.reader(file)
    header=next(reader)
    for row in reader:
        data.append(list(map(float, row)))
data=np.array(data)

with open("training_labels.csv", 'r') as file:
    reader=csv.reader(file)
    header=next(reader)
    for row in reader:
        labels.append(int(row[0]))
labels=np.array(labels)

test_size=0.1
X_train, X_val, y_train, y_val=train_test_split(data, labels, test_size=test_size)

model=GradientBoostingClassifier(learning_rate=0.05, max_features=4, max_depth=3, random_state=0)
model.fit(X_train, y_train)
y_pred=model.predict(X_val)
predictions=[round(value) for value in y_pred]
accuracy=accuracy_score(y_val, y_pred)
print("acc: %.2f%%" % (accuracy*100))

X_test=list()
with open('test_data.csv', 'r') as file:
    reader=csv.reader(file)
    header=next(reader)
    for row in reader:
        X_test.append(list(map(float, row)))
X_test=np.array(X_test)
test_predictions=model.predict(X_test)

with open('res_GradientBoosting.csv', 'w', encoding='utf-8', newline='') as file:
    writer=csv.writer(file)
    header=['PassengerId', 'Survived']
    writer.writerow(header)
    for i in range(len(test_predictions)):
        writer.writerow([i+892, test_predictions[i]])