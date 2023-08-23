# Tutorial from Mosh Hamedani
# https://www.youtube.com/watch?v=7eh4d6sabA0&ab_channel=ProgrammingwithMosh

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn import tree

music_data = pd.read_csv('music.csv')

# predictors or input (age and gender in this case)
X = music_data.drop(columns = ['genre'])

# output
y = music_data['genre']

# Split for training and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Build model, fit, predict
model = DecisionTreeClassifier()
model.fit(X_train.values, y_train.values)

# Visualize
graphOutName = 'music-recommender.dot'
tree.export_graphviz(model, graphOutName, 
                     feature_names = ['age', 'gender'], 
                     class_names = sorted(y.unique()), 
                     label = 'all', 
                     rounded = True, 
                     filled = True)
print('Saved model graphic to ' + graphOutName)

# Predict and score
predictions = model.predict(X_test.values)
score = accuracy_score(y_test, predictions)
print(predictions)
print(score)

# Save trained model
modelOutName = 'music-recommender.joblib'
joblib.dump(model, modelOutName)
print('Saved model to ' + modelOutName)

# Load model
loadedModel = joblib.load(modelOutName)
predictions = model.predict([[21, 1]])
print(predictions)