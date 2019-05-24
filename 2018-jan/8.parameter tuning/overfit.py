import pandas as pd
import os
from sklearn import tree
from sklearn import model_selection

os.chdir('C:/Users/Algorithmica/Downloads')

#read and explore data
titanic_train = pd.read_csv('titanic_train.csv')
titanic_train.shape
titanic_train.info()

#convert categorical features to one-hot encoded continuous features
features = ['Pclass', 'Sex', 'Embarked']
titanic_train1 = pd.get_dummies(titanic_train, columns=features)
print(titanic_train1.shape)

#Drop features not useful for learning pattern
features_to_drop = ['PassengerId', 'Survived', 'Name', 'Age', 'Ticket', 'Cabin']
titanic_train1.drop(features_to_drop, axis=1, inplace=True)

X_train = titanic_train1
y_train = titanic_train[['Survived']]

dt_estimator = tree.DecisionTreeClassifier(random_state=100)
dt_grid = {'max_depth':list(range(3,12))}
grid_dt_estimator = model_selection.GridSearchCV(dt_estimator, dt_grid, cv=10)
grid_dt_estimator.fit(X_train, y_train)

print(grid_dt_estimator.best_estimator_)
print(grid_dt_estimator.best_params_)

#find the cv and train scores of final model
print(grid_dt_estimator.best_score_)
print(grid_dt_estimator.score(X_train, y_train))

