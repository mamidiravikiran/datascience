import pandas as pd
from sklearn import tree, model_selection, preprocessing, ensemble, feature_selection
from sklearn_pandas import CategoricalImputer
import numpy as np
import xgboost as xgb

#creation of data frames from csv
titanic_train = pd.read_csv("C:\\Users\\Algorithmica\\Downloads\\titanic_train.csv")
print(titanic_train.info())

#read test data
titanic_test = pd.read_csv("C:\\Users\\Algorithmica\\Downloads\\titanic_test.csv")
print(titanic_test.info())

titanic = pd.concat((titanic_train, titanic_test), axis = 0)
titanic.drop(['Survived'], axis=1, inplace=True)

#impute missing values for continuous features
imputable_cont_features = ['Age', 'Fare']
cont_imputer = preprocessing.Imputer()
cont_imputer.fit(titanic[imputable_cont_features])
print(cont_imputer.statistics_)
titanic[imputable_cont_features] = cont_imputer.transform(titanic[imputable_cont_features])

#impute missing values for categorical features
cat_imputer = CategoricalImputer()
cat_imputer.fit(titanic['Embarked'])
print(cat_imputer.fill_)
titanic['Embarked'] = cat_imputer.transform(titanic['Embarked'])

#create title feature from name
def extract_title(name):
     return name.split(',')[1].split('.')[0].strip()
#extract_title('abc, mr.def')
titanic['Title'] = titanic['Name'].map(extract_title)

#create family size feature from sibsp, parch
titanic['FamilySize'] = titanic['SibSp'] +  titanic['Parch'] + 1

#create family group feature from family-size
def convert_familysize(size):
    if(size == 1): 
        return 'Single'
    elif(size <=5): 
        return 'Medium'
    else: 
        return 'Large'
titanic['FamilyGroup'] = titanic['FamilySize'].map(convert_familysize)

#label encoding of categorical (string) features
lab_encoder_sex = preprocessing.LabelEncoder()
lab_encoder_sex.fit(titanic['Sex'])
print(lab_encoder_sex.classes_)
titanic['Sex'] = lab_encoder_sex.transform(titanic['Sex'])

lab_encoder_emb = preprocessing.LabelEncoder()
lab_encoder_emb.fit(titanic['Embarked'])
print(lab_encoder_emb.classes_)
titanic['Embarked'] = lab_encoder_emb.transform(titanic['Embarked'])

lab_encoder_pclass = preprocessing.LabelEncoder()
lab_encoder_pclass.fit(titanic['Pclass'])
print(lab_encoder_pclass.classes_)
titanic['Pclass'] = lab_encoder_pclass.transform(titanic['Pclass'])

lab_encoder_title = preprocessing.LabelEncoder()
lab_encoder_title.fit(titanic['Title'])
print(lab_encoder_title.classes_)
titanic['Title'] = lab_encoder_title.transform(titanic['Title'])

lab_encoder_fgroup = preprocessing.LabelEncoder()
lab_encoder_fgroup.fit(titanic['FamilyGroup'])
print(lab_encoder_fgroup.classes_)
titanic['FamilyGroup'] = lab_encoder_fgroup.transform(titanic['FamilyGroup'])

#one hot encoding of categorical integer features
ohe_features = ['Sex','Embarked','Pclass', 'Title', 'FamilyGroup']
ohe = preprocessing.OneHotEncoder()
ohe.fit(titanic[ohe_features])
print(ohe.n_values_)
tmp1 = ohe.transform(titanic[ohe_features]).toarray()

features = ['Age', 'Fare', 'Parch' , 'SibSp', 'FamilySize']
tmp2 = titanic[features].values

tmp = np.concatenate((tmp1,tmp2), axis=1)

X_train = tmp[:titanic_train.shape[0]]
y_train = titanic_train['Survived']

#ada boost
ada_estimator = ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(),random_state=100)
ada_grid = {'n_estimators':list(range(50,101,50)), 'learning_rate':[0.1,0.2,1.0], 'base_estimator__max_depth':[1,3,5], 'base_estimator__criterion':['entropy', 'gini'] }
ada_grid_estimator = model_selection.GridSearchCV(ada_estimator, ada_grid, scoring='accuracy', cv=10, return_train_score=True)
ada_grid_estimator.fit(X_train, y_train)

print(ada_grid_estimator.best_score_)
print(ada_grid_estimator.best_params_)
final_estimator = ada_grid_estimator.best_estimator_
print(final_estimator.estimators_)
print(final_estimator.estimator_weights_)
print(final_estimator.estimator_errors_)
print(final_estimator.score(X_train, y_train))

#gradient boosting
gb_estimator = ensemble.GradientBoostingClassifier(random_state=100)
gb_grid = {'n_estimators':list(range(50,101,50)), 'learning_rate':[0.1,0.2,1.0], 'max_depth':[1,3,5]}
gb_grid_estimator = model_selection.GridSearchCV(gb_estimator, gb_grid, scoring='accuracy', cv=10, return_train_score=True)
gb_grid_estimator.fit(X_train, y_train)

print(gb_grid_estimator.best_score_)
print(gb_grid_estimator.best_params_)
final_estimator = gb_grid_estimator.best_estimator_
print(final_estimator.estimators_)
print(final_estimator.score(X_train, y_train))

#xgb = boosting + regulaization for overfit control
xgb_estimator = xgb.XGBClassifier(random_state=100, n_jobs=-1)
xgb_grid = {'n_estimators':list(range(50,101,50)), 'learning_rate':[0.1,0.2,1.0], 'max_depth':[1,3,5], 'gamma':[0,0.01,0.1,0.2], 'reg_alpha':[0,0.5,1], 'reg_lambda':[0,0.5,1]}
xgb_grid_estimator = model_selection.GridSearchCV(xgb_estimator, xgb_grid, scoring='accuracy', cv=10, return_train_score=True)
xgb_grid_estimator.fit(X_train, y_train)

print(xgb_grid_estimator.best_score_)
print(xgb_grid_estimator.best_params_)
final_estimator = xgb_grid_estimator.best_estimator_
print(final_estimator.score(X_train, y_train))

X_test = tmp[titanic_train.shape[0]:]
titanic_test['Survived'] = final_estimator.predict(X_test)
titanic_test.to_csv("C:\\Users\\Algorithmica\\Downloads\\submission.csv", columns=["PassengerId", "Survived"], index=False)
