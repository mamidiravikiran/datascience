import sys
sys.path.append("E:/")

import pandas as pd
import os
import common_utils as utils
from sklearn import preprocessing, neighbors, svm, linear_model, ensemble, pipeline, model_selection, feature_selection
import classification_utils as cutils
import seaborn as sns
import tpot
import numpy as np

dir = 'E:/'
titanic_train = pd.read_csv(os.path.join(dir, 'train.csv'))
print(titanic_train.shape)
print(titanic_train.info())

titanic_test = pd.read_csv(os.path.join(dir, 'test.csv'))
print(titanic_test.shape)
print(titanic_test.info())
titanic_test['Survived'] = None

titanic = pd.concat([titanic_train, titanic_test], ignore_index=True)
print(titanic.shape)
print(titanic.info())

def extract_title(name):
     return name.split(',')[1].split('.')[0].strip()
titanic['Title'] = titanic['Name'].map(extract_title)
sns.factorplot(x="Title", hue="Survived", data=titanic, kind="count", size=6)

#create family size feature from sibsp, parch
titanic['FamilySize'] = titanic['SibSp'] +  titanic['Parch'] + 1
sns.FacetGrid(titanic, hue="Survived",size=8).map(sns.kdeplot, "FamilySize").add_legend()

#create family group feature from family-size
def convert_familysize(size):
    if(size == 1): 
        return 'Single'
    elif(size <=5): 
        return 'Medium'
    else: 
        return 'Large'
titanic['FamilyGroup'] = titanic['FamilySize'].map(convert_familysize)
sns.factorplot(x="FamilyGroup", hue="Survived", data=titanic, kind="count", size=6)

sns.countplot(x='Cabin',data=titanic)
titanic['Cabin'] = titanic['Cabin'].fillna('U')

titanic = utils.drop_features(titanic, ['PassengerId', 'Name', 'Survived', 'Ticket'])

#type casting
utils.cast_to_cat(titanic, ['Sex', 'Pclass', 'Embarked', 'Title', 'FamilyGroup', 'Cabin'])

cat_features = utils.get_categorical_features(titanic)
print(cat_features)
cont_features = utils.get_continuous_features(titanic)
print(cont_features)

#handle missing data(imputation)
cat_imputers = utils.get_categorical_imputers(titanic, cat_features)
titanic[cat_features] = cat_imputers.transform(titanic[cat_features])
cont_imputers = utils.get_continuous_imputers(titanic, cont_features)
titanic[cont_features] = cont_imputers.transform(titanic[cont_features])

#one hot encoding
titanic = utils.ohe(titanic, cat_features)

#scale the data
scaler = preprocessing.StandardScaler()
tmp = scaler.fit_transform(titanic)
titanic = pd.DataFrame(tmp, columns=titanic.columns)

titanic_train1 = titanic[:titanic_train.shape[0]]
y_train = titanic_train['Survived']

rf_estimator = ensemble.RandomForestClassifier()
rf_grid  = {'max_depth':list(range(1,9)), 'n_estimators':list(range(1,300,100)) }
rf_final_estimator = cutils.grid_search_best_model(rf_estimator, rf_grid, titanic_train1, y_train)
X_train = utils.select_features(rf_final_estimator, titanic_train1, threshold='mean')

params = {'max_depth': np.arange(1,200,1),
          'learning_rate': np.arange(0.0001,0.1,0.0001),
          'n_estimators': np.arange(1,200,1),
          'nthread':[6],
          'gamma':np.arange(0.00001,0.1,0.00001),
          'subsample':np.arange(0.1,2,0.1),
          'reg_lambda': np.arange(0.1,200,1),
          'reg_alpha': np.arange(1,200,1),
          'min_child_weight': np.arange(1,200,1),
         'colsample_bytree': np.arange(0.1,2,0.1),
          'colsample_bylevel': np.arange(0.1,2,0.1)
         }


tpot_estimator = tpot.TPOTClassifier(generations=5, population_size=100,
                                offspring_size=250,
                                verbosity=2, early_stop=3,                                
                                config_dict={'xgboost.XGBClassifier': params},
                                cv = 5, scoring = 'accuracy')
tpot_estimator.fit(X_train, y_train)
print(tpot_estimator.score(X_train, y_train))
print(tpot_estimator.fitted_pipeline_)
print(tpot_estimator._optimized_pipeline)
print(tpot_estimator.evaluated_individuals_)

titanic_test1 = titanic[titanic_train.shape[0]:]
X_test = utils.select_features(rf_final_estimator, titanic_test1, threshold='mean')

titanic_test['Survived'] = tpot_estimator.predict(X_test)
titanic_test.to_csv(os.path.join(dir, 'submission.csv'), columns=['PassengerId', 'Survived'], index=False)