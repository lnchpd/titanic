import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#READ DATA IN FROM FILE
data = pd.read_csv('titanic_data.csv')

"""data key

VARIABLE        DEFINITION      KEY

PassengerId     index           0...
Survived        survival        0 = No, 1 = Yes
Pclass          ticket class    1 = 1st, 2 = 2nd, 3 = 3rd
Name            sur, tit fir
Sex             gender          'male'/'female'
Age             age             in years (float; partial for <1)
SibSp           # sibs/spouses
Parch	        # of par/chil
Ticket          ticket number
Fare	        passenger fare
Cabin	        cabin number
Embarked	    port             C = Cherbourg, Q = Queenstown, S = Southampton
"""

#convert integers for count of relatives to boolean
data['SibSp'] = np.where(data['SibSp']>0, True, False)
data['Parch'] = np.where(data['Parch']>0, True, False)

#drop unnecessary fields
data.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)

#detemine whether inputs are categorical or continuous
categorical_inputs = ['Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch']
continuous_inputs = ['Age', 'Fare']

#create dummy variables for each categorical input, and drop original columns
for cur_input in categorical_inputs:
    data = pd.concat([data,pd.get_dummies(data[cur_input], prefix=cur_input)], axis=1)
    data.drop(cur_input, axis=1, inplace=True)

#store transormations to continuous inputs
continuous_input_adj = []

#standardize continuous inputs to z-scores
for cur_input in continuous_inputs:
    mean = data[cur_input].mean()
    std = data[cur_input].std()
    continuous_input_adj.append([cur_input, mean, std])
    data[cur_input] = (data[cur_input] - mean) / std

#Drop missing data
data = data.dropna()

#separate targets from inputs (targeting survival!)
inputs, test_inputs, targets, test_targets = \
    train_test_split(data.drop('Survived', axis=1), data['Survived'], test_size = 0.15)
