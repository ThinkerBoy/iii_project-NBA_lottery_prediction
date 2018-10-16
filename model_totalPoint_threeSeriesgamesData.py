
# coding: utf-8

import pandas as pd
from sqlalchemy import create_engine

#Data load
#Gambling data from MySQL
engine = create_engine("mysql+pymysql://cb102:iiicb102@10.120.23.10:3306")
engine.execute("USE nba;")
#賭盤數據-總得分
gambling_data = pd.read_sql('''SELECT m.*, g.utcMillis AS SeansonYear, g.awayScore, g.homeScore, 
IF( awayScore+homeScore > totalPoint, 1, 0) AS gamblingTotal, IF(homeScore+m.spread > awayScore, 1, 0) AS gamblingWin,IF(homeScore > awayScore, 1, 0) AS gameWin
FROM nba.matchup AS m LEFT JOIN nba.gameresult AS g ON m.gameId = g.gameId;''',engine)
engine.dispose()

#player data from csv file
player_data = pd.read_csv('./recentlythreeSeriesgames.csv')
player_data = player_data.drop('Unnamed: 0',axis=1)
player_data = player_data.sort_values(by='gameId')
player_data['gameId'] = player_data['gameId'].astype(int).astype(str).str.pad(10,fillchar='0')
player_data['playerId'] = player_data['playerId'].astype(int).astype(str)
#index data from csv file
record_index_df = pd.read_csv('./record_index_recentlythreeSeriesgames.csv')
record_index_df = record_index_df.drop('Unnamed: 0',axis=1)
record_index_df['gameId'] = record_index_df['gameId'].astype(int).astype(str).str.pad(10,fillchar='0')
record_index_df['playerId'] = record_index_df['playerId'].astype(int).astype(str)
record_index_df['belongTeamId'] = record_index_df['belongTeamId'].astype(int).astype(str)

#計算PIE index
def PIE_index(row):
    P_index = (row['points'] + row['fgm'] +row['ftm'] - row['fga'] - row['fta'] +row['defRebs'] + (.5 * row['offRebs']) + row['assists'] + row['steals'] + (.5 * row['blocks']) - row['fouls'] - row['turnovers']) 
    P_index = P_index
    return  P_index

#Data combime & ETL for TotalPoints
model_data_cols = ['gameId', 'assists', 'blocks','defRebs', 'fga', 'fgm', 'fouls', 'fta', 'ftm','mins', 'offRebs', 'points', 
                    'rebs', 'steals', 'tpa', 'tpm','turnovers']
model_data_x = player_data[model_data_cols].groupby(by='gameId').sum().reset_index()
model_data_x1 = pd.merge(gambling_data,model_data_x,on='gameId',how='inner')
model_data_x1['PIE'] = model_data_x1.apply(PIE_index,axis=1)
model_data = pd.merge(model_data_x1,record_index_df.groupby(by=['gameId']).sum().reset_index(),on='gameId',how='inner')
model_data['SeansonYear'] = model_data['SeansonYear'].astype(str)
#filter date
model_data_filter = model_data[(model_data['totalPoint'] <= 222) & (model_data['totalPoint'] >= 192)]

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


#切割training data & testing data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

model_x_cols = ['assists', 'blocks','defRebs', 'fga', 'fgm', 'fouls', 'fta', 'ftm','mins', 'offRebs', 'points', 
                    'rebs', 'steals', 'tpa', 'tpm','turnovers','PIE', 'Pace', 'PProd', 'Offensive_poss', 'DRtg', 'OWS',
       'DWS', 'WS', 'PER']
model_y_cols = ['gamblingTotal']
model_data_x = model_data_filter[model_x_cols]
model_data_y = model_data_filter[model_y_cols]

#70%做training, 30%做testing
X_train, X_test, y_train, y_test = train_test_split(model_data_x, model_data_y, test_size=0.3, random_state=102)

#Cross validation
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

model_x_cols = ['assists', 'blocks','defRebs', 'fga', 'fgm', 'fouls', 'fta', 'ftm','mins', 'offRebs', 'points', 
                    'rebs', 'steals', 'tpa', 'tpm','turnovers','PIE', 'Pace', 'PProd', 'Offensive_poss', 'DRtg', 'OWS',
       'DWS', 'WS', 'PER']
model_y_cols = ['gamblingTotal']
model_data_x = model_data_filter[model_x_cols]
model_data_y = model_data_filter[model_y_cols].values.ravel()

#building model
#決策樹 CART
from sklearn import tree
from sklearn.metrics import confusion_matrix

CART_model = tree.DecisionTreeClassifier()
scores = cross_val_score(CART_model, model_data_x, model_data_y, cv=5)
print("Accuracy of CART_model: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#SVM
from sklearn import svm

SVM_model = svm.SVC()  
scores = cross_val_score(SVM_model, model_data_x, model_data_y, cv=5)
print("Accuracy of SVM_model: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#Logistic regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression().fit(X_train, y_train.values.ravel())
scores = cross_val_score(logreg, model_data_x, model_data_y, cv=5)
print("Accuracy of logreg: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#XGBoost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
# fit model no training data
XGB_model = XGBClassifier()
scores = cross_val_score(XGB_model, model_data_x, model_data_y, cv=5)
print("Accuracy of XGB_model: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

from sklearn import ensemble, metrics
# 建立 random forest 模型
forest = ensemble.RandomForestClassifier(n_estimators = 100)
scores = cross_val_score(forest, model_data_x, model_data_y, cv=5)
print("Accuracy of forest: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#optimal parameter for Logistic Regression
from sklearn.metrics import confusion_matrix
#optimal parameter for logistic regression
from sklearn.linear_model import LogisticRegressionCV

logreg_CV = LogisticRegressionCV(cv=5).fit(X_train, y_train.values.ravel())
print('Accuracy of logistic regression classifier on test set: {:.4f}'.format(logreg_CV.score(X_test, y_test)))
print('parameter:', logreg_CV)
print('confusion matrix result:\n',  confusion_matrix(y_test, logreg_CV.predict(X_test)))
     


import pickle
# save the classifier
s = 'Logistic_TotalPoint' + '.' + 'pickle'
with open(s, 'wb') as f:
    pickle.dump(logreg_CV, f) 

