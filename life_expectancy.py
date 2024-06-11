import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from statsmodels.api import OLS
from sklearn.impute import SimpleImputer
from  sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import stats

df=pd.read_csv(r"D:\coding journey\aiml\python\task\data set of ML project\LInear regression\Life Expectancy Data (1).csv")
x=df.iloc[:, [0,1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
y=df.iloc[:, [3,4,5]]
print(x.isna().sum(axis=0))
print("\n",y.isna().sum(axis=0))
x=x.values
y=y.values
lr=LabelEncoder()
x[:, 0]=lr.fit_transform(x[:, 0])
sp=SimpleImputer(missing_values=np.nan,strategy="mean")
x[:, [3,5,7,9,10,11,13,14,15,16,17,18]]=sp.fit_transform(x[:, [3,5,7,9,10,11,13,14,15,16,17,18]])
y[:, [0,1]]=sp.fit_transform(y[:, [0,1]])

ct=ColumnTransformer(transformers=[("encder",OneHotEncoder(),[1,2])],remainder="passthrough")
x=np.array(ct.fit_transform(x))
print(x)




St=StandardScaler()
x=St.fit_transform(x)
st1=StandardScaler()
y=st1.fit_transform(y)
regressor=RandomForestRegressor(n_estimators=1000,random_state=0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
regressor.fit(x_train,y_train)
y_predict=regressor.predict(x_test)
print(r2_score(y_predict,y_test))
y_predict=st1.inverse_transform(y_predict)
print(y_predict,"\n")
y_test=st1.inverse_transform(y_test)
print(y_test)




