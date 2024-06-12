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
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv(r"D:\coding journey\aiml\python\task\data set of ML project\LInear regression\insurance.csv")
x=df.iloc[:, :-1].values
y=df.iloc[:, -1].values
le=LabelEncoder()
x[:, 1]=le.fit_transform(x[:, 1])
le2=LabelEncoder()
x[:, 4]=le.fit_transform(x[:, 4])

ct=ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[5])],remainder="passthrough")
x=ct.fit_transform(x)
print(x)
x_opt=x;
x_opt=np.append(np.ones((x.shape[0],1)).astype(int),x,axis=1)
x_opt=np.asarray(x_opt,dtype=np.float64)
print(x_opt.shape)
x_opt=x_opt[:,[col for col in range(x_opt.shape[1]) if col not in [1]]]
print(x_opt.shape)
y=np.asarray(y,dtype=np.float64)
def pvalue_checker(x,y, sl=0.05):
    num_var=x.shape[1]
    for i in range(num_var):
        ols = OLS(endog=y, exog=x).fit()
        max_pvalue=max(ols.pvalues)
        if max_pvalue>sl:
            max_indx=np.argmax(ols.pvalues)
            x = x[:, [col for col in range(x.shape[1]) if col not in [max_indx]]]
            return x

x_opt=pvalue_checker(x_opt,y)
ols=OLS(endog=y,exog=x_opt).fit()
print(ols.summary())

x_train, x_test, y_train, y_test = train_test_split(x_opt, y, test_size=0.2, random_state=1)

lr=RandomForestRegressor(n_estimators=500,random_state=0)

lr.fit(x_train,y_train)
y_predict=lr.predict(x_test)

print(r2_score(y_predict,y_test))





