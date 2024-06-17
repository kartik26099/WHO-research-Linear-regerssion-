import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv(r"D:\coding journey\aiml\python\task\data set of ML project\LInear regression\uber.csv")
df=df.dropna()


x=df.iloc[:, [col for col in range(df.shape[1]) if col not in [0,2]]]

x = x.drop(columns=['pickup_datetime',"key"])

print(x.columns)
df_fianl=x;
y=df.iloc[:, 2]
df_fianl["target"]=y
#sns.pairplot(df_fianl,hue="target")
#plt.show()
from sklearn.preprocessing import StandardScaler
x=x.values
y=y.values
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

sc1=StandardScaler()

x_train=sc1.fit_transform(x_train)
x_test=sc1.transform(x_test)

sc2=StandardScaler()

y_train=sc2.fit_transform(y_train.reshape(-1,1))
y_test=sc2.transform(y_test.reshape(-1,1))

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
model ={"linear regression":LinearRegression(),
        "decision tree": DecisionTreeRegressor(min_samples_split=10),
        "random forest":RandomForestRegressor(n_estimators=50),
        "SVM":SVR(kernel="linear")}
accuracy={}
for name,models in model.items():
        regression=models.fit(x_train,y_train)
        y_predict=models.predict(x_test)
        accuracy[name]=r2_score(y_test,y_predict)

print(accuracy)