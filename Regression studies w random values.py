import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataframe={'haftalar':np.arange(1,21), 'degerler':np.arange(1,41,2)}
df=pd.DataFrame(data=dataframe)
haftalar=df[['haftalar']]
degerler=df[['degerler']]
print(df)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(haftalar,degerler,test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
tahmin=lr.predict(np.arange(1,30).reshape(-1,1))#21.ci haftayı tahmın edıyor
print("Lineer Regresyon",tahmin)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
haftalar_poly=poly_reg.fit_transform(haftalar)
lin_reg2=LinearRegression()
lin_reg2.fit(haftalar_poly,degerler)
poly_sonuc=lin_reg2.predict(poly_reg.fit_transform(np.array(21).reshape(1,-1)))
print("Polynom regresion",poly_sonuc)

from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(haftalar,degerler)
print("Random forest regressor",rf_reg.predict(np.array(21).reshape(1,1)))