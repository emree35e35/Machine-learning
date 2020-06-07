import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Users/emree/Desktop/verianalizi/corona.csv",sep=";",header=0)
#csvdeki kolonların tanımlanması
gun = data.iloc[:,0].values.reshape(-1,1)
vaka = data.iloc[:,1].values.reshape(-1,1)
olum=data.iloc[:,2].values.reshape(-1,1)

#Gelecek tahminlemesi için istediğim aralık 1 ile 14gün
yeniguntahmin=np.arange(1,20).reshape(-1,1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(gun,vaka,test_size=0.33,random_state=0)

####Linear olarak Hesaplama
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
tahmin=lr.predict(yeniguntahmin)
#print("Lineer Regresyon",tahmin)

#Vaka hesaplama başlangıç
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
gunler_poly=poly_reg.fit_transform(gun)
lin_reg_vaka=LinearRegression()
lin_reg_vaka.fit(gunler_poly,vaka)
poly_sonuc_vaka=lin_reg_vaka.predict(poly_reg.fit_transform(yeniguntahmin))
print("Vaka Tahminleri:",poly_sonuc_vaka)
#Vaka hesaplama bitiş

#ölüm hesaplama başlangıç
poly=PolynomialFeatures(degree=2)
linear=LinearRegression()


x_poly=poly.fit_transform(gun)#X'i yani araba fiyatını polynoma cevırdık
linear.fit(x_poly,vaka)#Polinom olmuş fiyat ile normal hızı alıp fit ettik

y_headtahmincizgisi=linear.predict(x_poly)
#ölüm hesaplama bitiş





# Grafik şeklinde ekrana basmak için
plt.scatter(yeniguntahmin, poly_sonuc_vaka, color='red',label="VakaArtışTahmin")
plt.plot(yeniguntahmin, poly_sonuc_vaka, color='blue',label="VakaArtışTahmin")

plt.scatter(gun, olum, color='black',label="Ölüm")
plt.plot(gun, olum, color='yellow',label="ÖlümArtış")


plt.xlabel("Gün")
plt.ylabel("Vaka-Ölüm")
plt.legend()#Sol üste yazı atıyor
plt.show()