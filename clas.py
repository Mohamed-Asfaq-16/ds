import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as py

df=pd.read_csv("data.csv")
sns.histplot(df["diagnosis"])
plt.show()

sns.scatterplot(x='radius_mean',y='texture_mean',hue='diagnosis',data=df)
plt.show()

sns.boxplot(x="radius_mean",y="texture_mean",data=df)
plt.show()


df = df.drop(['id','Unnamed: 32'], axis=1)
print(df.isnull().sum())
df.fillna(df.mean(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})

x=df.drop("diagnosis",axis=1)
y=df['diagnosis']

corr=df.corr(numeric_only=True)
sns.heatmap(corr,annot=True,cmap="coolwarm")
plt.show()

x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42
)

scaler=StandardScaler()

x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

lr = LogisticRegression(max_iter=1000)

model=lr.fit(x_train,y_train)
ans=model.predict(x_test)

acc=accuracy_score(y_test,ans)
print(acc)





