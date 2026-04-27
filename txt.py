import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

df=pd.read_csv("IMDBDataset.csv")

df=df[['review','sentiment']]

df.columns=['text','label']

df['label']=df['label'].map({'negative':0,'positive':1})

tfidf=TfidfVectorizer()

x=tfidf.fit_transform(df['text'])
y=df['label']

x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42
)

model = MultinomialNB()
model.fit(x_train,y_train)

pred = model.predict(x_test)

acc=accuracy_score(y_test,pred)

print(acc)

print(classification_report(y_test,pred))

print(confusion_matrix(y_test,pred))

i=input("Enter a review: ")

pre=tfidf.transform([i])

ans=model.predict(pre)

print("The review is:", "positive" if ans[0] == 1 else "negative")
