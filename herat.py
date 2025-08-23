import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix,classification_report


df=pd.read_csv("heart.csv")
print(df)

print(df.isna().sum())
df=df.dropna()
print(df)
print(df.duplicated().sum())

lab=LabelEncoder()
df["Gender"]=lab.fit_transform(df["Gender"])
df["ChestPainType"]=lab.fit_transform(df["ChestPainType"])
df["RestingECG"]=lab.fit_transform(df["RestingECG"])
df["ExerciseAngina"]=lab.fit_transform(df["ExerciseAngina"])
df["ST_Slope"]=lab.fit_transform(df["ST_Slope"])

print(df)

cor=df.corr()
print(cor)
cor_label=cor["HeartDisease"].abs()
print(cor_label)
c=0.06
sel=cor_label[cor_label>c].index
print(sel)

df=df.loc[:,sel]
print(df)

x=df.drop("HeartDisease",axis=1)
y=df["HeartDisease"]

print(y.value_counts())


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)


model=RandomForestClassifier(n_estimators=100)
model.fit(xtrain,ytrain)

import joblib
joblib.dump(model,"new5.joblib")

ypredict=model.predict(xtest)

acc=accuracy_score(ytest,ypredict)
print("accuracy score",acc)

rec=recall_score(ytest,ypredict)
print("recall score",rec)
pre=precision_score(ytest,ypredict)
print("precision score",pre)
f=f1_score(ytest,ypredict)
print("f1 score",f)
conf=confusion_matrix(ytest,ypredict)
print("confusion matrix")
print(conf)
clas=classification_report(ytest,ypredict)
print(clas)


