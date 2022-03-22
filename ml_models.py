#%% Liblaries
import pandas as pd
import string
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from os import listdir
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#%% Loading the data
df = pd.DataFrame()
labels = {'pos':1, 'neg':0}
for f1 in ('test', 'train'):
    for f2 in ('pos', 'neg'):
        path="aclImdb/"+f1+"/"+f2
        print(path)
        for filename in listdir(path):
            specific_path = path + '/' + filename
            with open(specific_path,"r", encoding="utf8") as file:
                txt=file.read()
            df = df.append([[txt, labels[f2]]], ignore_index=True)
df.columns=["review", "rating"]

#%% Text processing
stop_words = set(stopwords.words('english'))
add={"movie", "film", "br", "one"} #most popular words in both positive and negative reviews
stop_words.update(add)
def text_process(txt):
    # remove punctuation
    txt=txt.translate(str.maketrans('', '', string.punctuation))
    #
    txt=word_tokenize(txt)
    txt=[word.lower() for word in txt if word.lower() not in stop_words]
    txt=" ".join(txt)
    return txt
    
df["processed"]=df["review"].apply(text_process)
count_vector=CountVectorizer()
features=count_vector.fit_transform(df["processed"])

#%% Spliting the data
X_train = features[:25000]
X_test = features[25000:]
y_test = df.loc[25000:, 'rating']
y_train = df.loc[:24999, 'rating']

#%% Naive Bayes Model
print("Naive Bayes")
from sklearn.naive_bayes import MultinomialNB
naive_bayes=MultinomialNB()
naive_bayes.fit(X_train, y_train)
y_predicted=naive_bayes.predict(X_test)
print(accuracy_score(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted))
with open('results.txt', 'a') as file:
    file.write(f"""Naive Bayes \n Accuracy: {str(accuracy_score(y_test, y_predicted))} \n Confusion atrix: \n{str(confusion_matrix(y_test, y_predicted))} \n \n""") 

#%% Random Forest
from sklearn.ensemble import RandomForestClassifier
print("Random Forest")
clf=RandomForestClassifier()
clf.fit(X_train, y_train)
y_predicted=clf.predict(X_test)
print(accuracy_score(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted))
with open('results.txt', 'a') as file:
    file.write(f"""Random Forest \n Accuracy: {str(accuracy_score(y_test, y_predicted))} \n Confusion atrix: \n{str(confusion_matrix(y_test, y_predicted))} \n \n""") 

#%% Logistic Regression
from sklearn.linear_model import LogisticRegression
print("Logistic Regression")
logreg=LogisticRegression(max_iter=160)
logreg.fit(X_train, y_train)
y_predicted=logreg.predict(X_test)
print(accuracy_score(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted))
with open('results.txt', 'a') as file:
    file.write(f"""Logistic Regression \n Accuracy: {str(accuracy_score(y_test, y_predicted))} \n Confusion atrix: \n{str(confusion_matrix(y_test, y_predicted))} \n \n""") 

#%% SVM
from sklearn import svm
print("SVM")
support_vector=svm.SVC()
support_vector.fit(X_train, y_train)
y_predicted=support_vector.predict(X_test)
print(accuracy_score(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted))
with open('results.txt', 'a') as file:
    file.write(f"""Support Vector Machine \n Accuracy: {str(accuracy_score(y_test, y_predicted))} \n Confusion atrix: \n{str(confusion_matrix(y_test, y_predicted))} \n \n""") 
