# %% Liblaries
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from os import listdir
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

# %% Loading the data
df = pd.DataFrame()
labels = {'pos': 1, 'neg': 0}
for f1 in ('test', 'train'):
    for f2 in ('pos', 'neg'):
        path = "aclImdb/" + f1 + "/" + f2
        print(path)
        for filename in listdir(path):
            specific_path = path + '/' + filename
            with open(specific_path, "r", encoding="utf8") as file:
                txt = file.read()
            df = df.append([[txt, labels[f2]]], ignore_index=True)
df.columns = ["review", "rating"]

# %% Text processing
stop_words = set(stopwords.words())
add = {"movie", "film", "br", "one"}  # most popular words in both positive and negative reviews
stop_words.update(add)

def text_process(txt):
    # remove punctuation
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    # remove stopwords
    txt = word_tokenize(txt)
    txt = [word.lower() for word in txt if word.lower() not in stop_words]
    txt = " ".join(txt)
    return txt

df["processed"] = df["review"].apply(text_process)
print(df)

# create bag of words
count_vector = CountVectorizer()
features = count_vector.fit_transform(df["processed"])

# %% Spliting the data
X_train = features[:25000]
X_test = features[25000:]
y_test = df.loc[25000:, 'rating']
y_train = df.loc[:24999, 'rating']

# %% Naive Bayes Model
from sklearn.naive_bayes import MultinomialNB
print("Naive Bayes")
param_grid = {'alpha': (1, 0.09, 0.1, 0.11, 0.012, 0.0001, 0.00001)}
naive_bayes = GridSearchCV(MultinomialNB(), param_grid)
naive_bayes.fit(X_train, y_train)
#print(naive_bayes.best_estimator_) #best estimator alpha=1
y_predicted = naive_bayes.predict(X_test)
print(accuracy_score(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted))
with open('results.txt', 'a') as file:
    file.write(
        f"""Naive Bayes \n Accuracy: {str(accuracy_score(y_test, y_predicted))} \n Confusion atrix: \n{str(confusion_matrix(y_test, y_predicted))} \n \n""")

# %% Logistic Regression
"""After testing different parameters with Grid Search, it turns out C=0.0189 gives best results for this model."""
from sklearn.linear_model import LogisticRegression
print("Logistic Regression")
logreg = LogisticRegression(max_iter=200, C=0.0189)
logreg.fit(X_train, y_train)
y_predicted = logreg.predict(X_test)
print(accuracy_score(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted))
with open('results.txt', 'a') as file:
    file.write(
        f"""Logistic Regression \n Accuracy: {str(accuracy_score(y_test, y_predicted))} \n Confusion atrix: \n{str(confusion_matrix(y_test, y_predicted))} \n \n""")


# %% Random Forest
"""After testing different parameters with Grid Search, it turns out defeault parameters give best results for this model."""
from sklearn.ensemble import RandomForestClassifier
print("Random Forest")
forest_clf = RandomForestClassifier()
forest_clf.fit(X_train, y_train)
y_predicted = forest_clf.predict(X_test)
print(accuracy_score(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted))
with open('results.txt', 'a') as file:
    file.write(
        f"""Random Forest \n Accuracy: {str(accuracy_score(y_test, y_predicted))} \n Confusion atrix: \n{str(confusion_matrix(y_test, y_predicted))} \n \n""")

# %% Ensemble mmodels
print("Ensembling models")
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators=[('n_b', naive_bayes), ('l_r', logreg), ('r_f', forest_clf)],voting='hard')
voting_clf.fit(X_train, y_train)
y_predicted = voting_clf.predict(X_test)
print(accuracy_score(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted))
with open('results.txt', 'a') as file:
    file.write(
        f"""Ensembling models \n Accuracy: {str(accuracy_score(y_test, y_predicted))} \n Confusion atrix: \n{str(confusion_matrix(y_test, y_predicted))} \n \n""")

#%% User review
question=input("Do you want me to check weather your review is positive or negative (y/n)? ")
while question=="y":
    custom_review=input("Type your review here: ")
    custom_review = text_process(custom_review)
    if voting_clf.predict(count_vector.transform([custom_review]))==1:
        print("Your review is positive.")
    else: print("Your review is negative.")

    question =input("Do you want to try again (y/n)? ")
