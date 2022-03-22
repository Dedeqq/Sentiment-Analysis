#%% Liblaries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.tokenize import word_tokenize
from os import listdir
import string
from nltk.corpus import stopwords

#%% Loading the data
reviews = pd.DataFrame()
for f2 in ('pos', 'neg'):
    for f1 in ('test', 'train'):
        path="aclImdb/"+f1+"/"+f2
        print(path)
        for filename in listdir(path):
            specific_path = path + '/' + filename
            with open(specific_path,"r", encoding="utf8") as file:
                txt=file.read()
            reviews = reviews.append([txt], ignore_index=True)
reviews.columns=["review"] 
# reviews.iloc[0:25000] positive reviews, reviews.iloc[25000:50000] negative reviews

#%% Text processing
stop_words = set(stopwords.words('english'))
add={"movie", "film", "br", "one"} #most popular words in both positive and negative reviews
stop_words.update(add)
def text_process(txt):
    # remove punctuation
    txt=txt.translate(str.maketrans('', '', string.punctuation))
    txt=word_tokenize(txt)
    # remove stopwords
    txt=[word.lower() for word in txt if word.lower() not in stop_words]
    txt=" ".join(txt)
    return txt
    
reviews["processed"]=reviews["review"].apply(text_process)


#%% Plots
pos=pd.Series(' '.join(reviews.processed.iloc[:25000]).split()).value_counts()[:50]
plt.figure(figsize=(20, 10))
sns.barplot(pos.iloc[:20].index, pos.iloc[:20].values)
plt.title("Positive reviews",size=40)
plt.xlabel("Word",size=30)
plt.ylabel("Word Count",size=30)
plt.savefig("pos.jpg")
plt.show()

neg=pd.Series(' '.join(reviews.processed.iloc[25000:]).split()).value_counts()[:50]
plt.figure(figsize=(20, 10))
sns.barplot(neg.iloc[:20].index, neg.iloc[:20].values)
plt.title("Negative reviews",size=40)
plt.xlabel("Word",size=30)
plt.ylabel("Word Count",size=30)
plt.savefig("neg.jpg")
plt.show()

#%% Further analysis
only_positive=[word for word in pos.index if word not in neg.index]
only_negative=[word for word in neg.index if word not in pos.index]
print(f"Most common words from positive reviews that are not in negative revievs: \n {only_positive}")
print(f"Most common words from negative reviews that are not in positive revievs: \n {only_negative}")

"""Some of the most popular words overlap, however some words like "bad"
 are very popular only in one class."""