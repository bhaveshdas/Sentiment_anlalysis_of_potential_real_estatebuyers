import numpy as np 
import pandas as pd
import openpyxl
from sklearn.model_selection import train_test_split 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.classify import SklearnClassifier
from sklearn.metrics import confusion_matrix
import re
import string
import seaborn as sns

data = pd.read_excel("ANP Balewadi.xlsx", engine = "openpyxl")

#Preprocessing
data = data[['Remark', 'Sentiment']]
data = data.dropna()
data = data[data.Sentiment != 'Neutral' ]

#train-test split 
train, test = train_test_split(data,test_size = 0.2)
train_pos = train[ train['Sentiment'] == 'Interested']
train_pos = train_pos['Remark']
train_neg = train[ train['Sentiment'] == 'Not interested']
train_neg = train_neg['Remark']



#function for changing the text to lower case
text = []
stopwords_set = set(stopwords.words("english"))

for index, row in train.iterrows():
    words_filtered = [e.lower() for e in row.Remark.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    text.append((words_without_stopwords, row.Sentiment))

test_pos = test[ test['Sentiment'] == 'Interested']
test_pos = test_pos['Remark']
test_neg = test[ test['Sentiment'] == 'Not interested']
test_neg = test_neg['Remark']
def get_words_in_texts(text):
    all = []
    for (words, sentiment) in text:
        all.extend(words)
    return all

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features
w_features = get_word_features(get_words_in_texts(text))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

training_set = nltk.classify.apply_features(extract_features,text)
classifier = nltk.NaiveBayesClassifier.train(training_set)

neg_cnt = 0  
pos_cnt = 0  
pos_neg_cnt = 0
neg_pos_cnt = 0
for obj in test_neg: 
    res =  classifier.classify(extract_features(obj.split()))
    if(res == 'Not interested'): 
        neg_cnt = neg_cnt + 1
    if(res == 'Interested'):
        neg_pos_cnt = neg_pos_cnt+1
for obj in test_pos: 
    res =  classifier.classify(extract_features(obj.split()))
    if(res == 'Interested'): 
        pos_cnt = pos_cnt + 1
    if(res == 'Not interested'):
        pos_neg_cnt = pos_neg_cnt+1
precision = pos_cnt/(pos_cnt+pos_neg_cnt)
recall = pos_cnt/(pos_cnt+neg_pos_cnt)
print('[Not interested]: %s/%s '  %(neg_cnt, len(test_neg)))        
print('[Interested]: %s/%s '  % (pos_cnt, len(test_pos)))
print('Accuracy =  ', (neg_cnt+pos_cnt)/(len(test_neg) + len(test_pos)))
print("TP= ", pos_cnt, " FP= ", pos_neg_cnt)
print("TN= ", neg_cnt, "FN= ", neg_pos_cnt)
print('Precision = ', precision, 'Recall = ', recall)
print('F1 score = ', 2*precision*recall/(precision+recall))