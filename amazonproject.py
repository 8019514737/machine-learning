# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 09:22:19 2018

@author: DIVYA SREE
"""

#importing the libraries
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import string
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

#importing the dataset
con=sqlite3.connect("C:/Users/DIVYA SREE/Desktop/Machine_Learning_A-Z_Template_Folder/amazon/database.sqlite")
messages=pd.read_sql_query('SELECT Score, Summary FROM Reviews WHERE Score!=3 ',con)
def partition(x):
    if x<3:
        return 'negative'
    return 'positive'
Score=messages['Score']
Score=Score.map(partition)
Summary=messages['Summary']

#initializing similar variable as messages
tmp=messages
tmp['Score']=tmp['Score'].map(partition)
print(tmp.head(20))

#CLeaning of data
#includes Lowering,Stemming,Tockenization,stopwords removal

from nltk.corpus import stopwords

#Tockenization
intab=string.punctuation
outtab=' '*len(string.punctuation)
trantab=str.maketrans(intab,outtab)

#Stemming
stemmer=PorterStemmer()
def stem_tokens(tokens,stemmer):
    Stemmed=[]
    for item in tokens:
     Stemmed.append(stemmer.stem(item))
    return Stemmed

#cleaning whole summary
corpus=[]
for line in Summary:
    line=line.lower()
    line=line.translate(trantab)
    tokens=nltk.word_tokenize(line)
    stems=stem_tokens(tokens,stemmer)
    stems=' '.join(stems)
    corpus.append(stems)
count_vect=CountVectorizer()
summary=count_vect.fit_transform(corpus)

#Splitting into training and test set  
X_train,X_test,y_train,y_test=train_test_split(summary,Score,test_size=0.2,random_state=42)

#initializing a dictionary
Prediction=dict()

#Applying Multinomial Naive Bayes learning model
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB().fit(X_train,y_train)
Prediction['Multinomial']=model.predict(X_test)



#Applying Bernouli Naive Bayes learning model
from sklearn.naive_bayes import BernoulliNB
model=BernoulliNB().fit(X_train,y_train)
Prediction['Bernoulli']=model.predict(X_test)


#Applying Logistic regression learning method
from sklearn import linear_model
logreg=linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train,y_train)
Prediction['LOgistic']=logreg.predict(X_test)  


#Results
#Plotting receiver operating characteristic curve(roc curve) 
def format(x):
    if x=='negative':
        return 0
    return 1
vfunc=np.vectorize(format)
cmp=0
colours=['b','g','y','m','k']
for model,predicted in Prediction.items():
    false_positive_rate,true_positive_rate,thresholds=roc_curve(y_test.map(format),vfunc(predicted))
    roc_auc=auc(false_positive_rate,true_positive_rate)
    plt.plot(false_positive_rate,true_positive_rate,colours[cmp],label='%s:AUC %0.2f'%(model,roc_auc))
    cmp+=1
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('true positive rate')
plt.xlabel('false positive rate')
plt.show()
plt.title('Classifiers comparison with ROC')


#visualization of Logistic Regression
print(metrics.classification_report(y_test,Prediction['LOgistic'],target_names=["positive","negative",]))
def plot_confusion_matrix(cm,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(set(Score)))
    plt.xticks(tick_marks,set(Score),rotation=45)
    plt.yticks(tick_marks,set(Score))
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Prediction Label')
    
    
#Compute Confusion matrix
cm=confusion_matrix(y_test,Prediction['LOgistic'])
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cm)
cm_normalized=cm.astype('float')
cm.sum(axis=1)[:,np.newaxis]
plt.figure()
plot_confusion_matrix(cm_normalized,title='Normalized Confusion Matrix')
plt.show()
    
 
    

        


    

    




