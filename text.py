# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 18:48:48 2022

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.datasets import fetch_20newsgroups

data=fetch_20newsgroups()
print(data.target_names) 

categories=['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
train=fetch_20newsgroups(subset='train',categories=categories)
test=fetch_20newsgroups(subset='test',categories=categories)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model=make_pipeline(CountVectorizer(),MultinomialNB())
model.fit(train.data,train.target)
labels=model.predict(test.data)

from sklearn.metrics import confusion_matrix
mat=confusion_matrix(test.target,labels)
sns.heatmap(mat.T,square=True ,annot=True,fmt='d',cbar=False
            ,xticklabels=train.target_names
            ,yticklabels=train.target_names)

plt.xlabel('true label')
plt.ylabel('predicted label')

def predict_category(s,train=train,model=model):
    pred=model.predict([s])
    return train.target_names[pred[0]]

print(predict_category('president of india'))