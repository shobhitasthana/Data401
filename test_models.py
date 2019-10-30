import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import datasets
from sklearn import svm
from matplotlib import pyplot as plt
import models
from models import *
import pandas as pd
import numpy as np
import os
from sklearn.datasets import load_svmlight_file

'''
iris,classes = datasets.load_iris(return_X_y=True)

print(classes)
bclasses = np.array( [int(x) for x in (classes != 0)])
print(bclasses)

svm_classes = []
for x in bclasses:
    if x == 0:
        svm_classes.append(-1)
    else:
        svm_classes.append(1)
print(svm_classes)

lda = models.LDA()

lda.fit(iris,svm_classes)


svm = models.SVM()

svm.fit(iris,svm_classes)
print(svm.predict(iris))
'''

train_given = load_svmlight_file('labeledBow.feat', n_features = 89527)
test_given = load_svmlight_file('labeledBowtest.feat', n_features = 89527)
test_data = test_given[0]
test_y =test_given[1]
test_y = np.where(test_y<=4,-1,1)
data = train_given[0][int(len(train_given[1])*0.45):int(len(train_given[1])*0.55),]
y = train_given[1][int(len(train_given[1])*0.45):int(len(train_given[1])*0.55),]

y =np.where(y<=4,-1,1)

lda_model = models.LDA()

lda_model.fit(data,y)

pred = lda_model.predict(test_data)
print(lda_model.score(np.array(pred),test_y))
