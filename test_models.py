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
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC


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

svm.fit(iris,svm_classes, c=.5, epoch=1000)
print(svm.predict(iris))
'''

train_given = load_svmlight_file('labeledBow.feat', n_features = 89527)
test_given = load_svmlight_file('labeledBowtest.feat', n_features = 89527)
test_data = test_given[0]
test_y =test_given[1]
test_y = np.where(test_y<=4,-1,1)
data = train_given[0]
y = train_given[1]

y =np.where(y<=4,-1,1)

svd = TruncatedSVD(n_components= 10, n_iter=7, random_state=42)
'''
-- PCA does not support sparse input

pca = PCA(n_components = 100)
pca.fit(data)
new_data2 = pca.fit_transform(data)
new_test_data2 = pca.fit_transform(test_data)
print('pca dimension fit')
'''

svd.fit(data)
print('SVD fit')

new_data = svd.fit_transform(data)
new_test_data = svd.fit_transform(test_data)
print('dimensions transformed')
print(new_data.shape)


lda_model = models.LDA()
s_lda = LinearDiscriminantAnalysis(solver = 'eigen')

s_lda.fit(new_data,y)
s_lda.fit(new_data2,y)

lda_model.fit(new_data,y)
print('model fit')
pred = lda_model.predict(new_test_data)
print(lda_model.score(np.array(pred),test_y))
print(s_lda.score(new_test_data,test_y))

clf = LinearSVC(random_state=0, tol=1e-5)

clf.fit(new_data,y)

print(clf.score(new_test_data,test_y))

# s_lda.fit(new_data2,y)
# lda_model.fit(new_data2,y)
# clf.fit(new_data2,y)

# pred = lda_model.predict(new_test_data2)
# print(lda_model.score(np.array(pred),test_y))
# print(s_lda.score(new_test_data2,test_y)) 
# print(clf.score(new_test_data2,test_y))
