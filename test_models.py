import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import datasets
from sklearn import svm
from matplotlib import pyplot as plt
import models
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
