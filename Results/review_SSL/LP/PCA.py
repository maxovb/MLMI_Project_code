import numpy as np
import time
import sys
import os
sys.path.insert(1,"../../../")
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from Utils.data_loader import load_joint_data_as_matrix


if __name__ == "__main__":
    num_samples = 100
    dataset = "MNIST"
    X_train, y_train, X_validation, y_validation, X_test, y_test = load_joint_data_as_matrix(num_samples,dataset=dataset)
    
    for n_components in [10,20,50,100,200,784]:

        t0 = time.time()
        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        t = time.time() - t0
        print("Dim reduction time",t)

        X_train_labelled = X_train[y_train!= -1]
        X_train_labelled_mapped = pca.transform(X_train_labelled)
        y_train_labelled = y_train[y_train != -1]

        model = SVC()
        model.fit(X_train_labelled_mapped,y_train_labelled)

        t0 = time.time()
        X_test_mapped = pca.transform(X_test)
        acc = model.score(X_test_mapped,y_test)
        t = time.time() - t0
        print("Training time", t)

        print("Accuracy PCA + SVM", "n_components",n_components, ":", acc)