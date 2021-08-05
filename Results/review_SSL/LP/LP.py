import numpy as np
import time
import sys
import os
sys.path.insert(1,"../../../")
from sklearn.semi_supervised import LabelPropagation
from Utils.data_loader import load_joint_data_as_matrix


if __name__ == "__main__":
    num_samples = 100
    dataset = "MNIST"
    X_train, y_train, X_validation, y_validation, X_test, y_test = load_joint_data_as_matrix(num_samples,dataset=dataset,percentage_unlabelled_set=0.01)
    

    n = X_train.shape[0]
    X_total = np.concatenate((X_train,X_test),axis=0)
    y_test_removed = np.copy(y_test)
    y_test_removed[:] = -1
    X_total = np.concatenate((X_train,X_test),axis=0)
    y_total = np.concatenate((y_train,y_test_removed),axis=0)
    print("Training")
    model = LabelPropagation(gamma=.25)
    t0 =  time.time()
    model.fit(X_total,y_total)
    t = time.time() - t0
    y_predicted = model.transduction_[n:]
    acc = np.sum(y_predicted == y_test)/y_test.shape[0]
    #acc = model.score(X_test,y_test)
    print("final accuracy label propagation",acc)
    print("running time", t)
