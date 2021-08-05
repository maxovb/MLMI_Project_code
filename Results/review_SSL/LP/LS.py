import numpy as np
import time
import sys
import os
sys.path.insert(1,"../../../")
from sklearn.semi_supervised import LabelSpreading
from Utils.data_loader import load_joint_data_as_matrix


if __name__ == "__main__":
    num_samples = 100
    dataset = "MNIST"
    X_train, y_train, X_validation, y_validation, X_test, y_test = load_joint_data_as_matrix(num_samples,dataset=dataset)

    print("Training")
    model = LabelSpreading()
    t0 = time.time()
    model.fit(X_train,y_train)
    t = time.time() - t0
    acc = model.score(X_test,y_test)
    print("final accuracy label spreading",acc)
    print("running time",t)
