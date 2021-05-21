from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def KNN_classifier(X_train, y_train, X_validation, y_validation, X_test, y_test, ks=[1, 2, 3, 5, 7, 10]):
    max_validation_accuracy_knn = 0
    optimal_k = 1
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        validation_accuracy = knn.score(X_validation, y_validation)

        if validation_accuracy > max_validation_accuracy_knn:
            optimal_k = k
            max_validation_accuracy_knn = validation_accuracy
    knn = KNeighborsClassifier(n_neighbors=int(optimal_k)).fit(X_train, y_train)
    return knn.score(X_test, y_test), optimal_k

def LR_classifier(X_train, y_train, X_validation, y_validation, X_test, y_test, cs=[1e-2,1,1e2,1e4,1e6,1e8,1e10], max_iter = 100):
    max_validation_accuracy_lr = 0
    optimal_c = 1
    for c in cs:
        reg = LogisticRegression(C=c, max_iter=max_iter).fit(X_train, y_train)
        validation_accuracy = reg.score(X_validation, y_validation)

        if validation_accuracy > max_validation_accuracy_lr:
            optimal_c = c
            max_validation_accuracy_lr = validation_accuracy
    reg = LogisticRegression(C=optimal_c, max_iter=max_iter).fit(X_train, y_train)
    return reg.score(X_test, y_test), optimal_c