from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

def benchmark_svc(x_train, y_train, x_test, y_test, gamma=0.001, C=100.):
    clf = svm.SVC(gamma=0.001, C=100.)
    clf.fit(x_train, y_train)
    result = clf.score(x_test, y_test)
    print("SVC: " + str(result))
    return result
    
def benchmark_LogisticRegression(x_train, y_train, x_test, y_test, max_iter=1000, random_state=123):
    clf = LogisticRegression(max_iter=1000, random_state=123)
    clf.fit(x_train, y_train)
    result = clf.score(x_test, y_test)
    print("LogisticRegression: " + str(result))
    return result
    
def benchmark_RandomForestClassifier(x_train, y_train, x_test, y_test, n_estimators=100, random_state=123):
    clf = RandomForestClassifier(n_estimators=100, random_state=123)
    clf.fit(x_train, y_train)
    result = clf.score(x_test, y_test)
    print("RandomForestClassifier: " + str(result))
    return result

def benchmark_GaussianNB(x_train, y_train, x_test, y_test):
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    result = clf.score(x_test, y_test)
    print("GaussianNB: " + str(result))
    return result

def benchmark_all(x_train, y_train, x_test, y_test, gamma=0.001, C=100., max_iter=1000, random_state=123, n_estimators=100):
    benchmark_svc(x_train, y_train, x_test, y_test, gamma, C)
    benchmark_LogisticRegression(x_train, y_train, x_test, y_test, max_iter, random_state)
    benchmark_RandomForestClassifier(x_train, y_train, x_test, y_test, n_estimators, random_state)
    benchmark_GaussianNB(x_train, y_train, x_test, y_test)