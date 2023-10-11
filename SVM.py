import pickle
from sklearn import svm
from sklearn.metrics import classification_report
class Svm:
    def __init__(self, x_train_c, x_test_c, y_train_c, y_test_c):
        self.x_train_c = x_train_c
        self.x_test_c = x_test_c
        self.y_train_c = y_train_c
        self.y_test_c = y_test_c

    def train(self):
        svm_model = svm.SVC()
        svm_model.fit(self.x_train_c, self.y_train_c)
        print("svm model accuracy: ", svm_model.score(self.x_test_c, self.y_test_c))
        with open("SVM_model_pickle", "wb") as file:
            pickle.dump(svm_model, file)

    def get_svm_model(self):
        with open("SVM_model_pickle", "rb") as file:
            svm_model = pickle.load(file)
            return svm_model

    def get_svm_report(self):
        svm_predict=self.get_svm_model().predict(self.x_test_c)
        print("SVM report : ")
        print(classification_report(self.y_test_c, svm_predict))

