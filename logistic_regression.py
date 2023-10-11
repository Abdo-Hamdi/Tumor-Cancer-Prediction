import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report





class Logstic_regression:

    def __init__(self, x_train_c, x_test_c, y_train_c, y_test_c):
        self.x_train_c = x_train_c
        self.x_test_c = x_test_c
        self.y_train_c = y_train_c
        self.y_test_c = y_test_c

    def train(self):
        logistic_model = LogisticRegression()
        logistic_model.fit(self.x_train_c, self.y_train_c)
        print("logistic model accuracy: ", logistic_model.score(self.x_test_c, self.y_test_c))
        with open("logistic_model_pickle", "wb") as file:
            pickle.dump(logistic_model, file)

    def get_logistic_model(self):
        with open("logistic_model_pickle", "rb") as file:
            logistic_model = pickle.load(file)
            return logistic_model

    def get_logistic_report(self):
        logistic_predict= self.get_logistic_model().predict(self.x_test_c)
        print("logistic regression report : ")
        print(classification_report(self.y_test_c, logistic_predict))