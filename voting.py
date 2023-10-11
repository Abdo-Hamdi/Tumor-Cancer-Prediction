import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from Preprocessing import Preprocessing
import numpy as np

class Vote:
    def __init__(self,svm_model,logistic_model,tree):
        self.svm_model=svm_model
        self.logistic_model=logistic_model
        self.tree=tree

    def voting_function(self):
        print("Please enter the name of the csv file ")
        path = input()
        path=path+".csv"
        cancer_prediction = pd.read_csv(path)
        preprocess = Preprocessing()
        cancer_prediction=preprocess.preproceccing(cancer_prediction)

        x_test_c=preprocess.scale(cancer_prediction)
        y_test_c=cancer_prediction['diagnosis']

        S_predict = self.svm_model.predict(x_test_c)
        DT_predict = self.tree.predict(x_test_c)
        l_predict = self.logistic_model.predict(x_test_c)

        final_vote = []
        final_vote_numbered = []
        for i in range(x_test_c.shape[0]):
            # print("for sample ", i+1)
            # if S_predict[i] == 0:
            #     print("SVM model prediction: M")
            # else :
            #     print("SVM model prediction: B")
            # if DT_predict[i] == 0:
            #     print("Decision tree model prediction: M")
            # else:
            #     print("Decision tree model prediction: B")
            # if l_predict[i] == 0:
            #     print("Logistic model prediction: M")
            # else:
            #     print("Logistic model prediction: B")

            if S_predict[i] + DT_predict[i] + l_predict[i] >= 2:
                final_vote_numbered.insert(i, 1)
                final_vote.insert(i, "B")
                # print("Final vote is ", "B")
            else:
                final_vote_numbered.insert(i, 0)
                final_vote.insert(i, "M")
                # print("Final vote is ", "M")
            # print()
        print("Final Prediction is : ", final_vote)
        print("Final voting accuracy is : ", metrics.accuracy_score(y_test_c, final_vote_numbered) * 100, "%")
