import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessing:

    def preproceccing(self,cancer_prediction):
        cancer_prediction['diagnosis'] = cancer_prediction['diagnosis'].replace("M", 0)
        cancer_prediction['diagnosis'] = cancer_prediction['diagnosis'].replace("B", 1)
        cancer_prediction['diagnosis'] = cancer_prediction['diagnosis'].astype('int64')


        return cancer_prediction

    def scale(self,cancer_prediction):
        scaling = StandardScaler()
        x_test_c=pd.DataFrame(scaling.fit_transform(cancer_prediction.iloc[:, 1:31]))
        return x_test_c
