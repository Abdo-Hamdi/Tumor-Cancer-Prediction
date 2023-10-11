from sklearn.model_selection import train_test_split
from voting import Vote
from decision_tree import Decision_tree
from logistic_regression import Logstic_regression
from Preprocessing import Preprocessing
from SVM import Svm
import pandas as pd


preprocess=Preprocessing()

cancer_df = pd.read_csv("Tumor_Cancer.csv")
cancer_df=preprocess.preproceccing(cancer_df)
x_cancer_df = preprocess.scale(cancer_df)
y_cancer_df = cancer_df['diagnosis']

x_train_c, x_test_c, y_train_c, y_test_c = train_test_split(x_cancer_df, y_cancer_df, test_size=0.25,shuffle=False)

svm=Svm(x_train_c, x_test_c, y_train_c, y_test_c)
logistic_regression=Logstic_regression(x_train_c, x_test_c, y_train_c, y_test_c)
decision_tree=Decision_tree(x_train_c, x_test_c, y_train_c, y_test_c)

svm.get_svm_report()
logistic_regression.get_logistic_report()
decision_tree.get_tree_report()

v=Vote(svm.get_svm_model(), logistic_regression.get_logistic_model(), decision_tree.get_decision_tree_model())
v.voting_function()





