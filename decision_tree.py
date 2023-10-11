import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


class Decision_tree:
    def __init__(self,x_train_c,x_test_c,y_train_c,y_test_c):

        self.x_train_c=x_train_c
        self.x_test_c=x_test_c
        self.y_train_c=y_train_c
        self.y_test_c=y_test_c

    def train(self):
        tree = DecisionTreeClassifier()
        tree.fit(self.x_train_c, self.y_train_c)
        print("decision tree model accuracy: ", tree.score(self.x_test_c, self.y_test_c))
        with open("tree_model_pickle", "wb") as file:
            pickle.dump(tree, file)

    def get_decision_tree_model(self):
        with open("tree_model_pickle", "rb") as file:
            tree = pickle.load(file)
            return tree

    def get_tree_report(self):
        tree_predict = self.get_decision_tree_model().predict(self.x_test_c)
        print("decision tree report : ")
        print(classification_report(self.y_test_c, tree_predict))