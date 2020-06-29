import datetime as dt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import KFold



results_test = dict()
results_test['accuracy'] = 0
results_test['precision'] = 0
results_test['recall'] = 0
results_test['f1_score'] = 0

class Model:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def confusion_matrix_func(self, Y_test, y_test_pred):
        '''
        This function plots the confusion matrix heatmap using the actual and predicted values.
        '''
        C = confusion_matrix(Y_test, y_test_pred)
        cm_df = pd.DataFrame(C)
        labels = ['dos', 'normal', 'probe', 'r2l', 'u2r']
        sns.set(font_scale=1.1)
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(cm_df, annot=True,
                         fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)

        plt.show()


    def reset_result(self):
        results_test['accuracy'] = 0
        results_test['precision'] = 0
        results_test['recall'] = 0
        results_test['f1_score'] = 0
        print('Prediction on test data:')


    def print_result(self):
        print('Accuracy score is:')
        print(results_test['accuracy'] / 5)
        print('=' * 50)
        print('Precision score is:')
        print(results_test['precision'] / 5)
        print('=' * 50)
        print('Recall score is:')
        print(results_test['recall'] / 5)
        print('=' * 50)
        print('F1-score is:')
        print(results_test['f1_score'] / 5)
        # add the trained  model to the results

        return results_test

    def model(self,model_name, X_train, Y_train, X_test, Y_test):
        '''
        This function computes the performance scores on the train and test data.
        '''
        model_name.fit(X_train, Y_train)
        y_test_pred = model_name.predict(X_test)

        print('=' * 50)

        print('Classification Report: ')
        result_classification_report = classification_report(Y_test, y_test_pred)
        print(result_classification_report)

        results_test['accuracy'] += accuracy_score(Y_test, y_test_pred)
        results_test['precision'] += precision_score(Y_test, y_test_pred, average='weighted')
        results_test['recall'] += recall_score(Y_test, y_test_pred, average='weighted')
        results_test['f1_score'] += f1_score(Y_test, y_test_pred, average='weighted')

        print('Confusion Matrix is:')
        self.confusion_matrix_func(Y_test, y_test_pred)


    def kfold_validation(self,classifer):
        kf = KFold(n_splits=5)
        start = dt.datetime.now()
        self.reset_result()
        X = self.X
        Y = self.Y

        for train_index, test_index in kf.split(self.X, self.Y):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            self.model(classifer, X_train, Y_train, X_test, Y_test)
        self.print_result()
        print('Completed')
        print('Time taken:', dt.datetime.now() - start)


    # Model 1: Decision Tree

    def decision_tree(self):
        print('---------------------------')
        print('|        Decision Tree       |')
        print('---------------------------\n')
        hyperparameter = {'max_depth': [5, 10, 20, 50, 100, 500], 'min_samples_split': [5, 10, 100, 500]}

        decision_tree = DecisionTreeClassifier(criterion='gini', class_weight='balanced')
        self.kfold_validation(decision_tree)

    # Model 2: Random Forest
    def random_forest(self):
        print('---------------------------')
        print('|        Random Forest       |')
        print('---------------------------\n')
        randomForest = RandomForestClassifier(n_estimators=100)
        self.kfold_validation(randomForest)

    # Model 3: Naive Bayes
    def naive_bayes(self):
        print('---------------------------')
        print('|        Naive Bayes       |')
        print('---------------------------\n')
        gaussian_nb = GaussianNB()
        self.kfold_validation(gaussian_nb)

    # Model 4: KNN
    def k_nearest_neighbor(self):
        print('---------------------------')
        print('|        K Nearest Neighbor       |\n')
        print('---------------------------')
        knn = KNeighborsClassifier(3)
        self.kfold_validation(knn)

