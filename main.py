import pandas as pd
import numpy as np
# ML and metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
# Data manipulation
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
# data display
import matplotlib.pyplot as plt


class ModelStats:
    def __init__(self, model):
        self.model = model
        self.mean = None
        self.median = None
        self.std_dev = None
        self.variance = None
        self.q1 = None
        self.q2 = None
        self.q3 = None

    def set_stats(self, mean, median, std_dev, variance, q1, q2, q3):
        self.mean = mean
        self.median = median
        self.std_dev = std_dev
        self.variance = variance
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3

    def print_stats(self):
        print("Model:", self.model)
        print("Mean:", round(self.mean, 3))
        print("Median:", round(self.median, 3))
        print("Standard Deviation:", round(self.std_dev, 3))
        print("Variance:", round(self.variance, 3))
        print("1st Quartile (Q1):", round(self.q1, 3))
        print("2nd Quartile (Q2/Median):", round(self.q2, 3))
        print("3rd Quartile (Q3):", round(self.q3, 3))

    def get_mean(self):
        return self.mean

    def get_model(self):
        return self.model


def getPerformance(ytest, yhat, yhat_proba):
    accuracy = accuracy_score(ytest, yhat)
    f1 = f1_score(ytest, yhat)
    auc_roc = roc_auc_score(ytest, yhat_proba)
    weightedAvg = accuracy * 0.2 + f1 * 0.4 + auc_roc * 0.4

    return weightedAvg


def getStats(weighted_averages, model):
    mean = np.mean(weighted_averages)
    median = np.median(weighted_averages)
    std_dev = np.std(weighted_averages)
    variance = np.var(weighted_averages)
    q1 = np.percentile(weighted_averages, 25)
    q2 = np.percentile(weighted_averages, 50)
    q3 = np.percentile(weighted_averages, 75)

    model_stats = ModelStats(model)
    model_stats.set_stats(mean, median, std_dev, variance, q1, q2, q3)
    return model_stats


def buildBoxPlot(weighted_averages, model):
    plt.boxplot(weighted_averages)
    plt.ylim(0.5, 1)
    plt.yticks([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.00])
    filename = model + '_norm.png'
    plt.savefig(filename)


def cleanData():
    data = pd.read_csv("creditcard.csv")

    # scales the amount variable in order to prevent massive outliers
    sc = StandardScaler()
    amount = data['Amount'].values
    data['Amount'] = sc.fit_transform(amount.reshape(-1, 1))

    # drops time var, should come back later to see if I can use that as an external variable
    data.drop(['Time'], axis=1, inplace=True)
    # removes duplicate transactions (approx 9k)
    data.drop_duplicates(inplace=True)

    # defines dependent and independent variables
    x = data.drop('Class', axis=1).values
    y = data['Class'].values

    return x, y


def runDT(xtrain, xtest, ytrain, ytest):
    dt = DecisionTreeClassifier(max_depth=4, criterion='entropy')
    dt.fit(xtrain, ytrain)
    tree_yhat = dt.predict(xtest)
    tree_yhat_proba = dt.predict_proba(xtest)[:, 1]

    return getPerformance(ytest, tree_yhat, tree_yhat_proba)


def runKNN(xtrain, xtest, ytrain, ytest):
    n = 7
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(xtrain, ytrain)
    knn_yhat = knn.predict(xtest)
    knn_yhat_proba = knn.predict_proba(xtest)[:, 1]

    return getPerformance(ytest, knn_yhat, knn_yhat_proba)


def runLR(xtrain, xtest, ytrain, ytest):
    lr = LogisticRegression()
    lr.fit(xtrain, ytrain)
    lr_yhat = lr.predict(xtest)
    lr_yhat_proba = lr.predict_proba(xtest)[:, 1]

    return getPerformance(ytest, lr_yhat, lr_yhat_proba)


def runSVM(xtrain, xtest, ytrain, ytest):
    svm = SVC()
    svm.fit(xtrain, ytrain)
    svm_yhat = svm.predict(xtest)
    svm_decision_scores = svm.decision_function(xtest)
    svm_yhat_proba = 1 / (1 + np.exp(-svm_decision_scores))

    return getPerformance(ytest, svm_yhat, svm_yhat_proba)


def runRF(xtrain, xtest, ytrain, ytest):
    rf = RandomForestClassifier(max_depth=4)
    rf.fit(xtrain, ytrain)
    rf_yhat = rf.predict(xtest)
    rf_yhat_proba = rf.predict_proba(xtest)[:, 1]

    return getPerformance(ytest, rf_yhat, rf_yhat_proba)


def runXGB(xtrain, xtest, ytrain, ytest):
    xgb = XGBClassifier(max_depth=4)
    xgb.fit(xtrain, ytrain)
    xgb_yhat = xgb.predict(xtest)
    xgb_yhat_proba = xgb.predict_proba(xtest)[:, 1]

    return getPerformance(ytest, xgb_yhat, xgb_yhat_proba)


def runModels(xtrain, xtest, ytrain, ytest, inmodel):
    if inmodel == "dt":
        return runDT(xtrain, xtest, ytrain, ytest)
    elif inmodel == "knn":
        return runKNN(xtrain, xtest, ytrain, ytest)
    elif inmodel == "lr":
        return runLR(xtrain, xtest, ytrain, ytest)
    elif inmodel == "svm":
        return runSVM(xtrain, xtest, ytrain, ytest)
    elif inmodel == "rf":
        return runRF(xtrain, xtest, ytrain, ytest)
    elif inmodel == "xgb":
        return runXGB(xtrain, xtest, ytrain, ytest)


def runTraining(x, y, ranstate, model, refit):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=ranstate)

    if refit == "SMOTE":
        smote = SMOTE(sampling_strategy=0.1, random_state=ranstate)
        xtrain_resampled, ytrain_resampled = smote.fit_resample(xtrain, ytrain)

        return runModels(xtrain_resampled, xtest, ytrain_resampled, ytest, model)

    elif refit == "under":
        undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=ranstate)
        xtrain_undersampled, ytrain_undersampled = undersampler.fit_resample(xtrain, ytrain)

        return runModels(xtrain_undersampled, xtest, ytrain_undersampled, ytest, model)

    else:
        return runModels(xtrain, xtest, ytrain, ytest, model)


def main():
    x, y = cleanData()
    models = ['dt', 'lr', 'svm', 'rf', 'xgb']
    refit = input("Data resampling model: ")
    model_stats = []
    for model in models:
        weighted_averages = []
        print(model)
        for ranstate in range(1, 26):
            weighted_averages.append(runTraining(x, y, ranstate, model, refit))
            print(ranstate)

        buildBoxPlot(weighted_averages, model)
        model_stats.append(getStats(weighted_averages, model))
        plt.clf()

    best = ''
    current = model_stats[0].get_mean()
    for models in model_stats:
        if models.get_mean() > current:
            current = models.get_mean()
            best = models.get_model()

    print(best)


if __name__ == '__main__':
    main()
