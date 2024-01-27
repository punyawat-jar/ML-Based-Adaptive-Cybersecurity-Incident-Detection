from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def getModel():
    models = {
        'LogisticRegression': LogisticRegression(max_iter=10000, n_jobs=-1),
        'ExtraTrees': ExtraTreesClassifier(n_jobs=-1),
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis(),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(),
        'KNeighbors': KNeighborsClassifier(n_jobs=-1),
        'GaussianNB': GaussianNB(),
        'Perceptron': Perceptron(n_jobs=-1),
        'AdaBoost': AdaBoostClassifier()
    }
    
    return models
