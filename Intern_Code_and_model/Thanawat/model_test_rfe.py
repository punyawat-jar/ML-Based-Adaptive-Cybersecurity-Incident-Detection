import joblib
import pandas as pd
import numpy as np
from sklearn import model_selection, metrics,utils
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
for j in range (1,14):
    # print("Import data")

    X = pd.read_hdf(f'/home/s2316001/codepaper/data_window_botnet{j}.h5', key='data')
    X.reset_index(drop=True, inplace=True)

    X2 = pd.read_hdf(f'/home/s2316001/codepaper/data_window3_botnet{j}.h5', key='data')
    X2.reset_index(drop=True, inplace=True)

    X = X.join(X2)

    X.drop('window_id', axis=1, inplace=True)

    y = X['Label_<lambda>']
    X.drop('Label_<lambda>', axis=1, inplace=True)

    labels = np.load(f"/home/s2316001/codepaper/data_window_botnet{j}_labels.npy", allow_pickle= True)

    # print(X.columns.values)
    # print(labels)
    # print(np.where(labels == 'flow=From-Botne')[0][0])

    y_bin_new = y==np.where(labels == 'flow=From-Botne')[0][0]

    #print("y", np.unique(y, return_counts=True))

    nb_prediction = 50
    accuracy_new = np.array([0.]*nb_prediction)
    precision_new = np.array([0.]*nb_prediction)
    recall_new = np.array([0.]*nb_prediction)
    fbeta_new = np.array([0.]*nb_prediction)
    tab_seed = np.random.randint(0, 1000000000, nb_prediction)
    for i in range(0,nb_prediction):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_bin_new, test_size=0.33, random_state=None)
        #X_train_new, y_train_new = utils.resample(X, y_bin_new, n_samples=X_train.shape[0]*10, random_state=tab_seed[i])
        # Load the trained model
        model_path = f'/home/s2316001/codepaper/model_rfe/rfe_file13_model{i}.sav'  # Change the path to the saved model
        loaded_model = joblib.load(model_path)
        #print("testing: ", model_path)
        
        # Perform predictions on the new data
        #loaded_model = loaded_model.fit(X_train_new,y_train_new)
        predictions_new = loaded_model.predict(X_test)
        
        # Calculate evaluation metrics
        accuracy_new[i] = metrics.accuracy_score(y_test, predictions_new)
        precision_new[i] = metrics.precision_score(y_test, predictions_new)
        recall_new[i] = metrics.recall_score(y_test, predictions_new)
        fbeta_new[i] = metrics.fbeta_score(y_test, predictions_new, beta=1)  # Change beta value if needed
        # precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_test, predictions_new)
        # precision_new[i] = precision[1]
        # recall_new[i] = recall[1]
        # fbeta_new[i] = fbeta_score[1]
         # confusion matrix
        matrix = metrics.confusion_matrix(y_test, predictions_new)
        #print(matrix)
        # Plot Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig(f'/home/s2316001/codepaper/model_rfe_test_file13/rfe_confusion_matrix_file{j}_No{i}.png')
        #tn[i], fp[i], fn[i], tp[i] = metrics.confusion_matrix(y_test, predictions_new,label = [True, False] )        
        # # Print the evaluation metrics
        # print("Accuracy:",accuracy_new)
        # print("Precision:", precision_new)
        # print("Recall:", recall_new)
        # print("F-beta score:", fbeta_new)

    print(f"-----------------Total {j}---------------")
    print("Test")
    print("Accuracy:",accuracy_new.mean(), accuracy_new.std(), accuracy_new)
    print("precision = ", precision_new.mean(), precision_new.std(), precision_new)
    print("recall = ", recall_new.mean(), recall_new.std(), recall_new)
    print("fbeta_score = ", fbeta_new.mean(), fbeta_new.std(), fbeta_new)
    # print("TN = ", tn.mean(), tn.std(),tn.new)
    # print("FP = ", fp.mean(), fp.std(),fp.new)
    # print("FN = ", fn.mean(), fn.std(),fn.new)
    # print("TP = ", tp.mean(), tp.std(),tp.new)