import joblib
import pandas as pd
import numpy as np
from sklearn import model_selection, metrics
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
    for i in range(0,nb_prediction):
        # Load the trained model
        model_path = f'/home/s2316001/codepaper/model_train_file3/logistic_regression{i}.sav'  # Change the path to the saved model
        loaded_model = joblib.load(model_path)
        #print("testing: ", model_path)
        
        # Perform predictions on the new data
        predictions_new = loaded_model.predict(X)

        # Calculate evaluation metrics
        accuracy_new[i] = metrics.accuracy_score(y_bin_new, predictions_new)
        # precision_new[i] = metrics.precision_score(y_bin_new, predictions_new)
        # recall_new[i] = metrics.recall_score(y_bin_new, predictions_new)
        # fbeta_new[i] = metrics.fbeta_score(y_bin_new, predictions_new, beta=1)  # Change beta value if needed
        precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_bin_new, predictions_new)
        precision_new[i] = precision[0]
        recall_new[i] = recall[0]
        fbeta_new[i] = fbeta_score[0]

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
    