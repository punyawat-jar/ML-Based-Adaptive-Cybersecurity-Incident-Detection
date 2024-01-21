# import joblib
# import pandas as pd
# import numpy as np
# from sklearn import model_selection, feature_selection, kernel_approximation, ensemble, linear_model, metrics

# # Load the trained model
# model_path = '/home/s2316001/codepaper/model/svm.sav'  # Change the path to the saved model
# loaded_model = joblib.load(model_path)
# print("testing: ", model_path)
# print("Import data")

# X = pd.read_hdf('/home/s2316001/codepaper/data_window_botnet1.h5', key='data')
# X.reset_index(drop=True, inplace=True)

# X2 = pd.read_hdf('/home/s2316001/codepaper/data_window3_botnet1.h5', key='data')
# X2.reset_index(drop=True, inplace=True)

# X = X.join(X2)

# X.drop('window_id', axis=1, inplace=True)

# y = X['Label_<lambda>']
# X.drop('Label_<lambda>', axis=1, inplace=True)

# labels = np.load("/home/s2316001/codepaper/data_window_botnet1_labels.npy", allow_pickle= True)

# print(X.columns.values)
# print(labels)
# print(np.where(labels == 'flow=From-Botne')[0][0])

# y_bin_new = y==np.where(labels == 'flow=From-Botne')[0][0]
# print("y", np.unique(y, return_counts=True))
# feature_map_nystroem = kernel_approximation.Nystroem(kernel='poly', gamma=None, degree=2, n_components=200, random_state=123456)
# feature_map_nystroem.fit(X)
# X = feature_map_nystroem.transform(X)

# # Perform predictions on the new data
# predictions_new = loaded_model.predict(X)

# # Calculate evaluation metrics
# accuracy_new = metrics.accuracy_score(y_bin_new, predictions_new)
# precision_new = metrics.precision_score(y_bin_new, predictions_new)
# recall_new = metrics.recall_score(y_bin_new, predictions_new)
# fbeta_new = metrics.fbeta_score(y_bin_new, predictions_new, beta=1)  # Change beta value if needed

# # Print the evaluation metrics
# print("Accuracy:",accuracy_new)
# print("Precision:", precision_new)
# print("Recall:", recall_new)
# print("F-beta score:", fbeta_new)
import joblib
import pandas as pd
import numpy as np
from sklearn import model_selection, feature_selection, kernel_approximation, ensemble, linear_model, metrics

# Load the trained model
model_path = '/home/s2316001/codepaper/model_train_file3/svm.sav'  # Change the path to the saved model
loaded_model = joblib.load(model_path)
print("testing: ", model_path)
print("Import data")
for j in range(1,14):
    print(f'--------------- Test {j} --------------------')
    X = pd.read_hdf(f'/home/s2316001/codepaper/data_window_botnet{j}.h5', key='data')
    X.reset_index(drop=True, inplace=True)

    X2 = pd.read_hdf(f'/home/s2316001/codepaper/data_window3_botnet{j}.h5', key='data')
    X2.reset_index(drop=True, inplace=True)

    X = X.join(X2)

    X.drop('window_id', axis=1, inplace=True)

    y = X['Label_<lambda>']
    X.drop('Label_<lambda>', axis=1, inplace=True)

    labels = np.load(f"/home/s2316001/codepaper/data_window_botnet{j}_labels.npy", allow_pickle= True)

    #print(X.columns.values)
    #print(labels)
    #print(np.where(labels == 'flow=From-Botne')[0][0])

    y_bin_new = y==np.where(labels == 'flow=From-Botne')[0][0]
    #print("y", np.unique(y, return_counts=True))
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_bin_new, test_size=0.33, random_state=123456)
    feature_map_nystroem = kernel_approximation.Nystroem(kernel='poly', gamma=None, degree=2, n_components=200, random_state=123456)
    feature_map_nystroem.fit(X_train)
    X_train_new = feature_map_nystroem.transform(X_train)
    X_test_new = feature_map_nystroem.transform(X_test)

    # Perform predictions on the new data
    loaded_model.fit(X_train_new,y_train)
    predictions_new = loaded_model.predict(X_test_new)

    # Calculate evaluation metrics
    accuracy_new = metrics.accuracy_score(y_test, predictions_new)
    precision_new = metrics.precision_score(y_test, predictions_new)
    recall_new = metrics.recall_score(y_test, predictions_new)
    fbeta_new = metrics.fbeta_score(y_test, predictions_new, beta=1)  # Change beta value if needed

    # Print the evaluation metrics
    print("Accuracy:",accuracy_new)
    print("Precision:", precision_new)
    print("Recall:", recall_new)
    print("F-beta score:", fbeta_new)