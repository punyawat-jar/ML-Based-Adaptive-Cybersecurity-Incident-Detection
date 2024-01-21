import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier 
import joblib
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import h5py
import joblib
from sklearn import model_selection, feature_selection, utils, ensemble, linear_model, metrics, kernel_approximation
import random
import seaborn as sns
nb_prediction = 30
accuracy_new = np.zeros((30,13))
precision_new = np.zeros((30,13))
recall_new = np.zeros((30,13))
fbeta_new = np.zeros((30,13))
for k in range (0,30):
    for i in range (1,14):
        model_path = f'/home/s2316001/model_rfe/{k+1}/rfe_model{i}.sav'  # Change the path to the saved model
        exec(f'loaded_model{i} = joblib.load(model_path)')
    for z in range (0,13):
        new_data = pd.read_csv(f"/home/s2316001/Clean_data/{k+1}/test{z+1}.csv")
        X_en = new_data.drop(columns=[new_data.columns[0],"Label"], axis=1)
        y_en = new_data['Label']
        predictions=[]
        prediction = loaded_model1.predict(X_en)
        predictions.append(prediction)
        prediction = loaded_model2.predict(X_en)
        predictions.append(prediction)
        prediction = loaded_model3.predict(X_en)
        predictions.append(prediction)
        prediction = loaded_model4.predict(X_en)
        predictions.append(prediction)
        prediction = loaded_model5.predict(X_en)
        predictions.append(prediction)
        prediction = loaded_model6.predict(X_en)
        predictions.append(prediction)
        prediction = loaded_model7.predict(X_en)
        predictions.append(prediction)
        prediction = loaded_model8.predict(X_en)
        predictions.append(prediction)
        prediction = loaded_model9.predict(X_en)
        predictions.append(prediction)
        prediction = loaded_model10.predict(X_en)
        predictions.append(prediction)
        prediction = loaded_model11.predict(X_en)
        predictions.append(prediction)
        prediction = loaded_model12.predict(X_en)
        predictions.append(prediction)
        prediction = loaded_model13.predict(X_en)
        predictions.append(prediction)
        temp = np.stack((predictions[0],predictions[1],predictions[2],predictions[3],predictions[4],predictions[5],predictions[6],predictions[7],predictions[8],predictions[9],predictions[10],predictions[11],predictions[12]), axis=-1)
        result = np.arange(temp.shape[0])
        rows = temp.shape[0]
        columns= temp.shape[1]
        for i in range (rows):
            check = 0
            for j in range (columns):
                if  temp[i][j] == 1:
                    result[i]=1
                    break
                else:
                    result[i]=0
        print(f"------------------ round {k+1} Test file {z+1} -----------------------")
        score = accuracy_score(y_en, result)
        print('Ensemble Model Accuracy: {}'.format(score))
        precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_en, result)
        print(f"Ensemble Model precesion: {precision[1]}")
        print(f"Ensemble Model recall: {recall[1]}")
        print(f"Ensemble Model fbeta_score: {fbeta_score[1]}")
        accuracy_new[k][z] = score
        precision_new[k][z] = precision[1]
        recall_new[k][z] = recall[1]
        fbeta_new[k][z] = fbeta_score[1]
        matrix = metrics.confusion_matrix(y_en, result)
        #print(matrix)
        # Plot Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig(f'/home/s2316001/Ensemble_matrix/{k+1}/manual_all13_file{z+1}.png')
        plt.show()
acc_mean = np.mean(accuracy_new, axis=1)
acc_std = np.std(accuracy_new, axis=1)
prec_mean = np.mean(precision_new, axis=1)
prec_std = np.std(precision_new, axis=1)
rec_mean = np.mean(recall_new, axis=1)
rec_std = np.std(recall_new, axis=1)
f_mean  = np.mean(fbeta_new, axis=1)
f_std = np.std(fbeta_new, axis=1)
for z in range (0,13):
    print(f"-----------------test{z+1}---------------")
    print("Test")
    print("Accuracy:",acc_mean[z], acc_std[z], accuracy_new[z])
    print("precision = ", prec_mean[z], prec_std[z], precision_new[z])
    print("recall = ", rec_mean[z], rec_std[z], recall_new[z])
    print("fbeta_score = ", f_mean[z], f_std[z], fbeta_new[z])
            
         