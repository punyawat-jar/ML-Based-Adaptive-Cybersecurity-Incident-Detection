import numpy as np
import pandas as pd
import joblib
from sklearn import model_selection, feature_selection, kernel_approximation, ensemble, linear_model, metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
import seaborn as sns
for i in range (1,14):
    X = pd.read_hdf(f'/home/s2316001/codepaper/data_window_botnet{i}.h5', key='data')
    X.reset_index(drop=True, inplace=True)

    X2 = pd.read_hdf(f'/home/s2316001/codepaper/data_window3_botnet{i}.h5', key='data')
    X2.reset_index(drop=True, inplace=True)

    X = X.join(X2)

    X.drop('window_id', axis=1, inplace=True)

    y = X['Label_<lambda>']
    X.drop('Label_<lambda>', axis=1, inplace=True)
    labels = np.load(f"/home/s2316001/codepaper/data_window_botnet{i}_labels.npy", allow_pickle=True)

    #print("combined_labels:",combined_labels)
    # Split the combined data into train and test sets
    y_bin6 = y == np.where(labels == 'flow=From-Botne')[0][0]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_bin6, test_size=0.33, random_state=123456)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(256, input_shape=(1, X_train.shape[2]), activation='relu', return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Accuracy Train: %.3f, Test: %.3f' % (train_acc, test_acc))
        # plot loss during training
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()
    # # Plot the loss and accuracy
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(history.history['accuracy'], label='Training Accuracy')
    # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()

    plt.tight_layout()
    plt.savefig(f'/home/s2316001/codepaper/graph/loss_accuracy_plot{i}.png')  # Save the plot as an image file

    # predict probabilities for test set
    yhat_probs = model.predict(X_test, verbose=0)
    # predict crisp classes for test set
    yhat_classes = (model.predict(X_test) > 0.5).astype("int32")
    # reduce to 1d array
    # yhat_probs = yhat_probs.reshape(yhat_probs.shape[0])
    # yhat_classes = yhat_classes.reshape(yhat_classes.shape[0])
    print(f"---------------------- File {i} --------------------------")
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, yhat_classes)
    print('F1 score: %f' % f1)

    # confusion matrix
    matrix = confusion_matrix(y_test, yhat_classes)
    print(matrix)
    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(f'/home/s2316001/codepaper/graph/lstm_confusion_matrix{i}.png')
    plt.show()
    # Save the model
    model.save(f'/home/s2316001/codepaper/LSTM/lstm_model{i}.h5')