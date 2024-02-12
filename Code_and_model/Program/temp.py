import pandas as pd
import os
import gc
from tqdm import tqdm
import numpy as np
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
os.getcwd()

df = pd.read_csv('kdd/train_test_folder/train_kdd/train.csv')
traindf = pd.read_csv('./kdd/dataset/mix_dataset/neptune.csv')

train_test_index = [df['Unnamed: 0'], df['Unnamed: 0']]

from multiprocessing import Pool, cpu_count

window_size = 64
batch_size = 2
epochs = 20

X = traindf.drop('label', axis=1)
y = traindf['label']

Data = TimeseriesGenerator(X, y, length=window_size, sampling_rate=1, batch_size=batch_size)


def process_data(Data, train_index, test_index, batch_size, window_size, chunk_size=200):
    def rearrange_sequences(generator, index, chunk_size):
        num_batches = len(generator)
        rearranged_data = []

        for chunk_start in tqdm(range(0, num_batches, chunk_size), desc="Processing chunks"):
            chunk_end = min(chunk_start + chunk_size, num_batches)
            for i in range(chunk_start, chunk_end):
                batch_x, batch_y = generator[i]  # Access each batch from the generator

                for j in range(batch_x.shape[0]):  # Process each sequence in the batch
                    sequence_start_index = i * batch_size + j
                    original_indices = index[sequence_start_index: sequence_start_index + window_size].tolist()
                    rearranged_data.append((batch_x[j], batch_y[j], original_indices))
            gc.collect()
        gc.collect()
        return rearranged_data
    # Process training data
    print('Training data processing...')
    train_data = rearrange_sequences(Data, train_index, chunk_size)
    del rearrange_sequences
    print('Testing data processing...')
    test_data = rearrange_sequences(Data, test_index, chunk_size)
    gc.collect()
    return train_data
    
def separate_features_labels(data, batch_size=128):
    total_batches = len(data) // batch_size + (1 if len(data) % batch_size > 0 else 0)
    features_list, labels_list = [], []

    for batch_num in tqdm(range(total_batches), desc="Separating features and labels"):
        batch_start = batch_num * batch_size
        batch_end = min((batch_num + 1) * batch_size, len(data))
        batch_data = data[batch_start:batch_end]

        batch_features = [np.array(item[0], dtype=np.float32) for item in batch_data]  # Reduce precision
        batch_labels = [item[1] for item in batch_data]

        features_list.append(np.concatenate(batch_features, axis=0))  # Concatenate immediately to reduce list overhead
        labels_list.append(np.array(batch_labels, dtype=np.float32))  # Adjust dtype as necessary

        gc.collect()
        
    gc.collect()
    
    features_array = np.concatenate([f.astype(np.float32) for f in features_list], axis=0)
    labels_array = np.concatenate([l.astype(np.float32) for l in labels_list], axis=0)

    return features_array, labels_array

Data = TimeseriesGenerator(X, y, length=window_size, sampling_rate=1, batch_size=batch_size)
print(f'Data type {type(Data)}')
train_index, test_index = train_test_index
print('Processing Training data...')

train_data, test_data = process_data(Data, train_index, test_index, batch_size, window_size)
gc.collect()
X_train, y_train = separate_features_labels(train_data)
X_test, y_test = separate_features_labels(test_data)

sequence_start_index = 90597 - (window_size - 1)
# Original data point
array = traindf.drop('label', axis=1).iloc[90597].to_numpy()
batch_index = sequence_start_index // batch_size
# Data from the generator
datagen = Data[batch_index][0][0][window_size-1]

# Comparison
equal = np.array_equal(array, datagen)
print(f"Compare equal :{equal}")