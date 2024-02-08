import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from multiprocessing import Pool, cpu_count

import gc

def rearrange_sequences_linear_chunked(generator, df_index, batch_size, window_size, chunk_size=100):
    num_batches = len(generator)
    
    rearranged_data = []  # This will store the final results

    for chunk_start in tqdm(range(0, num_batches, chunk_size), desc="Processing chunks"):
        chunk_end = min(chunk_start + chunk_size, num_batches)
        for i in range(chunk_start, chunk_end):
            batch_x, batch_y = generator[i]  # Access each batch from the generator

            for j in range(batch_x.shape[0]):  # Process each sequence in the batch
                sequence_start_index = i * batch_size + j
                original_indices = df_index[sequence_start_index: sequence_start_index + window_size].tolist()
                rearranged_data.append((batch_x[j], batch_y[j], original_indices))
        
        gc.collect()  # Suggest to the garbage collector to release unreferenced memory

    return rearranged_data

def main():
    df = pd.read_csv('kdd/train_test_folder/train_kdd/train.csv')
    traindf = pd.read_csv('./kdd/dataset/mix_dataset/neptune.csv')

    window_size = 512
    batch_size = 8
    epochs = 20

    X = traindf.drop('label', axis=1)
    y = traindf['label']

    Data = TimeseriesGenerator(X, y, length=window_size, sampling_rate=1, batch_size=batch_size)

    rearranged_data = rearrange_sequences_linear_chunked(Data, df['Unnamed: 0'], batch_size, window_size, chunk_size=50)
    print('All process is done...')

# Protect the entry point of the script
if __name__ == '__main__':
    main()