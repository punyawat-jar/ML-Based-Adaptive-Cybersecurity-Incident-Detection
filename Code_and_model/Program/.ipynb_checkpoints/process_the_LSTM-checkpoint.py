import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

df = pd.read_csv('kdd/train_test_folder/train_kdd/train.csv')
traindf = pd.read_csv('./kdd/dataset/mix_dataset/neptune.csv')

from multiprocessing import Pool, cpu_count

window_size = 64
batch_size = 1
epochs = 20

X = traindf.drop('label', axis=1)
y = traindf['label']

Data = TimeseriesGenerator(X, y, length=window_size, sampling_rate=1, batch_size=batch_size)

def process_batch(args):
    print('Processing1....')
    batch_range, generator, df_index, batch_size, window_size, show_progress = args
    rearranged_batch_data = []
    print('Processing3....')
    # Wrap the enumeration with tqdm if this is the first subprocess to show progress
    iterable = enumerate(preloaded_data)
    if show_progress:
        iterable = tqdm(iterable, total=len(batch_range), desc="Processing batches")

    for idx, (batch_x, batch_y) in iterable:
        i = batch_range[idx]
        for j in range(batch_x.shape[0]):
            sequence_start_index = i * batch_size + j
            original_indices = df_index[sequence_start_index: sequence_start_index + window_size].tolist()
            rearranged_batch_data.append((batch_x[j], batch_y[j], original_indices))

    return rearranged_batch_data

def rearrange_sequences_multiprocessing(generator, df_index, batch_size, window_size):
    num_batches = len(generator)
    num_processes = cpu_count()

    # Split the batch indices into approximately equal chunks for each process
    batch_ranges = np.array_split(range(num_batches), num_processes)

    # Prepare the arguments for each process, setting show_progress=True only for the first batch
    print('Prepare the arguments...')
    process_args = [(batch_range, generator, df_index, batch_size, window_size, i == 0) 
                for i, batch_range in tqdm(enumerate(batch_ranges), total=len(batch_ranges), desc="Preparing process arguments")]

    print('Done...')
    rearranged_data = []
    with Pool(num_processes) as pool:
        results = pool.map(process_batch, process_args)

        # Combine the results from all processes
        for result in results:
            rearranged_data.extend(result)

    return rearranged_data

rearranged_data = rearrange_sequences_multiprocessing(Data, df['Unnamed: 0'], batch_size, window_size)
print('All process is done...')