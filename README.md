# Machine Learning Based Adaptive Cybersecurity Incident Detection

## To train your data
For running the program with your data's format, you need to preprocess the data based on your criteria. The data will be trained with the scikit-learn machine learning models. We use python version 3.10.13 to train and evaluate the Multimodel.

## How to use (By demo)
To setting the conda environment, the requirements will be contained in `requirements.txt`, which you can use `pip` to install packages.
```
pip install -r requirements.txt
```


### Setting Environment
The program located in [here](https://github.com/punyawat-jar/ML-Based-Adaptive-Cybersecurity-Incident-Detection/tree/main/Code_and_model/Program). To build the setting environment, you can run `setting.py` to create the ready-to-use environment. The input data parameter `--data` is the data's format. For ours demo, we use `kdd` and `cic`. 

To run the kdd dataset:

```
python setting.py --data kdd
```

or to run the cic dataset:

```
python setting.py --data cic
```

### Input Dataset
The data's format folder will be created. In this state, you can put the dataset (based on the data's format) in the `./dataset/InputDataset/`. The NSL-KDD dataset provided the `Train+.txt` and `Test+.txt` for training and evaluation, which you can download [here](https://drive.google.com/drive/folders/1hv6hIJobDL3sDOftqieC8CACUt6uLcjU?usp=share_link) You can put both files in this location. In addition, the program also accept the `.csv` format for this data's format, in case that the files need to be preprocessed separately.

As well as the CIC-IDS2017, the original dataset provided `.csv` files of 7 days attacks. You can input all the files in the location according to the data's format folder, which you can download [here](https://drive.google.com/drive/folders/18dUb7JCzX-v4VIMpQ2gsToqEb2nWU8Qm?usp=share_link)

### Process the Input dataset
After input the files in the folder, the preprocessing is needed. The `preprocess.py` will split the dataset that contain all attack labels to a dataset that specifically contain only one label. Therefore, the amount of the split dataset will following the amount of attack label of the input dataset. Moreover, in case you already have the preprocessed dataset, you can use the input argrument `--network` to located the concated dataset.

To run the kdd dataset:

```
python preprocess.py --data kdd
```

or to run the cic dataset:

```
python preprocess.py --data cic
```

In addition, you can use argrument `--multiProcess` to use the multiple processing to handle large amount of data. Also, the argrument `--n_Process` to set the number of processes, in cases that all the CPU process required massive amount of RAM.

### Training Multimodel

To train the model, you can run `train.py` to training the model with the preprocessed dataset. In addition, the `--multiProcess` and the `--n_Process` argruments is available. Moreover, the `--model` argrument will set the type of the training (ML or DL), which the default setting is ML.
To select the algorithm of the ML (Also, DL), the algorithm's list will be shown in the `./module/model.py`. The first (or top) algorithm that input in the `model` Dictionarie got highest priority, if the evaluation score are equal.

To run the kdd dataset:
```
python train.py --data kdd
```

or to run the cic dataset:

```
python train.py --data cic
```

### Evaluation Multimodel
To evaluation the multimodel, you can run `test.py` to evaluate the multimodels. To show how the single data can be detected, the `--sequence_mode` argrument is to turn on the sequence mode, which will input the data from testing set once at a time.

To run the kdd dataset:
```
python test.py --data kdd
```

or to run the cic dataset:
```
python test.py --data cic
```

## License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.
