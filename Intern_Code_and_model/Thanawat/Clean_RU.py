import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
window_width = 120
window_stride = 120
def RU(df):
    if df.shape[0] == 1:
        return 1.0
    else:
        proba = df.value_counts()/df.shape[0]
        h = proba*np.log10(proba)
        return -h.sum()/np.log10(df.shape[0])
def normalize_column(dt, column):
    mean = dt[column].mean()
    std = dt[column].std()
    print(mean, std)

    dt[column] = (dt[column]-mean) / std

for i in range (1,14):
    df = pd.read_csv(f'/home/s2316001/data/{i}.csv')
    data = pd.DataFrame(df)
    data['StartTime'] = pd.to_datetime(df['StartTime']).astype(np.int64)*1e-9
    datetime_start = data['StartTime'].min()
    data['Window_lower'] = (data['StartTime']-datetime_start-window_width)/window_stride+1
    data['Window_lower'].clip(lower=0, inplace=True)
    data['Window_upper_excl'] = (data['StartTime']-datetime_start)/window_stride+1
    data = data.astype({"Window_lower": int, "Window_upper_excl": int})
    data = data.sort_values(by=['StartTime'], ascending=True)
    data['Label'] = np.where(data['Label'].str.slice(0, 15) == 'flow=From-Botne', 1, 0)
    nb_windows = data['Window_upper_excl'].max()
    X = pd.DataFrame(data)
    X['Sport_RU'] = 0
    X['DstAddr_RU'] = 0
    X['Dport_RU'] = 0
    for j in range(nb_windows):
        window_condition = (X['Window_lower'] <= j) & (X['Window_upper_excl'] > j)
        window_data = X[window_condition]

        gb = window_data.groupby(['SrcAddr']).agg({'Sport': RU, 'DstAddr': RU, 'Dport': RU,'State': RU,'Dir': RU,'sTos': RU,'dTos': RU,'Proto': RU})
        gb.columns = ['Sport_RU', 'DstAddr_RU', 'Dport_RU','State','Dir','sTos','dTos','Proto']

        X.loc[window_condition, 'Sport_RU'] = X.loc[window_condition, 'SrcAddr'].map(gb['Sport_RU'])
        X.loc[window_condition, 'DstAddr_RU'] = X.loc[window_condition, 'SrcAddr'].map(gb['DstAddr_RU'])
        X.loc[window_condition, 'Dport_RU'] = X.loc[window_condition, 'SrcAddr'].map(gb['Dport_RU'])
        X.loc[window_condition, 'State'] = X.loc[window_condition, 'SrcAddr'].map(gb['State'])
        X.loc[window_condition, 'Dir'] = X.loc[window_condition, 'SrcAddr'].map(gb['Dir'])
        X.loc[window_condition, 'sTos'] = X.loc[window_condition, 'SrcAddr'].map(gb['sTos'])
        X.loc[window_condition, 'dTos'] = X.loc[window_condition, 'SrcAddr'].map(gb['dTos'])
        X.loc[window_condition, 'Proto'] = X.loc[window_condition, 'SrcAddr'].map(gb['Proto'])
    # # Create dummy variables for the 'Color' column
    # dummies = pd.get_dummies(X['State'], dtype = int)
    # # Concatenate the original DataFrame with the dummy variables
    # merged_df2 = pd.concat([X, dummies], axis=1)
    # # Create dummy variables for the 'Color' column
    # dummies = pd.get_dummies(merged_df2['Dir'], dtype = int)
    # # Concatenate the original DataFrame with the dummy variables
    # merged_df3 = pd.concat([merged_df2, dummies], axis=1)
    # # Create dummy variables for the 'Color' column
    # dummies = pd.get_dummies(merged_df3['Proto'], dtype = int)
    # # Concatenate the original DataFrame with the dummy variables
    # merged_df4 = pd.concat([merged_df3, dummies], axis=1)
    X.drop(['StartTime','SrcAddr','Sport','Dport','DstAddr','Window_lower','Window_upper_excl'],axis=1,inplace=True) 
    columns_to_normalize = list(X.drop('Label',axis=1).columns.values)
    normalize_column(X, columns_to_normalize)
    X.to_csv(f'/home/s2316001/Clean_data/Clean_{i}-XX.csv', encoding='utf-8')