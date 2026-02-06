import os

import matplotlib.pyplot as plt
import pandas as pd
from sphinx.addnodes import index
from sqlalchemy import column


def combina():
    # Define the folders
    scenarios_folder = './'

    # Initialize lists
    data_list = []
    binary_labels = []


    counter=1
    # Read all CSV files from the healthy folder
    # for file in os.listdir(healthy_folder):
    #     if file.endswith('.csv'):
    #         file_path = os.path.join(healthy_folder, file)
    #         df=pd.read_csv(file_path)
    #         df["event"]=[0 for _i in range(df.shape[0])]
    #         df["source"]=[file.split(".csv")[0][-3:] for _i in range(df.shape[0])]
    #         data_list.append(df)
    #         counter+=1
    #         binary_labels.append(0)  # Healthy files are labeled as 0

    # Read all CSV files from the scenarios folder
    starttime=pd.to_datetime('2020-01-01 00:00:00')
    for file in os.listdir(scenarios_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(scenarios_folder, file)
            df = pd.read_csv(file_path,index_col=0)
            df["event"] = [1 for _i in range(df.shape[0])]
            df["source"] = [file.split("features.csv")[0][-2:-1] for _i in range(df.shape[0])]
            df["Artificial_timestamp"] = [starttime + pd.to_timedelta(cycle, unit='h') for cycle in df["Cycle_Index"]]
            df.drop(columns=["Cycle_Index"],inplace=True)
            data_list.append(df)
            counter += 1
            binary_labels.append(1)  # Scenario files are labeled as 1


    columns=data_list[0].columns.tolist()
    for df in data_list:
        # Example preprocessing: Fill missing values with the mean of each column
        print(len(df.columns.tolist()))
        columns= set(columns).intersection(set(df.columns.tolist()))

    # print("-------------------")
    # print(len(columns))
    # print("-------------------")
    # for df in data_list:
    #     # Example preprocessing: Fill missing values with the mean of each column
    #     print(set(df.columns.tolist()).difference(columns))

    # combine all dataframes into a single dataframe
    combined_df = pd.concat(data_list, ignore_index=True)

    combined_df.to_csv("HNEI_combined.csv", index=False)

def show():
    from sklearn.preprocessing import MinMaxScaler
    df=pd.read_csv("./HNEI_combined.csv")
    cols_not=['source','Artificial_timestamp','event']
    columns=[col for col in df.columns.tolist() if col not in cols_not]
    df[columns]=MinMaxScaler().fit_transform(df[columns])
    df['Artificial_timestamp']=pd.to_datetime(df['Artificial_timestamp'])

    for source in df['source'].unique():
        df_source=df[df['source']==source]
        df_source.sort_values(by='RUL',inplace=True)
        df_source[columns].plot()
        plt.show()

# combina()
show()