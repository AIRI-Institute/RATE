import os
import pandas as pd
import argparse
import numpy as np

def combine_csv_files(directory, mode):
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    combined_df = pd.DataFrame()

    string = csv_files[0]
    string_no_data = string.split('_')[:-6]
    new_string = ''
    for el in string_no_data:
        new_string += el + "_"

    new_string = new_string[:-1]

    new_string_no_run = new_string.split('_')[:-2]

    new_string2 = ''
    for el in new_string_no_run:
        new_string2 += el + "_"
    
    new_string2 = new_string2[:-1]
    

    final_folder = f"{directory}/{new_string2}"
    isExist = os.path.exists(final_folder)
    if not isExist:
        os.makedirs(final_folder)

    for num, file_name in enumerate(csv_files):
        df = pd.read_csv(os.path.join(directory, file_name))
        if num > 0:
            df = df.iloc[[1]]
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    for i in range(len(combined_df.loc[1:, 'Unnamed: 1'])):
        combined_df.loc[1:, 'Unnamed: 1'][i+1] = int(combined_df.loc[1:, 'Unnamed: 1'][i+1])

    combined_df_values = pd.DataFrame()
    for i in range(combined_df.shape[0]):
        if i > 0:
            combined_df_values = pd.concat([combined_df_values, combined_df.iloc[[i]]], ignore_index=True)

    combined_df_values = combined_df_values.sort_values(by='Unnamed: 1')

    combined_df = pd.concat([combined_df.iloc[[0]], combined_df_values], ignore_index=True)

    combined_df_stat = combined_df.iloc[[-1]].reset_index(drop=True)
    combined_df_stat.iloc[0, 0] = ''
    combined_df_stat.iloc[0, 1] = ''
    combined_df_stat.iloc[0, 2] = ''
    for i in range(7):
        combined_df_stat.iloc[0, 3+i*2] = np.mean(np.array(combined_df.iloc[1:, 3::2].iloc[:, i].astype(float)))
        combined_df_stat.iloc[0, 4+i*2] = np.std(np.array(combined_df.iloc[1:, 3::2].iloc[:, i].astype(float)))
    combined_df_stat

    combined_df = pd.concat([combined_df, combined_df_stat], ignore_index=True)
    
    
    combined_df.to_csv(f"{directory}/{new_string2}/FINAL_{mode}_{directory.split('/')[-1]}.csv", index=False)



    return combined_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')

    parser.add_argument('--directory', type=str, default='', help='Description of directory argument')
    parser.add_argument('--mode', type=str, default='', help='Description of directory argument')
    args = parser.parse_args()

    directory = args.directory
    mode = args.mode
    combine_csv_files(directory, mode)
