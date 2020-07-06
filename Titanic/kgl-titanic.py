# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 21:14:19 2020

a try to make this code work from this page:
https://stackoverflow.com/questions/43516982/import-kaggle-csv-from-download-url-to-pandas-dataframe    

@author: Olek
"""
import os
import requests
# from os.path import join
# path = os.path.abspath('D:\Scripts\Python\Kaggle\Titanic')

def from_kaggle(data_sets, competition):
    """Fetches data from Kaggle

    Parameters
    ----------
    data_sets : (array)
        list of dataset filenames on kaggle. (e.g. train.csv.zip)

    competition : (string)
        name of kaggle competition as it appears in url
        (e.g. 'rossmann-store-sales')

    """
    kaggle_dataset_url = "https://www.kaggle.com/c/{}/download/".format(competition)

    KAGGLE_INFO = {'UserName': 'oleksandrpogorilyi',
                   'Password': 'Madman1983apricot1984!'}

    for data_set in data_sets:
        data_url = os.path.join(kaggle_dataset_url, data_set)
        data_output = os.path.join('.', data_set)
        # Attempts to download the CSV file. Gets rejected because we are not logged in.
        r = requests.get(data_url)
        # Login to Kaggle and retrieve the data.
        r = requests.post(r.url, data=KAGGLE_INFO, stream=True)
        # Writes the data to a local file one chunk at a time.
        with open(data_output, 'wb') as f:
            # Reads 512KB at a time into memory
            for chunk in r.iter_content(chunk_size=(512 * 1024)):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    
#%% Example use:
sets = ['train.csv.zip',
        'test.csv.zip',
        'store.csv.zip',
        'sample_submission.csv.zip',]
from_kaggle(sets, 'rossmann-store-sales')

#%% does not work. Obtained zip files of data sets are empty (0 bytes)