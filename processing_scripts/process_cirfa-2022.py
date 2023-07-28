"""Reading and formatting the CIRFA2022 buoy data"""

import pandas as pd
import json
import numpy as np
import os
import sys
sys.path.append('../')
from src import cleaning

dataloc = '../../data/CIRFA2022/'
saveloc = '../../data/processed/cirfa2022/'

deployments = ['2022_04_24', '2022_04_25', '2022_04_30',
               '2022_05_06', '2022_04_26', '2022_05_02',
               '2022_05_05', '2022_05_04', '2022_05_03']
filepaths = []
for date in deployments:
    files = os.listdir(dataloc + date)
    for file in files:
        if 'CIRFA' in file:
            subfiles = os.listdir(dataloc + date + '/' + file)
            for sfile in subfiles:
                if 'trajectory' in sfile:
                    filepaths.append(os.path.join(dataloc, date, file, sfile))

cirfa22 = {}
for path in filepaths:
    name = 'CIRFA22_' + path.split('_')[-2]
    df = pd.read_fwf(path, index_col=False, skiprows=1, header=None, names=['date', 'time', 'latitude', 'longitude'])
    df.index = [pd.to_datetime(d + ' ' + h) for d, h in zip(df['date'], df['time'])]
    df2 = cleaning.standard_qc(df.round(4), lat_range=(50, 90))
    if df2 is not None:
        df.loc[~df2.flag, ['latitude', 'longitude']].to_csv(
            saveloc + buoy + '.csv')