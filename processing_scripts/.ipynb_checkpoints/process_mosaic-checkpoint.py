"""Data cleaning for MOSAiC data
"""
import pandas as pd
import json
import numpy as np
import os
import sys
sys.path.append('../')
from src import cleaning

dataloc = '../../data/adc_dn_tracks/'
saveloc_dn1 = '../../data/processed/mosaic_DN1/'
saveloc_dn2 = '../../data/processed/mosaic_DN2/'

files = os.listdir(dataloc)
files = [f for f in files if f[0] not in ['.', 'S', 'D']]

metadata = pd.read_csv('../../data/adc_dn_tracks/DN_buoy_list_v2.csv').set_index('Sensor ID')

mosaic_data = {}

## List of V buoys with missing (-) in longitudes after crossing meridian
# Thanks Angela for finding these! Should be updated in the ADC drift set v3.
v_fix_list = {'M2_300234067064490_2019V2.csv': '2020-07-26 17:58:08',
              'M3_300234067064370_2019V3.csv': '2020-07-11 23:58:05',
              'M5_300234067066520_2019V4.csv': '2020-07-10 00:58:09'}

for file in files:
    buoy = file.split('_')[-1].replace('.csv', '')
    df = pd.read_csv(dataloc + file, index_col='datetime', parse_dates=True)

    # Adjust V buoys to UTC from Beijing time
    if 'V' in buoy:
        df.index = df.index - pd.to_timedelta('8H')

    # Apply correction to longitude issue for 3 V buoys
    if file in v_fix_list:
        time = pd.to_datetime(v_fix_list[file])
        df_subset = df[time:]
        df_subset.loc[:, 'longitude'] = df_subset.loc[:, 'longitude']*-1
        df.update(df_subset)
        if 'M5' in file.split('_'):        
            df_subset = df['2020-07-10 07:58:06':'2020-07-10 09:58:28']
            df_subset.longitude = df_subset.longitude*-1
            df.update(df_subset)

    df_qc = cleaning.standard_qc(df,
                        min_size=100,
                        gap_threshold='6H',                
                        segment_length=24,
                        lon_range=(-180, 180),
                        lat_range=(50, 90),
                        max_speed=1.5,
                        speed_window='3D',
                        verbose=False)
    if df_qc is not None:
        mosaic_data[buoy] = df_qc.loc[~df_qc.flag, ['latitude', 'longitude']]
        
for buoy in mosaic_data:
    if metadata.loc[buoy, 'Deployment Leg'] == 5:
        mosaic_data[buoy].to_csv(saveloc_dn2 + buoy + '.csv')
    else:
        mosaic_data[buoy].to_csv(saveloc_dn1 + buoy + '.csv')        