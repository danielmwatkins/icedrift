"""Apply cleaning and interpolation to the buoys in the MOSAiC distributed network."""

# Package imports
import numpy as np
import os
import pandas as pd
import sys

sys.path.append('../../drifter/')
import utilities.cleaning as clean
import utilities.interpolation as interp

# Folder with the drift tracks from 
# the Arctic Data Center (Bliss et al. 2021)
# https://arcticdata.io/catalog/view/urn%3Auuid%3A2b859a39-cada-4147-819a-dc222f6f89a3
dataloc = '../../data/adc_dn_tracks/'
saveloc = '../data/mosaic_interpolated/'
# Read in the files, including the metadata file
files = os.listdir(dataloc)
files = [f for f in files if f.split('.')[-1] == 'csv']
files = [f for f in files if f.split('_')[0] != 'DN']

metadata = pd.read_csv(dataloc + 'DN_buoy_list_v2.csv')
metadata['filename'] = ['_'.join([x, str(y), z]) for 
                        x, y, z in zip(metadata['DN Station ID'],
                                       metadata['IMEI'],
                                       metadata['Sensor ID'])]

# Optional: focus in on only a portion of the tracks
begin = '2019-10-15-00 00:00'
end = '2022-01-01 00:00'
    
file_present = np.array([f + '.csv' in files for f in metadata.filename])
metadata = metadata.loc[file_present].copy()
metadata.set_index('Sensor ID', inplace=True)

# Maximum gap in hours for the interpolation algorithm
max_dt = 2

# Pad for interpolation, so that the first and last values aren't missing.
margin = pd.to_timedelta('6H')
buoy_data = {}
for file in metadata.filename:
    sensorweb_id = file.split('_')[-1]
    data = pd.read_csv(dataloc + file + '.csv',
                       index_col='datetime', parse_dates=True)
    check_dup = clean.flag_duplicates(data, date_index=True)
    check_dates = clean.check_dates(data, date_index=True)
    
    data = data.where(~(check_dup | check_dates)).dropna()
    if len(data.loc[slice(begin, end)]) > 30*24:
        # First speed check on original time axis, to remove
        # bad data and to calculate
        data = clean.compute_speed(data, date_index=True, difference='backward')    
        data['speed_flag'] = clean.check_speed(data, date_index=True, method='z-score', sigma=6, window='3D')
        
        data = data.where(~data.speed_flag).dropna()
        
        if len(data.loc[slice(begin, end)]) > 30*24:
            dt = pd.to_timedelta(np.diff(data.index)).median().seconds/3600

            if np.round(dt,1) <= max_dt:
                # interpolate before saving
                # Is it worth adding an extra step so that the 
                # interpolation is in LAEA space?
                data_interp = interp.interpolate_buoy_track(
                    data.loc[slice(pd.to_datetime(begin)-margin,
                                   pd.to_datetime(end)+margin)],
                    xvar='longitude', yvar='latitude', freq='1H',
                    maxgap_minutes=240)

                buoy_data[sensorweb_id] = clean.compute_speed(
                    data_interp, date_index=True, rotate_uv=True, difference='centered').loc[slice(begin, end)]
            else:
                print(sensorweb_id, dt, 'Frequency too low')
        else:
            print(sensorweb_id, 'Insufficient data after speed check: ', len(data))
    else:
        print(sensorweb_id, 'Insufficient data after dup/date checks: ', len(data))

for buoy in buoy_data:
    buoy_data[buoy].to_csv(saveloc + metadata.loc[buoy, 'filename'] + '.csv')