"""Uses the splines method to identify problem points before interpolation"""

# Package imports
import numpy as np
import os
import pandas as pd
import sys
from scipy.spatial.distance import cdist
sys.path.append('../../drifter/')
import utilities.cleaning as clean
import utilities.interpolation as interp

# Folder with the drift tracks from 
# the Arctic Data Center (Bliss et al. 2021)
# https://arcticdata.io/catalog/view/urn%3Auuid%3A2b859a39-cada-4147-819a-dc222f6f89a3
dataloc = '../../data/adc_dn_tracks/'
saveloc = '../data/mosaic_interpolated/'
# Maximum gap in hours for the interpolation algorithm
max_dt = 2

# Pad for interpolation, so that the first and last values aren't missing.
margin = pd.to_timedelta('6H')
buoy_data = {}


# Optional: focus in on only a portion of the tracks
begin = '2019-10-15 00:00'
end = '2022-01-01 00:00'

# Read in the files, including the metadata file
files = os.listdir(dataloc)
files = [f for f in files if f.split('.')[-1] == 'csv']
files = [f for f in files if f.split('_')[0] != 'DN']

metadata = pd.read_csv(dataloc + 'DN_buoy_list_v2.csv')
metadata['filename'] = ['_'.join([x, str(y), z]) for 
                        x, y, z in zip(metadata['DN Station ID'],
                                       metadata['IMEI'],
                                       metadata['Sensor ID'])]
    
file_present = np.array([f + '.csv' in files for f in metadata.filename])
metadata = metadata.loc[file_present].copy()
metadata.set_index('Sensor ID', inplace=True)

buoy_data = {}
for file in files:
    buoy_data[file.replace('.csv', '').split('_')[-1]] = pd.read_csv(dataloc + file,
                                            index_col='datetime', parse_dates=True)

# Step one: compute dist_from_median_uv, dist_from_median_xy, distance to closest point
for buoy in buoy_data:
    data = buoy_data[buoy]
    fwd_speed = clean.compute_speed(data.copy(), date_index=True, difference='forward')   
    bwd_speed = clean.compute_speed(data.copy(), date_index=True, difference='backward')
    speed = pd.DataFrame({'b': bwd_speed['speed'], 'f': fwd_speed['speed']}).min(axis=1)
    u = pd.DataFrame({'b': np.abs(bwd_speed['u']), 'f': np.abs(fwd_speed['u'])}).min(axis=1)
    v = pd.DataFrame({'b': np.abs(bwd_speed['v']), 'f': np.abs(fwd_speed['v'])}).min(axis=1)    
    buoy_data[buoy]['speed_bf'] = speed
    buoy_data[buoy]['x'] = fwd_speed['x']
    buoy_data[buoy]['y'] = fwd_speed['y']

    check_dup = clean.flag_duplicates(data, date_index=True)
    check_dates = clean.check_dates(data, date_index=True)
    check_speed = clean.check_speed(data, date_index=True, sigma=10, window=5)

    flags = (check_dup | check_dates) | check_speed
    
    data = data.where(~flags).dropna()
    
    # Add check with the spline function
    
    if len(data.loc[slice(begin, end)]) > 30*24:        
        dt = pd.to_timedelta(np.diff(data.index)).median().seconds/3600
        if np.round(dt,1) <= max_dt:
            data_interp = interp.interpolate_buoy_track(
                data.loc[slice(pd.to_datetime(begin)-margin,
                               pd.to_datetime(end)+margin)],
                xvar='longitude', yvar='latitude', freq='1H',
                maxgap_minutes=240)
            data_interp['day_count'] = data_interp.rolling(window='1D', center=True).count()['longitude']
            data_interp = data_interp.where(data_interp.day_count >= 12).dropna()
                
            # Compute speed with interpolated data
            data_interp = clean.compute_speed(
                data_interp, date_index=True, rotate_uv=True, difference='centered').loc[slice(begin, end)]
            
            # Write to file
            data_interp.to_csv(saveloc + metadata.loc[buoy, 'filename'] + '.csv')
            
        else:
            print(buoy, dt, 'Frequency too low')
    else:
        print(buoy, 'Insufficient data after dup/date/speed check: ', len(data))


