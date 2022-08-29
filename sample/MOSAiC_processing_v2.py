"""Uses the XA/UA method for identifying problem points before applying interpolation"""

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

begin = '2019-10-15 00:00'
end = '2022-01-01 00:00'
max_dt = 2
margin = pd.to_timedelta('6H')

# Window can be either based on time window or on the an integer number of neighbors.
window = 5
eps = 1





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

buoy_data = {}
for file in files:
    buoy_data[file.replace('.csv', '').split('_')[-1]] = pd.read_csv(dataloc + file,
                                            index_col='datetime', parse_dates=True)



def dist_from_median(xvar, yvar, data, window):
    """Computes the distance of the point from the median of a moving window."""
    xa = data[xvar] - data[xvar].rolling(window, center=True).median()
    ya = data[yvar] - data[yvar].rolling(window, center=True).median()
    
    return np.sqrt(xa**2 + ya**2)

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
    buoy_data[buoy]['u'] = u
    buoy_data[buoy]['v'] = v

    data['dist_from_median_uv'] = dist_from_median('u', 'v', data, window)
    data['dist_from_median_xy'] = dist_from_median('x', 'y', data, window)

    # standardize
    # Attempting to use rolling 30-day interquartile range
    for var in ['dist_from_median_uv', 'dist_from_median_xy']:
        #X = data[var]
        #q75 = X[X>0].quantile(0.75)
        #q25 = X[X>0].quantile(0.25)
        q75 = data[var].where(data[var] > 0).rolling('30D', center=True, min_periods=15).quantile(0.75)
        q25 = data[var].where(data[var] > 0).rolling('30D', center=True, min_periods=15).quantile(0.25)
        data[var] = data[var]/(q75-q25)

    index = data.dropna().index
    X = data.loc[index, ['dist_from_median_uv', 'dist_from_median_xy']].values
    d = cdist(X, X, 'euclidean')
    data.loc[index, 'NN'] = (d < eps).sum(axis=1) # Number of data points  closer than eps        
    closest = []
    for idx in range(len(index)):
        dmin = d[idx,:]
        closest.append(dmin[dmin > 0].min())
    data.loc[index, 'closest'] = closest
    buoy_data[buoy] = data.copy()
    buoy_data[buoy].to_csv('../../data/annotated_adc_dn_tracks/' + metadata.loc[buoy,'filename'] + '.csv')

for buoy in buoy_data:
    data = buoy_data[buoy]
    sensorweb_id = buoy
    check_dup = clean.flag_duplicates(data, date_index=True)
    check_dates = clean.check_dates(data, date_index=True)
    
    data = data.where(~(check_dup | check_dates)).dropna()
    if len(data.loc[slice(begin, end)]) > 30*24:
        data['speed_flag'] = (data.dist_from_median_uv > 50) | (data.dist_from_median_xy > 50)
        data = data.where(~data.speed_flag).dropna()        
        
        if len(data.loc[slice(begin, end)]) > 30*24:
            dt = pd.to_timedelta(np.diff(data.index)).median().seconds/3600

            if np.round(dt,1) <= max_dt:
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