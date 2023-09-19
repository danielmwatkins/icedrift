"""Data cleaning for MOSAiC data. These datasets
"""
import pandas as pd
import json
import numpy as np
import os
import sys
sys.path.append('../../') # path to drifter package
from drifter import cleaning
from drifter import interpolation
import xarray as xr
import pyproj

dataloc = '../../data/adc_dn_tracks/'
sic_loc = '/Users/dwatkin2/Documents/research/data/nsidc_daily_cdr/'

saveloc_dn1 = '../../data/processed/mosaic_dn1/'
saveloc_dn2 = '../../data/processed/mosaic_dn2/'

interploc_dn1 = '../../data/interpolated/mosaic_dn1/'
interploc_dn2 = '../../data/interpolated/mosaic_dn2/'


files = os.listdir(dataloc)
files = [f for f in files if f[0] not in ['.', 'S', 'D']]

metadata = pd.read_csv('../../data/adc_dn_tracks/DN_buoy_list_v2.csv').set_index('Sensor ID')

## List of V buoys with missing (-) in longitudes after crossing meridian
# Thanks Angela for finding these! Should be updated in the ADC drift set v3.
v_fix_list = {'M2_300234067064490_2019V2.csv': '2020-07-26 17:58:08',
              'M3_300234067064370_2019V3.csv': '2020-07-11 23:58:05',
              'M5_300234067066520_2019V4.csv': '2020-07-10 00:58:09'}

def sic_along_track(position_data, sic_data):
    """Uses the xarray advanced interpolation to get along-track sic
    via nearest neighbors. Nearest neighbors is preferred because numerical
    flags are used for coasts and open ocean, so interpolation is less meaningful."""
    # Sea ice concentration uses NSIDC NP Stereographic
    crs0 = pyproj.CRS('WGS84')
    crs1 = pyproj.CRS('epsg:3411')
    transformer_stere = pyproj.Transformer.from_crs(crs0, crs_to=crs1, always_xy=True)
    
    sic = pd.Series(data=np.nan, index=position_data.index)
    
    for date, group in position_data.groupby(position_data.index.date):
        x_stere, y_stere = transformer_stere.transform(
            group.longitude, group.latitude)
        
        x = xr.DataArray(x_stere, dims="z")
        y = xr.DataArray(y_stere, dims="z")
        SIC = sic_data.sel(time=date.strftime('%Y-%m-%d'))['sea_ice_concentration'].interp(
            {'x': x,
             'y': y}, method='nearest').data
        
        sic.loc[group.index] = np.round(SIC.T, 3)
    sic[sic > 100] = np.nan
    return sic   

def get_frequency(buoy_df):
    """Calculates the median frequency and returns as
    an integer number of minutes. Prints warning if the
    maximum and minimum of 7D aggregates is different."""
    t = buoy_df.index.to_series()
    dt = t - t.shift(1)
    f = int(np.round(dt.median().total_seconds()/60, 0))
    # Check if representative of weekly data
    fmax = int(np.round(dt.resample('7D').median().max().total_seconds()/60, 0))
    fmin = int(np.round(dt.resample('7D').median().min().total_seconds()/60, 0))
    if (np.abs(f - fmax) > 0) | (np.abs(f - fmin) > 0):
        print('Warning: buoy has varying frequency. fmin=', fmin, 'fmax=', fmax, 'f=', f)
        
    if f <= 30:
        interp_freq = '30min'
    elif f <= 65: # There's a couple that are at 61, which is certainly an accident
        interp_freq = '60min'
    else:
        interp_freq = str(np.round(f, -1)) + 'min'


    return interp_freq

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
        df = df_qc.loc[~df_qc.flag, ['latitude', 'longitude']]
        
        if metadata.loc[buoy, 'Deployment Leg'] == 5:
            df.to_csv(saveloc_dn2 + buoy + '.csv')
        else:
            df.to_csv(saveloc_dn1 + buoy + '.csv')
            
        # interpolate appropriately
        # add sea ice concentration
        freq = get_frequency(df)
        print(buoy, freq)
        maxgap = 4 * int(freq.replace('min', ''))
        df = interpolation.interpolate_buoy_track(df,
                                                  xvar='longitude', yvar='latitude', 
                                                  freq=freq, maxgap_minutes=max(maxgap, 120))
        
        # add sea ice concentration from daily cdr
        dfs_by_year = {year: group for year, group in df.groupby(df.index.year)}
        
        for year in dfs_by_year:
            with xr.open_dataset(sic_loc + '/aggregate/seaice_conc_daily_nh_' + \
                         str(year) + '_v04r00.nc') as sic_data:
                
                ds = xr.Dataset({'sea_ice_concentration':
                                 (('time', 'y', 'x'), sic_data['cdr_seaice_conc'].data)},
                           coords={'time': (('time', ), sic_data['time'].data),
                                   'x': (('x', ), sic_data['xgrid'].data), 
                                   'y': (('y', ), sic_data['ygrid'].data)})

                sic = sic_along_track(dfs_by_year[year], ds)
                dfs_by_year[year]['sea_ice_concentration'] = sic
                
        df_with_sic = pd.concat(dfs_by_year).reset_index(drop=True)
        df['sea_ice_concentration'] = np.round(df_with_sic['sea_ice_concentration'].values*100, 0)
        # values in netcdf are fractions, but the flag values refer to percentages, so we bring it back into percentages
        
        if metadata.loc[buoy, 'Deployment Leg'] == 5:
            df.to_csv(interploc_dn2 + buoy + '.csv')
        else:
            df.to_csv(interploc_dn1 + buoy + '.csv')
          