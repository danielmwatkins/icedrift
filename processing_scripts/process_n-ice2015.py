"""The N-ICE2015 data have already been quality controlled. This script essentially just makes sure that the data pass the same checks I use for other datasets by applying the cleaning.standard_qc algorithm. I only process the buoys that were used in Itkin et al. 2017 "thin ice and storms" and only saves the tracks that are at least 30 days long. It also reformats the data from JSON to CSV."""

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

dataloc = '../../data/NICE2015/'
saveloc = '../../data/processed/n-ice2015/'
sic_loc = '/Users/dwatkin2/Documents/research/data/nsidc_daily_cdr/'
interploc = '../../data/interpolated/n-ice2015/'

files = os.listdir(dataloc)
files = [f for f in files if f.split('.')[-1] == 'json']
nice2015_data = {}
for file in files:
    data = json.load(open(dataloc + file))
    coords = np.array(data['geometry']['coordinates'])
    lon = coords[:, 0]
    lat = coords[:, 1]
    qc = data['properties']['quality']
    time = [pd.to_datetime(x.replace('Z', '')) for x in data['properties']['measured']]
    buoy = data['properties']['buoy']
    nice2015_data[buoy] = pd.DataFrame({'longitude': lon, 'latitude': lat, 'qc': qc}, index=time).sort_index()

not_used = ['SIMBA_2015b', 'SIMBA2015e',
            'SIMBA_2015f', 'WAVE_2015a',
            'STRESS_2015a', 'IMB-B_2015a',
            'IMB-B_2015b', 'IMB-B_2015c',
            'IMB-2_2015a', 'IC_2015a',
            'IMB_2016a'] # This one is from a later expedition

too_short = [b for b in nice2015_data if (b not in not_used) & \
             ((nice2015_data[b].index[-1] - nice2015_data[b].index[0]) < pd.to_timedelta('30D'))]

buoys = [b for b in nice2015_data if b not in not_used + too_short]

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

for buoy in buoys:
    buoy_init = nice2015_data[buoy].loc[nice2015_data[buoy].qc==1].copy()
    
    df = cleaning.standard_qc(buoy_init)
    df = df.loc[~df.flag]
    if len(df) > 0:
        df.to_csv(saveloc + 'n-ice2015_' + buoy + '.csv')    
        freq = get_frequency(buoy_init)
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
        
        df.to_csv(interploc + buoy + '.csv')    