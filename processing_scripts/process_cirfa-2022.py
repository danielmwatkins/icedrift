"""Reading and formatting the CIRFA2022 buoy data. Applies the standard_qc routine.
Interpolates data to 30 minutes and saves results into the interpolated folder.
Adds daily sea ice concentration from the NSIDC CDR product."""

import pandas as pd
import json
import numpy as np
import os
import pyproj
import sys
sys.path.append('../../') # path to drifter package
from drifter import cleaning
from drifter import interpolation
import xarray as xr

dataloc = '../../data/CIRFA2022/'
sic_loc = '/Users/dwatkin2/Documents/research/data/nsidc_daily_cdr/'
saveloc = '../../data/processed/cirfa2022/'
interploc = '../../data/interpolated/cirfa2022/'


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

# Not used here but could be added
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
    return f  

cirfa22 = {}
for path in filepaths:
    name = 'CIRFA22_' + path.split('_')[-2]
    df = pd.read_fwf(path, index_col=False, skiprows=1, header=None, names=['date', 'time', 'latitude', 'longitude'])
    df.index = [pd.to_datetime(d + ' ' + h) for d, h in zip(df['date'], df['time'])]
    df2 = cleaning.standard_qc(df.round(4), lat_range=(50, 90))
    if df2 is not None:
        df.loc[~df2.flag, ['latitude', 'longitude']].to_csv(
            saveloc + name + '.csv')
        df = interpolation.interpolate_buoy_track(df,
                                                  xvar='longitude', yvar='latitude', 
                                                  freq='30min', maxgap_minutes=120)
        
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
        
        df.to_csv(interploc + name + '.csv')