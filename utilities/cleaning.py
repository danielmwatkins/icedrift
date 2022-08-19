"""Utility functions for flagging nonphysical behavior in drift tracks.
Currently, columns are added. This should be optional.
"""

import pandas as pd
import numpy as np
import pyproj

def flag_duplicates(buoy_df):
    """Returns a boolean Series object with the same index
    as buoy_df with True where multiple reports for a given
    time or coordinate exist. Times are rounded to the nearest minute prior
    to comparison. Expects 'date' to be a column in buoy_df. Latitude and
    longitude are rounded to the 4th decimal place.
    
    Data are flagged if
    1. A date is repeated anywhere
    2. If the exact longitude-latitude pair is repeated anywhere
    3. If a latitude point is repeated
    4. If a longitude point is repeated
    
    Number 2 is distinct from 3 and 4 in that it allows repeated patterns to be removed.
    Numbers 3 and 4 are potentially overly restrictive, as it may be the case that a buoy moves 
    due east/west or due north/south.
    """

    date = pd.to_datetime(buoy_df.date).round('1min')
    lats = buoy_df.latitude.round(4)
    lons = buoy_df.longitude.round(4)
    duplicated_times = date.duplicated(keep='first')
    
    # single repeat
    repeated_lats = lats.shift(1) == lats    
    repeated_lons = lons.shift(1) == lons
     
    duplicated_latlon = pd.Series([(x, y) for x, y in zip(buoy_df.longitude.round(4), buoy_df.latitude.round(4))],
                                  index=buoy_df.index).duplicated(keep='first')
    
    
    flag = duplicated_times + repeated_lats + repeated_lons + duplicated_latlon
    return flag > 0

def check_times(buoy_df):
    """Check if there are reversals in the time or if data are isolated in time"""

    threshold = 12*60*60 # threshold is 12 hours
    
    date = pd.to_datetime(buoy_df.date)
    
    time_till_next = date.shift(-1) - date
    time_since_last = date - date.shift(1)

    negative_timestep = time_since_last.dt.total_seconds() < 0
    gap_too_large = (time_till_next.dt.total_seconds() > threshold) | (time_since_last.dt.total_seconds() > threshold)
    
    return negative_timestep | gap_too_large

def check_speed(buoy_df):
    """Flags if the buoy drift speed is faster than 1.5 m/s."""
    projIn = 'epsg:4326' # WGS 84 Ellipsoid
    projOut = 'epsg:3571' # Lambert Azimuthal Equal Area centered at north pole, lon0 is 180
    transformer = pyproj.Transformer.from_crs(projIn, projOut, always_xy=True)

    lon = buoy_df.longitude.values
    lat = buoy_df.latitude.values

    x, y = transformer.transform(lon, lat)
    buoy_df['x'] = x
    buoy_df['y'] = y

    dt = (buoy_df.date.shift(-1) - buoy_df.date.shift(1)).dt.total_seconds()

    def centered_velocity(xvar, yvar, data, dt=3600):
        """Assumes the rows are a datetime index with 30 min step size."""
        dx = data[xvar].shift(-1) - data[xvar].shift(1)
        dy = data[yvar].shift(-1) - data[yvar].shift(1)
        return dx/dt, dy/dt

    dxdt, dydt = centered_velocity('x', 'y', buoy_df, dt)
    buoy_df['u'] = dxdt
    buoy_df['v'] = dydt
    buoy_df['speed'] = np.sqrt(buoy_df['v']**2 + buoy_df['u']**2)
    
    return buoy_df['speed'] > 1.5 