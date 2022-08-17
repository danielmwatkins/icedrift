"""Utility functions for flagging nonphysical behavior in drift tracks."""

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
    print(buoy_df.head())
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
    """Check if there are reversals in the time"""
    date = pd.to_datetime(buoy_df.date)
    dt = date.shift(1) - date
    dt = dt.total_seconds()
    return dt < 0

def check_for_jumps(buoy_df):
    """Look for unphysical jumps in the buoy tracks"""
    
    return