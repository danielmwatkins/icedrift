"""Utility functions for flagging nonphysical behavior in drift tracks."""

import pandas as pd
import numpy as np

def flag_duplicates(buoy_df):
    """Returns a boolean Series object with the same index
    as buoy_df with True where multiple reports for a given
    time or location exist. Times are rounded to the nearest minute prior
    to comparison. Expects 'date' to be a column in buoy_df. Latitude and
    longitude are rounded to the 4th decimal place.
    
    TBD: check for duplicated patterns (repeats of exact matches of lat/lon)
    """
    print(buoy_df.head())
    date = pd.to_datetime(buoy_df.date).round('1min')
    duplicated_times = date.duplicated(keep='first')
    repeated_lats = buoy_df.latitude.round(4).shift(1) == buoy_df.latitude.round(4)
    repeated_lons = buoy_df.longitude.round(4).shift(1) == buoy_df.longitude.round(4)
    
    return duplicated_times | (repeated_lats & repeated_lons)

def check_times(buoy_df):
    """Check if there are reversals in the time"""
    return

def check_for_jumps(buoy_df):
    """Look for unphysical jumps in the buoy tracks"""
    return