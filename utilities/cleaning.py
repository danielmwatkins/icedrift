"""Utility functions for flagging nonphysical behavior in drift tracks.

Functions starting with "check" return a boolean Series with True where the 
data is likely bad.



TBD: Currently, columns are added. This should be optional.
"""

import pandas as pd
import numpy as np
import pyproj
from .analysis import compute_velocity

def check_positions(buoy_df, pairs_only=False,
                   latname='latitude', lonname='longitude'):
    """Looks for duplicated or nonphysical position data. Defaults to masking any 
    data with exact matches in latitude or longitude. Setting pairs_only to false 
    restricts the check to only flag where both longitude and latitude are repeated
    as a pair.
    """

    lats = buoy_df[latname].round(10)
    lons = buoy_df[lonname].round(10)
    
    invalid_lats = np.abs(lats) > 90
    if np.any(lons < 0):
        invalid_lons = np.abs(lons) > 180
    else:
        invalid_lons = lons > 360
        
    invalid = invalid_lats | invalid_lons
    
    repeated = lats.duplicated(keep='first') | lons.duplicated(keep='first')
    
    duplicated = pd.Series([(x, y) for x, y in zip(lons, lats)],
                                  index=buoy_df.index).duplicated(keep='first')
    
    if pairs_only:
        return duplicated | invalid
    
    else:
         return repeated | duplicated | invalid


def check_dates(buoy_df, precision='1min', date_col=None):
    """Check if there are reversals in the time or duplicated dates. Optional: check
    whether data are isolated in time based on specified search windows and the threshold
    for the number of buoys within the search windows. Dates are rounded to <precision>,
    so in some cases separate readings that are very close in time will be flagged
    as duplicates. Assumes date_col is in a format readable by pandas to_datetime.
    """

    if date_col is None:
        date_values = buoy_df.index.values
        date = pd.Series(pd.to_datetime(date_values).round(precision),
                     index=buoy_df.index)
    else:
        date = pd.to_datetime(buoy_df[date_col]).round(precision)
    duplicated_times = date.duplicated(keep='first')
    
    time_till_next = date.shift(-1) - date
    time_since_last = date - date.shift(1)

    negative_timestep = time_since_last.dt.total_seconds() < 0

    return negative_timestep | duplicated_times
    

def check_gaps(buoy_df, threshold_gap='4H', threshold_segment=12, date_col=None):
    """Segments the data based on a threshold of <threshold_gap>. Segments shorter
    than <threshold_segment> are flagged."""
    
    if date_col is None:
        date_values = buoy_df.index.values
        date = pd.Series(pd.to_datetime(date_values),
                     index=buoy_df.index)
    else:
        date = pd.to_datetime(buoy_df[date_col])
    
    time_till_next = date.shift(-1) - date
    segment = pd.Series(0, index=buoy_df.index)
    counter = 0
    tg = pd.to_timedelta(threshold_gap)
    for t in segment.index:
        segment.loc[t] = counter
        if time_till_next[t] > tg:
            counter += 1
    
    # apply_filter
    new = buoy_df.groupby(segment).filter(lambda x: len(x) > threshold_segment).index
    flag = pd.Series(True, index=buoy_df.index)
    flag.loc[new] = False
    return flag

    
def fit_splines(date, data, xvar='x', yvar='y', zvar=None, df=25):
    """Fit regression model using natural cubic splines after
    removing 'date', and evaluate at 'date'.
    Returns dataframe with columns xvar, yvar, xvar_hat, yvar_hat,
    and err = sqrt((x-xhat)^2 + (y-yhat)^2)"""
    from sklearn.linear_model import LinearRegression
    from patsy import cr
    
    data_fit = data.drop(date)
    tfit = data_fit.index
    xfit = (tfit - tfit[0]).total_seconds()
    if zvar is not None:
        yfit = data_fit[[xvar, yvar, zvar]]
    else:
        yfit = data_fit[[xvar, yvar]]
        
    x_basis = cr(xfit, df=df, constraints="center")
    model = LinearRegression().fit(x_basis, yfit)

    if zvar is not None:

        t = data.index
        x = (t - t[0]).total_seconds()
        y = data[[xvar, yvar, zvar]]
        x_basis = cr(x, df=df, constraints="center")

        y_hat = model.predict(x_basis)
        fitted = pd.DataFrame(y_hat, index=t, columns=[xvar + '_hat', yvar + '_hat', zvar + '_hat'])
        fitted[xvar] = y[xvar]
        fitted[yvar] = y[yvar]
        fitted[zvar] = y[zvar]
        
        fitted['x_err'] = np.sqrt((fitted[xvar + '_hat'] - fitted[xvar])**2 + (fitted[yvar + '_hat'] - fitted[yvar])**2)
        fitted[zvar + '_err'] = fitted[zvar + '_hat'] - fitted[zvar]
        
        return fitted.loc[:, [xvar, yvar, zvar, xvar + '_hat', yvar + '_hat', zvar + '_hat', 'x_err', zvar + '_err']]
    
    else:
        t = data.index
        x = (t - t[0]).total_seconds()
        y = data[[xvar, yvar]]
        x_basis = cr(x, df=df, constraints="center")

        y_hat = model.predict(x_basis)
        fitted = pd.DataFrame(y_hat, index=t, columns=[xvar + '_hat', yvar + '_hat'])
        fitted[xvar] = y[xvar]
        fitted[yvar] = y[yvar]
        fitted['err'] = np.sqrt((fitted[xvar + '_hat'] - fitted[xvar])**2 + (fitted[yvar + '_hat'] - fitted[yvar])**2)
        return fitted.loc[:, [xvar, yvar, xvar + '_hat', yvar + '_hat', 'err']]

def test_point(date, data, xvar, yvar, df, fit_window, sigma):
    """Tests whether a point is within the expected range of a smoothed path.
    The smoothed path is computed by fitting a regression model with natural
    cubic splines on the data within (date-fit_window, date+fit_window) excluding
    the test point. The distance between the point and the predicted position of
    the point is compared to sigma * the standard deviation of the residual. Returns
    True if the distance is greater, False if less."""

    margin = pd.to_timedelta(fit_window)
    if type(date) == str:
        date = pd.to_datetime(date)
    fit_ts = slice(date - margin, date + margin)
    fit_df = fit_splines(date, data.loc[fit_ts], xvar, yvar, df)

    err_stdev = fit_df.drop(date)['err'].std()
    fit_df['flag'] = fit_df['err'] > (sigma * err_stdev)
    return fit_df

def check_position_splines(data, xvar, yvar, df, fit_window, sigma):
    """Use natural cubic splines to model the buoy track, flagging data where the difference between the modeled
    and actual tracks is large."""

    margin = pd.to_timedelta(fit_window)
    data['flag'] = 0
    test_dates = data.loc[slice(data.index.min() + margin, data.index.max() -  margin)].index
    
    for date in test_dates:
        test_fit = test_point(date, data.loc[data['flag'] != 1], xvar, yvar, df=df, fit_window='48H', sigma=10)
        if test_fit.loc[date, 'flag']:
            # Don't flag points right next to large gaps
            # TBD: Implement method to check error prior to gaps
            t = test_fit.index.to_series()
            dt_next = t.shift(-1) - t
            dt_prior = t - t.shift(1)
            gap_threshold = pd.to_timedelta('4H')
            if (dt_next[date] < gap_threshold) & (dt_prior[date] < gap_threshold):
                data.loc[date, 'flag'] = 1
                print('Flagged date ', date)

    return data['flag']

def identify_outliers(buoy_df, error_thresh, fit_margin, sigma=6, detailed_return=False):
    """Flags data that are likely outliers based on three criteria:
    1. Data have anom_dist > sigma*anom_std
    2. anom_dist is a local max
    3. speed is a local max
    4. Interpolation error is greater than the error_threshold
    Returns a boolean series of the same length as buoy_df, unless
    detailed_return=True, in which case a dataframe with the tested values is returned."""

    def est_middle(date, data, xvar, yvar):
        from scipy.interpolate import interp1d
        """Similar to the savgol filter, estimate the value at date with a polynomial fit.
        """
        t0 = (data.drop(date).index - data.index[0]).total_seconds()
        t1 = (date - data.index[0]).total_seconds()

        X = data.drop(date).loc[:,[xvar, yvar]].T
        return interp1d(t0, X.values, bounds_error=False, kind='cubic')(t1).T
   
    fit_margin = pd.to_timedelta(fit_margin)
    anom_std = np.sqrt(2 * buoy_df['anom_dist'].where(buoy_df['anom_dist'] > 0).mean())
    test_dates = buoy_df[['anom_dist', 'speed']][buoy_df['anom_dist'] > sigma*anom_std]
    test_dates = test_dates.sort_values('anom_dist')[::-1]

    #anom_above_threshold = buoy_df['anom_dist'] > (buoy_df.where(buoy_df.anom_dist > 0)['anom_dist']).rolling('30D', center=True).median()*2
    #speed_above_threshold = buoy_df['speed'] > buoy_df['speed'].rolling('30D', center=True).median()*2
    anom_local_max = buoy_df['anom_dist'] == buoy_df['anom_dist'].rolling(fit_margin, center=True).max()
    speed_local_max = buoy_df['speed'] == buoy_df['speed'].rolling(fit_margin, center=True).max()

    test_dates['anom_max'] = anom_local_max.loc[test_dates.index]
    test_dates['speed_max'] = speed_local_max.loc[test_dates.index]
    #canidates = (test_dates.anom_max & test_dates.speed_max).index
    test_dates['interp_error'] = np.nan

    for date in test_dates.index:
        date=pd.to_datetime(date)
        x0 = buoy_df.loc[date, 'x']
        y0 = buoy_df.loc[date, 'y']
        x1, y1 = est_middle(date, buoy_df.loc[slice(date-fit_margin, date+fit_margin)], 'x', 'y') 
        test_dates.loc[date, 'interp_error'] = np.sqrt((x0-x1)**2 + (y0-y1)**2)

    test_dates['exceeds_threshold'] = test_dates['interp_error'] > error_thresh
    test_dates['decision'] = (test_dates.anom_max & test_dates.speed_max) & test_dates.exceeds_threshold

    if detailed_return:
        return test_dates

    else:
        flag = pd.Series(data=False, index=buoy_df.index)
        flag.loc[test_dates.loc[test_dates['decision']].index] = True
        return flag