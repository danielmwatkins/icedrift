"""Utility functions for flagging nonphysical behavior in drift tracks.

Functions starting with "check" return a boolean Series with True where the 
data is likely bad.



TBD: Currently, columns are added. This should be optional.
"""

import pandas as pd
import numpy as np
import pyproj
from icedrift.analysis import compute_velocity

def check_positions(data, pairs_only=False,
                   latname='latitude', lonname='longitude'):
    """Looks for duplicated or nonphysical position data. Defaults to masking any 
    data with exact matches in latitude or longitude. Setting pairs_only to false 
    restricts the check to only flag where both longitude and latitude are repeated
    as a pair.
    """

    lats = data[latname].round(10)
    lons = data[lonname].round(10)
    
    invalid_lats = np.abs(lats) > 90
    if np.any(lons < 0):
        invalid_lons = np.abs(lons) > 180
    else:
        invalid_lons = lons > 360
        
    invalid = invalid_lats | invalid_lons
    
    repeated = lats.duplicated(keep='first') | lons.duplicated(keep='first')
    
    duplicated = pd.Series([(x, y) for x, y in zip(lons, lats)],
                                  index=data.index).duplicated(keep='first')
    
    if pairs_only:
        return duplicated | invalid
    
    else:
         return repeated | duplicated | invalid


def check_dates(data, precision='1min', date_col=None):
    """Check if there are reversals in the time or duplicated dates. Optional: check
    whether data are isolated in time based on specified search windows and the threshold
    for the number of buoys within the search windows. Dates are rounded to <precision>,
    so in some cases separate readings that are very close in time will be flagged
    as duplicates. Assumes date_col is in a format readable by pandas to_datetime.
    """

    if date_col is None:
        date_values = data.index.values
        date = pd.Series(pd.to_datetime(date_values).round(precision),
                     index=data.index)
    else:
        date = pd.to_datetime(data[date_col]).round(precision)
    duplicated_times = date.duplicated(keep='first')
    
    time_till_next = date.shift(-1) - date
    time_since_last = date - date.shift(1)

    negative_timestep = time_since_last.dt.total_seconds() < 0

    return negative_timestep | duplicated_times
    

def check_gaps(data, threshold_gap='4h', threshold_segment=12, date_col=None):
    """Segments the data based on a threshold of <threshold_gap>. Segments shorter
    than <threshold_segment> are flagged. If <date_col> not specified, then assumes
    that the data has a time index."""
    
    if date_col is None:
        date_values = data.index.values
        date = pd.Series(pd.to_datetime(date_values),
                     index=data.index)
    else:
        date = pd.to_datetime(data[date_col])
    
    time_till_next = date.shift(-1) - date
    segment = pd.Series(0, index=data.index)
    counter = 0
    tg = pd.to_timedelta(threshold_gap)
    for t in segment.index:
        segment.loc[t] = counter
        if time_till_next[t] > tg:
            counter += 1
    
    # apply_filter
    new = data.groupby(segment).filter(lambda x: len(x) > threshold_segment).index
    flag = pd.Series(True, index=data.index)
    flag.loc[new] = False
    return flag


def check_speed(buoy_df, date_index=True, window='3day', sigma=5, max_speed=1.5, date_col=None):
    """If the position of a point is randomly offset from the path, there will
    be a signature in the velocity. The size of the anomaly will differ depending
    on the time resolution. 
    
    Update to check sequentially, or to update if something is masked.
    
    window can be either time or integer, it is passed to the pandas rolling
    method for calculating anomalies. Default is to use 24 observations for the calculations.
    Data near endpoints are compared to 
    
    method will have more options eventually, for now just z score.
    
    In this method, I first calculate a Z-score for the u and v velocity components, using the 
    forward-backward difference method. This method calculates velocity with forward differences and
    with backward differences, and returns the value with the smallest magnitude. It is therefore
    designed to catch when there is a single out-of-place point. Z-scores are calcuted by first 
    removing the mean over a centered period with the given window size (default 3 days), then
    dividing by the standard deviation over the same period. The Z-scores are then detrended by
    subtracting the median over the same window. When a data point has a Z-score larger than 3, the 
    nearby Z-scores are recalculated with that value masked. Finally, Z-scores larger than 6 are masked.
    """

    buoy_df = buoy_df.copy()
    if date_index:
        date = pd.Series(pd.to_datetime(buoy_df.index.values).round('1min'),
                         index=pd.to_datetime(buoy_df.index))
    else:
        date = pd.to_datetime(buoy_df[date_col]).round('1min')
        init_index = buoy_df.index
        buoy_df = buoy_df.set_index('datetime')
    window = pd.to_timedelta(window)
    n_min = 0.4*buoy_df.rolling(window, center=True).count()['latitude'].median()

    if n_min > 0:
        n_min = int(n_min)
    else:
        print('n_min is', n_min, ', setting it to 10.')
        n_min = 10
        
    def zscore(df, window, n_min):
        uscore = (df['u'] - df['u'].rolling(
                    window, center=True, min_periods=n_min).mean()) / \
                 df['u'].rolling(window, center=True, min_periods=n_min).std()
        vscore = (df['v'] - df['v'].rolling(
                    window, center=True, min_periods=n_min).mean()) / \
                 df['v'].rolling(window, center=True, min_periods=n_min).std()

        zu_anom = uscore - uscore.rolling(window, center=True, min_periods=n_min).median()
        zv_anom = vscore - vscore.rolling(window, center=True, min_periods=n_min).median()
        
        return zu_anom, zv_anom

    # First calculate speed using backward difference and get Z-score
    df = compute_velocity(buoy_df, date_index=True, date_col=date_col, method='fb')

    zu_init, zv_init = zscore(df, window, n_min)
    zu, zv = zscore(df, window, n_min)

    # Anytime the Z score for U or V velocity is larger than 3, re-calculate Z
    # scores leaving that value out.
    # Probably should replace with a while loop so that it can iterate a few times
    # Alternatively, do a forward-looking search and recompute Z after dropping flagged values. 
    for date in df.index:
        if (np.abs(zu[date]) > 3) | (np.abs(zv[date]) > 3):
            # Select part of the data frame that is 2*n_min larger than the window
            idx = df.index[np.abs(df.index - date) < (1.5*window)].drop(date)
            df_new = compute_velocity(df.drop(date).loc[idx,:], method='fb')
            zu_idx, zv_idx = zscore(df_new, window, n_min)

            idx = zu_idx.index[np.abs(zu_idx.index - date) < (0.5*window)]
            zu.loc[idx] = zu_idx.loc[idx]
            zv.loc[idx] = zv_idx.loc[idx]

    flag = df.u.notnull() & ((np.abs(zu) > sigma) | (np.abs(zv) > sigma))
    df = compute_velocity(buoy_df.loc[~flag],
                          date_index=True,
                          date_col=date_col, method='fb')

    buoy_df['speed'] = np.nan
    buoy_df.loc[df.index, 'speed'] = df['speed']

    if not date_index:
        buoy_df = buoy_df.reset_index()
        buoy_df.index = init_index
    flag = pd.Series(flag.values, buoy_df.index)
    if np.any(buoy_df.speed > max_speed):
        flag = flag | (buoy_df.speed > max_speed)

    return flag

#### Define QC algorithm ####
def standard_qc(buoy_df,
                min_size=100,
                gap_threshold='6H',                
                segment_length=24,
                lon_range=(-180, 180),
                lat_range=(65, 90),
                max_speed=1.5,
                speed_window='3D',
                speed_sigma=4,
                verbose=False):
    """QC steps applied to all buoy data. Wrapper for functions in drifter.clean package.
    min_size = minimum number of observations
    gap_threshold = size of gap between observations that triggers segment length check
    segment_length = minimum size of segment to include
    lon_range = tuple with (min, max) longitudes
    lat_range = tuple with (min, max) latitudes
    verbose = if True, print messages to see where data size is reduced
    
    Algorithm
    1. Check for duplicated and reversed dates with check_dates()
    2. Check for duplicated positions with check_positions() with pairs_only set to True.
    3. Check for gaps and too-short segments using check_gaps()
    4. Check for anomalous speeds using check_speed()
    5. Mark all bad entries with a True flag column
    """
    buoy_df_init = buoy_df.reset_index()
    buoy_df_init['flag'] = True

    n = len(buoy_df_init)
    flag_date = check_dates(buoy_df_init, date_col="timestamp")
    flag_pos = check_positions(buoy_df_init, pairs_only=True)
    good_buoy_df = buoy_df_init.loc[~(flag_date | flag_pos)].copy()

    buoy_df_init = buoy_df_init.set_index("timestamp")
    good_buoy_df = good_buoy_df.set_index("timestamp")

    if verbose:
        if len(good_buoy_df) < n:
            print('Initial size', n, 'reduced to', len(good_buoy_df))

    def bbox_select(df):
        """Restricts the dataframe to data within
        the specified lat/lon ranges. Selects data from the earliest
        day that the data is in the range to the last day the data
        is in the range. In between, the buoy is allowed to leave
        the bounding box."""
        lon = df.longitude
        lat = df.latitude
        lon_idx = (lon > lon_range[0]) & (lon < lon_range[1])
        lat_idx = (lat > lat_range[0]) & (lat < lat_range[1])
        idx = df.loc[lon_idx & lat_idx].index
        if len(idx) > 0:
            return df.loc[(df.index >= idx[0]) & (df.index <= idx[-1])].copy()
        
    good_buoy_df = bbox_select(good_buoy_df)

    # Return None for insufficient data

    if good_buoy_df is None or len(good_buoy_df) < min_size:
        if verbose:
            print('Observations in bounding box', n, 'less than min size', min_size)
        return None


    flag_gaps = check_gaps(good_buoy_df,
                           threshold_gap=gap_threshold,
                           threshold_segment=segment_length)
    good_buoy_df = good_buoy_df.loc[~flag_gaps].copy()
    
    if len(good_buoy_df) < min_size:
        if verbose:
            print('Observations post gap-flag', n, 'less than min size', min_size)
        return None
    
    # Check speed
    flag_speed = check_speed(good_buoy_df, window=speed_window, max_speed=max_speed, sigma=speed_sigma)
    good_buoy_df = good_buoy_df.loc[~flag_speed].copy()

    if len(good_buoy_df) < min_size:
        if verbose:
            print('Observations post speed_flag', n, 'less than min size', min_size)
        return None
    
    buoy_df_init.loc[good_buoy_df.index, 'flag'] = False
    buoy_df_init = buoy_df_init.reset_index()

    buoy_df_init.loc[flag_date | flag_pos, 'flag'] = True

    buoy_df_init = buoy_df_init.set_index("timestamp")

    return buoy_df_init

    
     
    
    
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