"""Utility functions for flagging nonphysical behavior in drift tracks.
Currently, columns are added. This should be optional.
"""

import pandas as pd
import numpy as np
import pyproj

def check_duplicate_positions(buoy_df, date_index=False, pairs_only=False):
    """Returns a boolean Series object with the same index
    as buoy_df with True where multiple reports for a given
    time or coordinate exist. Times are rounded to the nearest minute prior
    to comparison. If date_index=False, expects 'date' to be a column in buoy_df. Latitude and
    longitude are rounded to the 4th decimal place.
    
    Data are flagged if
    1. If the exact longitude-latitude pair is repeated anywhere
    2. If a latitude point is repeated
    3. If a longitude point is repeated
    
    Number 1 is distinct from 2 and 3 in that it allows repeated patterns to be removed.
    Numbers 2 and 3 are potentially overly restrictive, as it may be the case that a buoy moves 
    due east/west or due north/south. In such cases, using "pairs_only=True" is recommended.
    """

    lats = buoy_df.latitude.round(10)
    lons = buoy_df.longitude.round(10)
    
    # single repeat
    repeated_lats = lats.shift(1) == lats    
    repeated_lons = lons.shift(1) == lons
     
    duplicated_latlon = pd.Series([(x, y) for x, y in zip(lons, lats)],
                                  index=buoy_df.index).duplicated(keep='first')
    
    if pairs_only:
        return duplicated_latlon
    
    else:
         return repeated_lats | repeated_lons | duplicated_latlon


def check_dates(buoy_df, date_index=False, check_gaps=False, gap_window='12H', gap_threshold=4):
    """Check if there are reversals in the time or duplicated dates. Optional: check
    whether data are isolated in time based on specified search windows and the threshold
    for the number of buoys within the search windows. Dates are rounded to the nearest
    minute, so in some cases separate readings that are very close in time will be flagged
    as duplicates."""

    if date_index:
        date = pd.Series(pd.to_datetime(buoy_df.index.values).round('1min'),
                         index=buoy_df.index)
    else:
        date = pd.to_datetime(buoy_df.date).round('1min')
        
    duplicated_times = date.duplicated(keep='first')

    time_till_next = date.shift(-1) - date
    time_since_last = date - date.shift(1)

    negative_timestep = time_since_last.dt.total_seconds() < 0
    
    if check_gaps:
        # Needs to be flexible to handle possible nonmonotonic date index
        gap_too_large = buoy_df.rolling(gap_window, center=True, min_periods=0).latitude.count() < gap_threshold
        return negative_timestep | gap_too_large | duplicated_times

    else:
        return negative_timestep | duplicated_times


def compute_speed(buoy_df, date_index=False, rotate_uv=False, difference='forward'):
    """Computes buoy velocity and (optional) rotates into north and east directions.
    If x and y are not in the columns, projects lat/lon onto LAEA x/y"""
    
    if date_index:
        date = pd.Series(pd.to_datetime(buoy_df.index.values).round('1min'), index=pd.to_datetime(buoy_df.index))
    else:
        date = pd.to_datetime(buoy_df.date).round('1min')
    delta_t_next = date.shift(-1) - date
    delta_t_prior = date - date.shift(1)
    min_dt = pd.DataFrame({'dtp': delta_t_prior, 'dtn': delta_t_next}).min(axis=1)

    # bwd endpoint means the next expected obs is missing: last data before gap
    bwd_endpoint = (delta_t_prior < delta_t_next) & (np.abs(delta_t_prior - delta_t_next) > 2*min_dt)
    fwd_endpoint = (delta_t_prior > delta_t_next) & (np.abs(delta_t_prior - delta_t_next) > 2*min_dt)

#     buoy_df['x'] = fwd_speed['x']
#     buoy_df['y'] = fwd_speed['y']
#     buoy_df['speed'] = pd.DataFrame({'bwd': bwd_speed['speed'], 'fwd': fwd_speed['speed']}).min(axis=1)
#     buoy_df.loc[fwd_endpoint, 'speed'] = fwd_speed['speed'].loc[fwd_endpoint]
#     buoy_df.loc[bwd_endpoint, 'speed'] = bwd_speed['speed'].loc[bwd_endpoint]
    
    if 'x' not in buoy_df.columns:
        projIn = 'epsg:4326' # WGS 84 Ellipsoid
        projOut = 'epsg:3571' # Lambert Azimuthal Equal Area centered at north pole, lon0 is 180
        transformer = pyproj.Transformer.from_crs(projIn, projOut, always_xy=True)

        lon = buoy_df.longitude.values
        lat = buoy_df.latitude.values

        x, y = transformer.transform(lon, lat)
        buoy_df['x'] = x
        buoy_df['y'] = y
    
    if difference == 'forward':
        dt = (date.shift(-1) - date).dt.total_seconds().values
        dxdt = (buoy_df['x'].shift(-1) - buoy_df['x'])/dt
        dydt = (buoy_df['y'].shift(-1) - buoy_df['y'])/dt

    elif difference == 'backward':
        dt = (date - date.shift(1)).dt.total_seconds()
        dxdt = (buoy_df['x'] - buoy_df['x'].shift(1))/dt
        dydt = (buoy_df['y'] - buoy_df['y'].shift(1))/dt

    elif difference == 'centered':
        dt = (date.shift(-1) - date.shift(1)).dt.total_seconds()
        dxdt = (buoy_df['x'].shift(-1) - buoy_df['x'].shift(1))/dt
        dydt = (buoy_df['y'].shift(-1) - buoy_df['y'].shift(1))/dt

        dt = (date.shift(-1) - date).dt.total_seconds().values
        fwd_dxdt = (buoy_df['x'].shift(-1) - buoy_df['x'])/dt
        fwd_dydt = (buoy_df['y'].shift(-1) - buoy_df['y'])/dt

        dt = (date - date.shift(1)).dt.total_seconds()
        bwd_dxdt = (buoy_df['x'] - buoy_df['x'].shift(1))/dt
        bwd_dydt = (buoy_df['y'] - buoy_df['y'].shift(1))/dt


        
    buoy_df['u'] = dxdt
    buoy_df['v'] = dydt
    
    if difference == 'centered':
        """Compute values at endpoints with fwd or bwd differences"""
        buoy_df.loc[fwd_endpoint, 'u'] = fwd_dxdt.loc[fwd_endpoint]
        buoy_df.loc[bwd_endpoint, 'u'] = bwd_dxdt.loc[bwd_endpoint]
        buoy_df.loc[fwd_endpoint, 'v'] = fwd_dydt.loc[fwd_endpoint]
        buoy_df.loc[bwd_endpoint, 'v'] = bwd_dydt.loc[bwd_endpoint]
        
        dxdt = buoy_df['u']
        dydt = buoy_df['v']
        
    buoy_df['speed'] = np.sqrt(buoy_df['v']**2 + buoy_df['u']**2)
    buoy_df['speed_flag'] = buoy_df['speed'] > 1.5 # will flag open ocean speeds, so use with care
    
    
    if rotate_uv:
        # Unit vectors
        buoy_df['Nx'] = 1/np.sqrt(buoy_df['x']**2 + buoy_df['y']**2) * -buoy_df['x']
        buoy_df['Ny'] = 1/np.sqrt(buoy_df['x']**2 + buoy_df['y']**2) * -buoy_df['y']
        buoy_df['Ex'] = 1/np.sqrt(buoy_df['x']**2 + buoy_df['y']**2) * -buoy_df['y']
        buoy_df['Ey'] = 1/np.sqrt(buoy_df['x']**2 + buoy_df['y']**2) * buoy_df['x']

        buoy_df['u'] = buoy_df['Ex'] * dxdt + buoy_df['Ey'] * dydt
        buoy_df['v'] = buoy_df['Nx'] * dxdt + buoy_df['Ny'] * dydt

        # Calculate angle, then change to 360
        heading = np.degrees(np.angle(buoy_df.u.values + 1j*buoy_df.v.values))
        heading = (heading + 360) % 360
        
        # Shift to direction from north instead of direction from east
        heading = 90 - heading
        heading = (heading + 360) % 360
        buoy_df['bearing'] = heading
        buoy_df['speed'] = np.sqrt(buoy_df['u']**2 + buoy_df['v']**2)
        buoy_df.drop(['Nx', 'Ny', 'Ex', 'Ey'], axis=1, inplace=True)
        
    # check if next to gaps
    
    return buoy_df

def check_speed(buoy_df, window, sigma, date_index=False, method='neighbor'):
    """Checks buoy speed by looking at the minimum of the speed calculated by
    forward differences and by backward differences. For single misplaced points,
    this will identify the point pretty well."""

    if date_index:
        date = pd.Series(pd.to_datetime(buoy_df.index.values).round('1min'), index=pd.to_datetime(buoy_df.index))
    else:
        date = pd.to_datetime(buoy_df.date).round('1min')

    
    fwd_speed = compute_speed(buoy_df.copy(), date_index=date_index, difference='forward')['speed']   
    bwd_speed = compute_speed(buoy_df.copy(), date_index=date_index, difference='backward')['speed']   
    speed = pd.DataFrame({'b': bwd_speed, 'f': fwd_speed}).min(axis=1)
    
    # Neighbor anomaly method
    if method == 'neighbor':
        min_values = 3
        speed_anom = speed - speed.rolling(window, center=True).median()
        speed_stdev = speed_anom.std()
        n = speed.rolling(window, center=True, min_periods=0).count()
        flag = np.abs(speed_anom) > sigma*speed_stdev
        flag = flag & (n > min_values)
        return flag

    elif method == 'z-score':
        z = (speed - speed.rolling(window, center=True, min_periods=2).mean()
            )/speed.rolling(window,
                            center=True,
                            min_periods=2).std()
        flag = np.abs(z) > sigma

    dt_next = date.shift(-1) - date
    dt_prior = date - date.shift(1)
    gap_threshold = pd.to_timedelta('4H')
    
    not_by_gap = (dt_next < gap_threshold) & (dt_prior < gap_threshold)
    
    return flag & not_by_gap

    
def fit_splines(date, data, xvar='x', yvar='y', df=25):
    """Fit regression model using natural cubic splines after
    removing 'date', and evaluate at 'date'.
    Returns dataframe with columns xvar, yvar, xvar_hat, yvar_hat,
    and err = sqrt((x-xhat)^2 + (y-yhat)^2)"""
    from sklearn.linear_model import LinearRegression
    from patsy import cr
    
    data_fit = data.drop(date)
    tfit = data_fit.index
    xfit = (tfit - tfit[0]).total_seconds()
    yfit = data_fit[[xvar, yvar]]
    x_basis = cr(xfit, df=df, constraints="center")
    model = LinearRegression().fit(x_basis, yfit)

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

def calc_speed_for_outlier_check(buoy_df, date_index=True):
    """Computes a measure of speed adapted for flagging bad data.
    Since most of the buoy data has some gaps, the algorithm needs 
    to calculate speed different at the start and end of regular observations
    versus in the middle of regular observations. It does this by looking
    at delta_t_prior and delta_t_post, the times since last and till
    next. If both are approximately the same size, then the returned 
    speed is the minimum of the forward and the backward difference estimates
    of velocity. If delta_t_prior << delta_t_post, only the backward difference is
    used. Otherwise the forward difference is used."""

    date_index=True
    buoy_df = buoy_df.dropna(subset=['latitude', 'longitude']).copy()
  
    if date_index:
        date = pd.Series(pd.to_datetime(buoy_df.index.values).round('1min'),
                         index=buoy_df.index)
    else:
        date = pd.to_datetime(buoy_df.date).round('1min')

    delta_t_next = date.shift(-1) - date
    delta_t_prior = date - date.shift(1)

    fwd_speed = compute_speed(buoy_df.copy(), date_index=True, difference='forward')
    bwd_speed = compute_speed(buoy_df.copy(), date_index=True, difference='backward')

    min_dt = pd.DataFrame({'dtp': delta_t_prior, 'dtn': delta_t_next}).min(axis=1)

    # bwd endpoint means the next expected obs is missing: last data before gap
    bwd_endpoint = (delta_t_prior < delta_t_next) & (np.abs(delta_t_prior - delta_t_next) > 2*min_dt)
    fwd_endpoint = (delta_t_prior > delta_t_next) & (np.abs(delta_t_prior - delta_t_next) > 2*min_dt)

    buoy_df['x'] = fwd_speed['x']
    buoy_df['y'] = fwd_speed['y']
    buoy_df['speed'] = pd.DataFrame({'bwd': bwd_speed['speed'], 'fwd': fwd_speed['speed']}).min(axis=1)
    buoy_df.loc[fwd_endpoint, 'speed'] = fwd_speed['speed'].loc[fwd_endpoint]
    buoy_df.loc[bwd_endpoint, 'speed'] = bwd_speed['speed'].loc[bwd_endpoint]
    
    return buoy_df

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