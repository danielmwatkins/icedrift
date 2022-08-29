"""Utility functions for flagging nonphysical behavior in drift tracks.
Currently, columns are added. This should be optional.
"""

import pandas as pd
import numpy as np
import pyproj

def flag_duplicates(buoy_df, date_index=False):
    """Returns a boolean Series object with the same index
    as buoy_df with True where multiple reports for a given
    time or coordinate exist. Times are rounded to the nearest minute prior
    to comparison. If date_index=False, expects 'date' to be a column in buoy_df. Latitude and
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

    if date_index:
        date = pd.to_datetime(buoy_df.index.values).round('1min')
    else:
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

def check_dates(buoy_df, date_index=False):
    """Check if there are reversals in the time or if data are isolated in time"""

    threshold = 12*60*60 # threshold is 12 hours
    
    if date_index:
        date = pd.Series(pd.to_datetime(buoy_df.index.values).round('1min'),
                         index=buoy_df.index)
    else:
        date = pd.to_datetime(buoy_df.date).round('1min')

    time_till_next = date.shift(-1) - date
    time_since_last = date - date.shift(1)

    negative_timestep = time_since_last.dt.total_seconds() < 0
    gap_too_large = (time_till_next.dt.total_seconds() > threshold) | (time_since_last.dt.total_seconds() > threshold)
    
    return negative_timestep | gap_too_large

def compute_speed(buoy_df, date_index=False, rotate_uv=False, difference='forward'):
    """Computes buoy velocity and (optional) rotates into north and east directions.
    If x and y are not in the columns, projects lat/lon onto LAEA x/y"""
    
    if date_index:
        date = pd.Series(pd.to_datetime(buoy_df.index.values).round('1min'), index=pd.to_datetime(buoy_df.index))
    else:
        date = pd.to_datetime(buoy_df.date).round('1min')
    
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

        
    buoy_df['u'] = dxdt
    buoy_df['v'] = dydt
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
    return buoy_df

def check_speed(data, window, sigma, date_index=False, method='neighbor'):
    """Checks buoy speed by looking at the minimum of the speed calculated by
    forward differences and by backward differences. For single misplaced points,
    this will identify the point pretty well."""
    
    fwd_speed = compute_speed(data.copy(), date_index=date_index, difference='forward')['speed']   
    bwd_speed = compute_speed(data.copy(), date_index=date_index, difference='backward')['speed']   
    speed = pd.DataFrame({'b': bwd_speed, 'f': fwd_speed}).min(axis=1)
    
    # Neighbor anomaly method
    if method == 'neighbor':
        min_values = 3
        speed_anom = speed - speed.rolling(window, center=True).median()
        speed_stdev = speed_anom.std()
        n = speed.rolling(window, center=True).count()
        flag = np.abs(speed_anom) > sigma*speed_stdev
        flag = flag & (n > min_values)
        return flag

    elif method == 'z-score':
        z = (speed - speed.rolling(window, center=True).mean())/speed.rolling(window, center=True).std()
        return np.abs(z) > sigma
    
def fit_splines(date, data, xvar='x', yvar='y', df=25):
    """Fit regression model using natural cubic splines after
    removing 'date', and evaluate at 'date'.
    Returns dataframe with columns xvar, yvar, xvar_hat, yvar_hat,
    and err = sqrt((x-xhat)^2 + (y-yhat)^2)"""
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
        test_fit = test_point(date, data.loc[data['flag'] != 1], 'x', 'y', df=df, fit_window='48H', sigma=10)
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