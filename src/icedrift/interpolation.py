"""Interpolation tools for buoy analysis. Based on Scipy.interpolate.interp1d and adds calculation of gap size.

TBD: rewrite varnames for consistency. should specify xvar to either be lon/lat or x/y, and if it's x/y don't do the transformation.
"""
import pandas as pd
import pyproj
import numpy as np 
from scipy.interpolate import interp1d

def regrid_buoy_track(buoy_df, precision='5min'):
    """Applies interp1d with cubic splines to align the buoy track to a 5 min grid.
    Assumes that the dataframe buoy_df has a datetime index. Errors are reported by
    computing the difference between the interpolating curve and the original
    data points, then linearly interpolating the error to the new grid. 
    Calculations carried out in north polar stereographic coordinates.
    """

    projIn = 'epsg:4326' # WGS 84 Ellipsoid
    projOut = 'epsg:3413' # NSIDC Polar Stereographic
    transform_to_xy = pyproj.Transformer.from_crs(projIn, projOut, always_xy=True)
    transform_to_ll = pyproj.Transformer.from_crs(projOut, projIn, always_xy=True)
    
    lon = buoy_df.longitude.values
    lat = buoy_df.latitude.values
    
    xvar = 'x_stere'
    yvar = 'y_stere'
    
    x, y = transform_to_xy.transform(lon, lat)
    buoy_df[xvar] = x
    buoy_df[yvar] = y


    t = pd.Series(buoy_df.index)
    t0 = t.min()
    t_seconds = pd.to_timedelta(t - t0).dt.total_seconds()

    tnew = t.round(precision)
     # Drop data points that are closer than <precision> to each other
    tnew = tnew.loc[~tnew.duplicated()]
    
    tnew_seconds = pd.to_timedelta(tnew - t0).dt.total_seconds()

    X = buoy_df[[xvar, yvar]].T.values
    Xnew = interp1d(t_seconds, X, bounds_error=False, kind='cubic')(tnew_seconds)
    idx = ~np.isnan(Xnew.sum(axis=0))
    buoy_df_new = pd.DataFrame(data=np.round(Xnew.T, 5), 
                          columns=[xvar, yvar],
                          index=tnew)
    buoy_df_new.index.names = ['datetime']

    # Next, get the absolute position error
    Xnew_at_old = interp1d(
        tnew_seconds[idx], Xnew[:, idx],
        bounds_error=False, kind='cubic')(t_seconds)
    X_err = pd.Series(
        np.sqrt(np.sum((X - Xnew_at_old)**2, axis=0)), t).ffill().bfill()

    # Finally, assign absolute position error to the new dataframe
    buoy_df_new['sigma_x_regrid'] = interp1d(t_seconds, X_err,
                                             bounds_error=False,
                                             kind='nearest')(tnew_seconds)

    x = buoy_df_new[xvar].values
    y = buoy_df_new[yvar].values

    lon, lat = transform_to_ll.transform(x, y)
    buoy_df_new['longitude'] = lon
    buoy_df_new['latitude'] = lat

    return buoy_df_new


# Interpolate to a regular grid
def interpolate_buoy_track(buoy_df, xvar='longitude', yvar='latitude', 
                           freq='1h', maxgap_minutes=120):
    """Applies interp1d with cubic splines to the pair of variables specied by
    xvar and yvar. Assumes that the dataframe buoy_df has a datetime index.
    Frequency should be in a form understandable to pandas date_range, e.g. '1H' for hourly.
    """

    buoy_df = buoy_df.dropna(subset=[xvar, yvar]).copy()

    # if x/y are longitude/latitude or lat/lon,
    # project to north polar stereographic first.
    if (xvar == 'longitude') | (xvar == 'lon'):
        reproject = True
        projIn = 'epsg:4326' # WGS 84 Ellipsoid
        projOut = 'epsg:3413' # NSIDC Polar Stereographic
        transform_to_xy = pyproj.Transformer.from_crs(
            projIn, projOut, always_xy=True)
        transform_to_ll = pyproj.Transformer.from_crs(
            projOut, projIn, always_xy=True)

        lon = buoy_df.longitude.values
        lat = buoy_df.latitude.values

        xvar = 'x_stere'
        yvar = 'y_stere'

        x, y = transform_to_xy.transform(lon, lat)
        buoy_df[xvar] = x
        buoy_df[yvar] = y
    else:
        reproject = False

    # Force the time to start at 0 minutes after the hour
    t = pd.Series(buoy_df.index)
    dt = pd.to_timedelta(t - t.min()).dt.total_seconds()
    tnew = pd.date_range(start=t.min().round(freq), end=t.max().round(freq), freq=freq).round(freq)
    dtnew = pd.to_timedelta(tnew - t.min()).total_seconds()
    
    X = buoy_df[[xvar, yvar]].T
    time_till_next = t.shift(-1) - t
    time_since_last = t - t.shift(1)

    time_till_next = time_till_next.dt.total_seconds()
    time_since_last = time_since_last.dt.total_seconds()

    Xnew = interp1d(dt, X.values, bounds_error=False, kind='cubic')(dtnew).T

    # add information on initial time resolution 
    data_gap = interp1d(dt, np.sum(np.array([time_till_next.fillna(0),
                                             time_since_last.fillna(0)]), axis=0),
                  kind='previous', bounds_error=False)(dtnew)

    df_new = pd.DataFrame(data=np.round(Xnew, 5), 
                          columns=[xvar, yvar],
                          index=tnew)
    df_new.index.names = ['datetime']
    
    df_new['data_gap_minutes'] = np.round(data_gap/60)/2 # convert from sum to average gap at point
    df_new = df_new.where(df_new.data_gap_minutes < maxgap_minutes).dropna()
    
    if reproject:
        x = df_new[xvar].values
        y = df_new[yvar].values

        lon, lat = transform_to_ll.transform(x, y)
        df_new['longitude'] = np.round(lon, 5)
        df_new['latitude'] = np.round(lat, 5)

    return df_new


def interpolate_buoy_track_to_reference(buoy_df, target_df, xvar='longitude', yvar='latitude'):
    """Applies interp1d with cubic splines to the pair of variables specied by
    xvar and yvar. Assumes that the dataframe buoy_df has a datetime index.
    Interpolates to target_df index where there is overlap.
    """

    t = pd.Series(buoy_df.index)
    dt = pd.to_timedelta(t - t.min()).dt.total_seconds()
    tnew = pd.Series(target_df.index)
    tnew = tnew[(tnew >= t.min()) & (tnew <= t.max())]    
    dtnew = pd.to_timedelta(tnew - t.min()).dt.total_seconds()
    
    X = buoy_df[[xvar, yvar]].T
    Xnew = interp1d(dt, X.values, bounds_error=False, kind='cubic')(dtnew).T

    df_new = pd.DataFrame(data=np.round(Xnew, 5), 
                          columns=[xvar, yvar],
                          index=tnew)
    df_new.index.names = ['datetime']
    return df_new


