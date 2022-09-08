"""Interpolation tools for buoy analysis. Based on Scipy.interpolate.interp1d and adds calculation of gap size."""
import pandas as pd
import pyproj
import numpy as np 
from scipy.interpolate import interp1d

def regrid_buoy_track(buoy_df, xvar='x', yvar='y', precision='5min'):
    """Applies interp1d with cubic splines to align the buoy track to a 5 min grid.
    Assumes that the dataframe buoy_df has a datetime index. Errors are reported by
    computing the difference between the interpolating curve and the original data points,
    then linearly interpolating the error to the new grid. 
    Only tested for xvar='x', yvar='y'. If 'x', 'y' not in columns, transforms to LAEA.
    """

    if xvar not in buoy_df.columns:
        if xvar == 'x':
            projIn = 'epsg:4326' # WGS 84 Ellipsoid
            projOut = 'epsg:3571' # Lambert Azimuthal Equal Area centered at north pole, lon0 is 180
            transformer = pyproj.Transformer.from_crs(projIn, projOut, always_xy=True)

            lon = buoy_df.longitude.values
            lat = buoy_df.latitude.values

            x, y = transformer.transform(lon, lat)
            buoy_df[xvar] = x
            buoy_df[yvar] = y


    t = pd.Series(buoy_df.index)
    t0 = t.min()
    t_seconds = pd.to_timedelta(t - t0).dt.total_seconds()


    tnew = t.round(precision)
    tnew = tnew.loc[~tnew.duplicated()] # Drop data points that are closer than 5 minutes to each other
    tnew_seconds = pd.to_timedelta(tnew - t0).dt.total_seconds()

    X = buoy_df[[xvar, yvar]].T.values
    Xnew = interp1d(t_seconds, X, bounds_error=False, kind='cubic')(tnew_seconds)
    idx = ~np.isnan(Xnew.sum(axis=0))
    buoy_df_new = pd.DataFrame(data=np.round(Xnew.T, 5), 
                          columns=[xvar, yvar],
                          index=tnew)
    buoy_df_new.index.names = ['datetime']

    # Next, get the absolute position error
    Xnew_at_old = interp1d(tnew_seconds[idx], Xnew[:, idx], bounds_error=False, kind='cubic')(t_seconds)
    X_err = pd.Series(np.sqrt(np.sum((X - Xnew_at_old)**2, axis=0)), t).ffill().bfill()

    # Finally, assign absolute position error to the new dataframe
    buoy_df_new['sigma_x_regrid'] = interp1d(t_seconds, X_err,
                                             bounds_error=False,
                                             kind='nearest')(tnew_seconds)

    if xvar == 'x':
        projOut = 'epsg:4326' # WGS 84 Ellipsoid
        projIn = 'epsg:3571' # Lambert Azimuthal Equal Area centered at north pole, lon0 is 180
        transformer = pyproj.Transformer.from_crs(projIn, projOut, always_xy=True)

        x = buoy_df_new[xvar].values
        y = buoy_df_new[yvar].values

        lon, lat = transformer.transform(x, y)
        buoy_df_new['longitude'] = lon
        buoy_df_new['latitude'] = lat
        
        
    return buoy_df_new


# Interpolate to a regular grid
def interpolate_buoy_track(buoy_df, xvar='longitude', yvar='latitude', freq='1H', maxgap_minutes=120):
    """Applies interp1d with cubic splines to the pair of variables specied by
    xvar and yvar. Assumes that the dataframe buoy_df has a datetime index.
    Frequency should be in a form understandable to pandas date_range, e.g. '1H' for hourly.
    """

    t = pd.Series(buoy_df.index)
    dt = pd.to_timedelta(t - t.min()).dt.total_seconds()
    tnew = pd.date_range(start=t.min().round('1H'), end=t.max().round('1H'), freq=freq).round('1H')
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
    return df_new


