"""Interpolation tools for buoy analysis. Based on Scipy.interpolate.interp1d and adds calculation of gap size."""
import pandas as pd
from scipy.interpolate import interp1d

# Interpolate to a regular grid
def interpolate_buoy_track(buoy_df, xvar='longitude', yvar='latitude', freq='1H', maxgap_minutes=120):
    """Applies interp1d with cubic splines to the pair of variables specied by
    xvar and yvar. Assumes that the dataframe buoy_df has a datetime index.
    Frequency should be in a form understandable to pandas date_range, e.g. '1H' for hourly.
    """

    t = pd.Series(buoy_df.index)
    dt = pd.to_timedelta(t - t.min()).dt.total_seconds()
    tnew = pd.date_range(start=t.min().round('1H'), end=t.max().round('1H'), freq=freq)
    dtnew = pd.to_timedelta(tnew - tnew.min()).total_seconds()
    
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
    
    df_new['data_gap_minutes'] = np.round(data_gap/60)
    df_new = df_new.where(df_new.data_gap_minutes < maxgap_minutes).dropna()
    return df_new


