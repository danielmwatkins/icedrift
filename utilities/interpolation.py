"""Interpolation tools for buoy analysis. Based on Scipy.interpolate.interp1d and adds calculation of gap size."""
import pandas as pd
from scipy.interpolate import interp1d

# Interpolate to a regular grid
def interpolate_buoy_track(buoy_df, xvar='longitude', yvar='latitude', freq='1H'):
    """Assumes that the dataframe buoy_df has a datetime index. Frequency should be 
    in a form understandable to pandas date_range, e.g. '1H' for hourly.
    """

    t = pd.Series(df.index)
    tnew = pd.date_range(begin=t.values[0], end=t.values[1], freq=freq)
    
    X = df[[xvar, yvar]].T
    time_till_next = datetime.shift(-1) - datetime
    time_since_last = datetime - datetime.shift(1)

    time_till_next = time_till_next.dt.total_seconds()
    time_since_last = time_since_last.dt.total_seconds()

    Xnew = interp1d(t.total_seconds(), X.values, bounds_error=False, kind='cubic')(tnew.total_seconds()).T

    # add information on initial time resolution 
    data_gap = interp1d(t.total_seconds(), np.sum(np.array([time_till_next.fillna(0),
                                       time_since_last.fillna(0)]), axis=0),
                  kind='previous', bounds_error=False)(tnew.total_seconds())

    df_new = pd.DataFrame(data=np.round(Xnew, 5), 
                          columns=[xvar, yvar],
                          index=tnew.index)
    df_new['time'] = tnew.values
    df_new['data_gap'] = data_gap
    return df_new
