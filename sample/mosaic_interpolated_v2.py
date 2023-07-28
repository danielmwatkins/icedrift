"""Code to produce the interpolated drift tracks in the mosaic_interpolated_v2 folder.

Functions defined here:
check_positions = check for invalid lat/lon and duplicated positions
check_dates = check for duplicated or reversed times
check_gaps = check length of segments, single points and short bursts are flagged
check_speed = checks u, v for outliers

"""
import numpy as np
import os
import pandas as pd
import pyproj
from scipy.interpolate import interp1d

#### Parameters
dataloc = '../../data/adc_dn_tracks/'
saveloc = '../data/mosaic_interpolated_v2/'

min_size=100
gap_threshold='6H'
segment_length=24
lon_range=(-180, 180)
lat_range=(65, 90)
max_speed=1.5

speed_window='3D'

#### Function definitions
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

    

def compute_velocity(buoy_df, date_index=True, rotate_uv=False, method='c'):
    """Computes buoy velocity and (optional) rotates into north and east directions.
    If x and y are not in the columns, projects lat/lon onto stereographic x/y prior
    to calculating velocity. Rotate_uv moves the velocity into east/west. Velocity
    calculations are done on the provided time index. Results will not necessarily 
    be reliable if the time index is irregular. With centered differences, values
    near endpoints are calculated as forward or backward differences.
    
    Options for method
    forward (f): forward difference, one time step
    backward (b): backward difference, one time step
    centered (c): 3-point centered difference
    forward_backward (fb): minimum of the forward and backward differences
    """
    buoy_df = buoy_df.copy()
    
    if date_index:
        date = pd.Series(pd.to_datetime(buoy_df.index.values), index=pd.to_datetime(buoy_df.index))
    else:
        date = pd.to_datetime(buoy_df.date)
        
    delta_t_next = date.shift(-1) - date
    delta_t_prior = date - date.shift(1)
    min_dt = pd.DataFrame({'dtp': delta_t_prior, 'dtn': delta_t_next}).min(axis=1)

    # bwd endpoint means the next expected obs is missing: last data before gap
    bwd_endpoint = (delta_t_prior < delta_t_next) & (np.abs(delta_t_prior - delta_t_next) > 2*min_dt)
    fwd_endpoint = (delta_t_prior > delta_t_next) & (np.abs(delta_t_prior - delta_t_next) > 2*min_dt)
    
    if 'x' not in buoy_df.columns:
        projIn = 'epsg:4326' # WGS 84 Ellipsoid
        projOut = 'epsg:3413' # NSIDC North Polar Stereographic
        transformer = pyproj.Transformer.from_crs(projIn, projOut, always_xy=True)

        lon = buoy_df.longitude.values
        lat = buoy_df.latitude.values

        x, y = transformer.transform(lon, lat)
        buoy_df['x'] = x
        buoy_df['y'] = y
    
    if method in ['f', 'forward']:
        dt = (date.shift(-1) - date).dt.total_seconds().values
        dxdt = (buoy_df['x'].shift(-1) - buoy_df['x'])/dt
        dydt = (buoy_df['y'].shift(-1) - buoy_df['y'])/dt

    elif method in ['b', 'backward']:
        dt = (date - date.shift(1)).dt.total_seconds()
        dxdt = (buoy_df['x'] - buoy_df['x'].shift(1))/dt
        dydt = (buoy_df['y'] - buoy_df['y'].shift(1))/dt

    elif method in ['c', 'fb', 'centered', 'forward_backward']:
        fwd_df = compute_velocity(buoy_df.copy(), date_index=date_index, method='forward')
        bwd_df = compute_velocity(buoy_df.copy(), date_index=date_index, method='backward')

        fwd_dxdt, fwd_dydt = fwd_df['u'], fwd_df['v']
        bwd_dxdt, bwd_dydt = bwd_df['u'], bwd_df['v']
        
        if method in ['c', 'centered']:
            dt = (date.shift(-1) - date.shift(1)).dt.total_seconds()
            dxdt = (buoy_df['x'].shift(-1) - buoy_df['x'].shift(1))/dt
            dydt = (buoy_df['y'].shift(-1) - buoy_df['y'].shift(1))/dt
        else:
            dxdt = np.sign(bwd_dxdt)*np.abs(pd.DataFrame({'f': fwd_dxdt, 'b':bwd_dxdt})).min(axis=1)
            dydt = np.sign(bwd_dxdt)*np.abs(pd.DataFrame({'f': fwd_dydt, 'b':bwd_dydt})).min(axis=1)

        dxdt.loc[fwd_endpoint] = fwd_dxdt.loc[fwd_endpoint]
        dxdt.loc[bwd_endpoint] = bwd_dxdt.loc[bwd_endpoint]
        dydt.loc[fwd_endpoint] = fwd_dydt.loc[fwd_endpoint]
        dydt.loc[bwd_endpoint] = bwd_dydt.loc[bwd_endpoint]
    
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
        
    else:
        buoy_df['u'] = dxdt
        buoy_df['v'] = dydt            
        buoy_df['speed'] = np.sqrt(buoy_df['v']**2 + buoy_df['u']**2)    

    return buoy_df

def check_speed(buoy_df, date_index=True, window='3D', sigma=5, max_speed=1.5):
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
    subtracting the median over the same window. 
    
    Next, 
    
    """

    buoy_df = buoy_df.copy()
    if date_index:
        date = pd.Series(pd.to_datetime(buoy_df.index.values).round('1min'), index=pd.to_datetime(buoy_df.index))
    else:
        date = pd.to_datetime(buoy_df.date).round('1min')

    window = pd.to_timedelta(window)
    
    n_min = 0.4*buoy_df.rolling(window, center=True).count()['latitude'].median()

    if n_min > 0:
        n_min = int(n_min)
    else:
        print('n_min is', n_min, ', setting it to 10 and hoping for the best.')
        n_min = 10
        
    def zscore(df, window, n_min):
        uscore = (df['u'] - df['u'].rolling(window, center=True, min_periods=n_min).mean()) / \
                 df['u'].rolling(window, center=True, min_periods=n_min).std()
        vscore = (df['v'] - df['v'].rolling(window, center=True, min_periods=n_min).mean()) / \
                 df['v'].rolling(window, center=True, min_periods=n_min).std()

        zu_anom = uscore - uscore.rolling(window, center=True, min_periods=n_min).median()
        zv_anom = vscore - vscore.rolling(window, center=True, min_periods=n_min).median()
        
        return zu_anom, zv_anom

    # First calculate speed using backward difference and get Z-score
    df = compute_velocity(buoy_df, date_index=True, method='fb')

    zu_init, zv_init = zscore(df, window, n_min)
    zu, zv = zscore(df, window, n_min)

    # Anytime the Z score for U or V velocity is larger than 3, re-calculate Z
    # scores leaving that value out.
    # Probably should replace with a while loop so that it can iterate a few times
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
    df = compute_velocity(buoy_df.loc[~flag], method='fb')
    if np.any(df.speed > max_speed):
        flag = flag | (df.speed > max_speed)

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
    """
    buoy_df_init = buoy_df.copy()
    n = len(buoy_df)
    flag_date = check_dates(buoy_df)
    flag_pos = check_positions(buoy_df, pairs_only=True)
    buoy_df = buoy_df.loc[~(flag_date | flag_pos)].copy()
    if verbose:
        if len(buoy_df) < n:
            print('Initial size', n, 'reduced to', len(buoy_df))

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
        return df.loc[(df.index >= idx[0]) & (df.index <= idx[-1])].copy()
        
    buoy_df = bbox_select(buoy_df)

    if verbose:
        if len(buoy_df) < n:
            print('Initial size', n, 'reduced to', len(buoy_df))
    
    # Return None if there's insufficient data
    if len(buoy_df) < min_size:
        print('Observations in bounding box', n, 'less than min size', min_size)
        return None

    flag_gaps = check_gaps(buoy_df,
                           threshold_gap=gap_threshold,
                           threshold_segment=segment_length)
    buoy_df = buoy_df.loc[~flag_gaps].copy()
    
    
    
    # Check speed
    flag_speed = check_speed(buoy_df, window=speed_window, max_speed=max_speed)
    buoy_df = buoy_df.loc[~flag_speed].copy()

    if len(buoy_df) < min_size:
        return None
    
    else:
        buoy_df_init['flag'] = True
        buoy_df_init.loc[buoy_df.index, 'flag'] = False
        return buoy_df_init

# Interpolate to a regular grid
def interpolate_buoy_track(buoy_df, xvar='longitude', yvar='latitude', 
                           freq='1H', maxgap_minutes=240):
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


    
    
##### Apply the algorithm ######
# Optional step to add later: read the metadata, only attempt 
files = os.listdir(dataloc)
files = [f for f in files if f[0] not in ['.', 'S', 'D']]

for file in files:
    buoy = file.split('_')[-1].replace('.csv', '')
    df = pd.read_csv(dataloc + file, index_col='datetime', parse_dates=True)

    # Adjust to UTC from Beijing time
    if 'V' in buoy:
        df.index = df.index - pd.to_timedelta('8H')

    df_qc = standard_qc(df, min_size=24)

    # Interpolate to hourly
    if df_qc is not None:
        df_interp = interpolate_buoy_track(df_qc.where(~df_qc.flag).dropna(), maxgap_minutes=240)

        # Possible other step:
        # Re-run qc on the interpolated tracks

        # Save interpolated tracks
        df_interp.loc[:, ['longitude', 'latitude']].to_csv(saveloc + file)
    
    