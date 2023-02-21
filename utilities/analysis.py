import pandas as pd
import numpy as np
import pyproj

def absolute_dispersion(vel_varname, data, max_length='30D', step_size=3600):
    """Computes the absolute dispersion for buoys in data. Data need
    to be aligned to a common time step. Assumes the start time is time 0,
    and will use data up to time 0 + max_length. Step size in seconds.
    """
    dt = pd.to_timedelta(max_length)
    vel_df = pd.DataFrame({b: data[b][vel_varname].loc[
                    slice(data[b].index[0], data[b].index[0] + dt)]
                           for b in data})
    # calculate integral
    x_df = pd.DataFrame({b: np.cumsum(vel_df[b]*step_size) for b in vel_df.columns})
    x_df = (x_df.T - x_df.iloc[0,:]).T # remove initial position
    N = len(x_df.columns)
    return 1/(N-1)*(x_df**2).sum(axis=1)

def compute_along_across_components(buoy_df, uvar='u', vvar='v', umean='u_mean', vmean='v_mean'):
    """Project the velocity into along-track and across track components."""

    ub = buoy_df[uvar]
    us = buoy_df[umean]
    vb = buoy_df[vvar]
    vs = buoy_df[vmean]

    scale = (ub*us + vb*vs)/(us**2 + vs**2)
    buoy_df[uvar + '_along'] = scale*us
    buoy_df[vvar + '_along'] = scale*vs

    buoy_df[uvar + '_across'] = ub - buoy_df[uvar + '_along']
    buoy_df[vvar + '_across'] = vb - buoy_df[vvar + '_along']
    
    sign = np.sign(buoy_df[umean]*buoy_df[vvar + '_across'] - buoy_df[vmean]*buoy_df[uvar + '_across'])
    # fluctuating component of velocity
    buoy_df['U_fluctuating'] = sign * np.sqrt(
         buoy_df[uvar + '_across']**2 + \
         buoy_df[vvar + '_across']**2)
    
    # along-track component of velocity
    buoy_df['U_along'] = sign * np.sqrt(
         buoy_df['u_along']**2 + \
         buoy_df['v_along']**2)
    
    return buoy_df


    
def compute_speed(buoy_df, date_index=False, rotate_uv=False, difference='forward'):
    """Computes buoy velocity and (optional) rotates into north and east directions.
    If x and y are not in the columns, projects lat/lon onto stere x/y"""
    
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

def strain_rate(buoys, data):
    """Compute the four components of strain rate for each
    date in data. 
    Columns: 'divergence', 'vorticity',
             'pure_shear', 'normal_shear',
             'maximum_shear_strain_rate', 'area', 'shape_flag'

    Additional columns for the uncertainty will be added.
    """
    def check_order(buoys, date, data):
        """Pass through. Right hand rule enforcement tbd"""
        return buoys

    def check_shape(buoys, date, data):
        """Pass through. Return True if the shape is too skewed."""
        return False

    def polygon_area(X, Y):

        s2 = 0.
        N = len(X)
        s1 = X[N-1]*Y[0] - X[0]*Y[N-1]
        for i in range(N - 1):
            s2 += X[i]*Y[i+1] - Y[i]*X[i+1]
        return (s2 + s1)*0.5

    def accel(X, U, A, sign):
        """Computes spatial derivative of velocity for 
        deformation."""
        N = len(X)
        sumvar = 0
        s1 = (U[0] + U[N-1])*(X[0] - X[N-1])
        for i in range(N - 1):
            sumvar += (U[i+1] + U[i])*(X[i+1] - X[i])
        return 1/(2*A) * (sumvar + s1) * sign

    X_data = pd.DataFrame({b: data[b]['x'] for b in buoys})
    Y_data = pd.DataFrame({b: data[b]['y'] for b in buoys})
    U_data = pd.DataFrame({b: data[b]['u'] for b in buoys})
    V_data = pd.DataFrame({b: data[b]['v'] for b in buoys})

    results = []
    for date in X_data.index:
        buoys = check_order(buoys, date, data)
        flag = check_shape(buoys, date, data)
        
        X = X_data.loc[date, :]
        Y = Y_data.loc[date, :]
        U = U_data.loc[date, :]
        V = V_data.loc[date, :]
        
        A = polygon_area(X, Y)
            
        dudx = accel(Y, U, A, 1)
        dudy = accel(X, U, A, -1)
        dvdx = accel(Y, V, A, 1)
        dvdy = accel(X, V, A, -1)
        
        results.append([
            dudx + dvdy, #div
            dvdx - dudy, #vor
            dudy + dvdx, #pure
            dudx - dvdy, #normal
            0.5*np.sqrt((dudx - dvdy)**2 + (dudy + dvdx)**2), #epsilon_ii
            A,
            flag
        ])
            
    return pd.DataFrame(
        np.vstack(results),
        columns=['divergence', 'vorticity', 'pure_shear',
                 'normal_shear', 'maximum_shear_strain_rate',
                 'area', 'shape_flag'],
        index=X_data.index)