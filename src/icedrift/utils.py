"""Helper and convenience functions"""

def convert_level1_iabp(df):
    """Renames IABP Level One columns and converts information to datetime. Masks missing data."""
    
    for var in ['BPT', 'BP', 'Ts', 'Ta', 'Th']:
        df[var] = df[var].where(df[var] > -999)

    df = df.rename({'BuoyID': 'buoy_id',
              'Year': 'year',
              'Month': 'month',
              'Day': 'day',
              'Hour': 'hour',
              'Minute': 'minute',
              'Second': 'second',
              'Lat': 'latitude',
              'Lon': 'longitude',
              'Delay(Min)': 'gps_delay_minutes',
              'BPT': 'barometric_pressure_tendency',
              'BP': 'barometric_pressure',
              'Ts': 'surface_temperature',
              'Ta': 'air_temperature',
              'Th': 'hull_temperature',
              'Batt': 'battery_voltage'
             }, axis=1)
    print(df.columns)
    df['datetime'] = pd.to_datetime({'year': df['year'], 'month': df['month'],
                                     'day': df['day'], 'hour': df['hour'],
                                     'minute': df['minute'], 'second': df['second']})
    
    return df
                                         