from icedrift.cleaning import check_dates, check_gaps
import pandas as pd
import numpy as np

# make synthetic data
df = pd.DataFrame({'date': pd.date_range('2020-01-01',
                                 '2020-01-01 06:00',
                                 freq='30min'),
                  'latitude': np.linspace(65, 95, 13),
                   'longitude': np.linspace(-100, 181, 13)})

# Add problems to dates
df.loc[3, 'date'] = df.loc[2, 'date']
df.loc[6, 'date'] = df.loc[5, 'date'] + pd.to_timedelta('30sec')
df.loc[9, 'date'] = df.loc[7, 'date'] + pd.to_timedelta('1min')

df['date_flag'] = check_dates(df, date_col='date')
print(df)

def test_date_false_positive():
    assert ~df.loc[2, 'date_flag'], 'Flagged good date'

def test_date_duplicates():
    assert df.loc[3, 'date_flag'], 'Missed exact duplicate time'
    # assert df.loc[6, 'date_flag'], 'Missed duplicate time within tolerance'    

def test_date_reversed():
    assert df.loc[9, 'date_flag'], 'Missed reversed time'    

# add gap
df.loc[8:, 'date'] += pd.to_timedelta('6h')
df.loc[9:, 'date'] += pd.to_timedelta('6h')
df['gap_flag'] = check_gaps(df, date_col='date',
                                  threshold_gap='2H',
                                  threshold_segment=1)
def test_gaps():
    assert df.loc[8, 'gap_flag'], 'Missed solitary data point'



