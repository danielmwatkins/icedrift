"""Tests for the check_dates, check_positions, and check_gaps tools. 

TBD: Test handling of cases with bad lat/lon data

"""

import pandas as pd
import numpy as np
import sys
sys.path.append('../../drifter') # learn right way to do this inside a package
import src.cleaning as clean

df = pd.DataFrame({'date': pd.date_range('2020-01-01',
                                 '2020-01-01 06:00',
                                 freq='30min'),
                  'latitude': np.linspace(65, 95, 13),
                   'longitude': np.linspace(-100, 181, 13)})

# Add problems to dates
df.loc[3, 'date'] = df.loc[2, 'date']
df.loc[6, 'date'] = df.loc[5, 'date'] + pd.to_timedelta('30sec')
df.loc[9, 'date'] = df.loc[7, 'date'] + pd.to_timedelta('1min')

# Add position duplicates
df.loc[6, 'latitude'] = df.loc[7, 'latitude']
df.loc[8, 'longitude'] = df.loc[7, 'longitude']
df.loc[9, 'latitude'] = df.loc[2, 'latitude']
df.loc[9, 'longitude'] = df.loc[2, 'longitude']

df['date_flag'] = clean.check_dates(df, date_col='date')
df['pos_flag'] = clean.check_positions(df, pairs_only=False)

#### Test standard use of check_dates
assert ~df.loc[2, 'date_flag'], 'Flagged good date'
assert df.loc[3, 'date_flag'], 'Missed exact duplicate time'
assert df.loc[6, 'date_flag'], 'Missed duplicate time within tolerance'
assert df.loc[9, 'date_flag'], 'Missed reversed time'
print('Passed date check')

#### Check positions w/ pairs_only = False
assert ~df.loc[2, 'pos_flag'], 'Flagged good position'
assert df.loc[7, 'pos_flag'], 'Missed duplicate latitude'
assert df.loc[8, 'pos_flag'], 'Missed duplicate longitude'

#### Check positions w/ pairs_only = True
df['pos_flag'] = clean.check_positions(df, pairs_only=True)
assert df.loc[9, 'pos_flag'], 'Missed duplicate lat/lon pair'
print('Passed position check')

#### Check gap detection
# Add gaps
df.loc[8:, 'date'] += pd.to_timedelta('6H')
df.loc[9:, 'date'] += pd.to_timedelta('6H')
df['gap_flag'] = clean.check_gaps(df, date_col='date',
                                  threshold_gap='2H',
                                  threshold_segment=1)

assert df.loc[8, 'gap_flag'], 'Missed solitary data point'
print('Passed gap check')