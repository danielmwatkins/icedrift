"""The N-ICE2015 data have already been quality controlled. This script essentially just makes sure that the data pass the same checks I use for other datasets by applying the cleaning.standard_qc algorithm. I only process the buoys that were used in Itkin et al. 2017 "thin ice and storms" and only saves the tracks that are at least 30 days long. It also reformats the data from JSON to CSV."""

import pandas as pd
import json
import numpy as np
import os
import sys
sys.path.append('../')
from src import cleaning

dataloc = '../../data/NICE2015/'
saveloc = '../../data/processed/n-ice2015/'

files = os.listdir(dataloc)
files = [f for f in files if f.split('.')[-1] == 'json']
nice2015_data = {}
for file in files:
    data = json.load(open(dataloc + file))
    coords = np.array(data['geometry']['coordinates'])
    lon = coords[:, 0]
    lat = coords[:, 1]
    qc = data['properties']['quality']
    time = [pd.to_datetime(x.replace('Z', '')) for x in data['properties']['measured']]
    buoy = data['properties']['buoy']
    nice2015_data[buoy] = pd.DataFrame({'longitude': lon, 'latitude': lat, 'qc': qc}, index=time).sort_index()

not_used = ['SIMBA_2015b', 'SIMBA2015e',
            'SIMBA_2015f', 'WAVE_2015a',
            'STRESS_2015a', 'IMB-B_2015a',
            'IMB-B_2015b', 'IMB-B_2015c',
            'IMB-2_2015a', 'IC_2015a',
            'IMB_2016a'] # This one is from a later expedition

too_short = [b for b in nice2015_data if (b not in not_used) & \
             ((nice2015_data[b].index[-1] - nice2015_data[b].index[0]) < pd.to_timedelta('30D'))]

buoys = [b for b in nice2015_data if b not in not_used + too_short]

for buoy in buoys:
    buoy_init = nice2015_data[buoy].loc[nice2015_data[buoy].qc==1].copy()
    df = cleaning.standard_qc(buoy_init)
    # print(buoy, df.flag.sum()) 
    # Uncomment if you want to see how many get flagged
    # When I last ran it, only 8 observations were flagged, because they were separated from the others by an over-6-hour gap.
    df.loc[~df.flag].to_csv(saveloc + 'n-ice2015_' + buoy + '.csv')