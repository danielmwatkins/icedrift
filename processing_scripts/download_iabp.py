""" Downloading IABP data. Based on guidelines on IABP website.
"""
import urllib
import pandas as pd

dataloc = "/Users/dwatkin2/Documents/research/data/buoy_data/IABP/"
tables = [t for t in os.listdir(dataloc) if 'Table' in t]
tables.sort()

# Slightly modified from example file at IABP website
for table_name in tables:
    year = table_name.split('_')[1].replace('.txt', '')
    
    with open(dataloc + 'ArcticTables/' + table_name, 'r') as f:
        table = f.read().split('\n')
        table = [t for t in table if len(t) > 0]
    nbuoys = len(table)
    
    if not os.path.exists(os.path.join(dataloc, year)):
        os.makedirs(os.path.join(dataloc, year))
    
    for row in table:   
        buoy_id = row.split(';')[0] # some files have a blank spot here, is it the header?
        new_path = os.path.join(dataloc, "webdata", year, buoy_id + '.csv')
        if True: #not os.path.exists(new_path):                                
            try:
                fid=urllib.request.urlopen('https://iabp.apl.uw.edu/WebData/' + buoy_id + '.dat')
                data=fid.read()
                fid.close()
                
                # convert to CSV
                
                df = pd.read_csv(io.BytesIO(data), delimiter=r"\s+")
                df.columns = ' '.join([x for x in df.columns if 'Unnamed' not in x]).split()
                df.to_csv(os.path.join(dataloc, "webdata", year, buoy_id + '.csv'))
            except:
                print(year, buoy_id, 'https://iabp.apl.uw.edu/WebData/' + buoy_id + '.dat', ' not found')