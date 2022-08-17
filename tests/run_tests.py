#if __name__ == 'main':
import pandas as pd
import sys
sys.path.append('../../drifter')
import utilities.cleaning
testdf = pd.read_csv('test_duplicates.csv')
testdf['date'] = pd.to_datetime(testdf.date.values)
print(testdf.info())
testdf['test_flag'] = utilities.cleaning.flag_duplicates(testdf)
assert testdf.test_flag.sum() == testdf.flag.sum()