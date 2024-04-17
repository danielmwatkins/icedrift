from icedrift.cleaning import check_positions
import pandas as pd
import numpy as np
import os

data = pd.read_csv(os.path.join('test_data', 'IABP_2010_300234010429080.csv'))

def test_check_pairs():
    comp_pairs_only = check_positions(data, pairs_only=True)
    assert np.all(comp_pairs_only == data['default_flag'])
    
def test_check_individual():
    comp_individual = check_positions(data, pairs_only=False)
    assert np.all(comp_individual == data['pairs_false_flag'])