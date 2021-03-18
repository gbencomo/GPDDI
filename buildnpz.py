
# The original dataset derived from Cheng et. al (2014) is in a .xlsx format.  The below script is designed to place the data
# we want into a .npz in one shot and save that .npz into our working directory.

import numpy as np
import pandas as pd

# pass in file path
def BuildNPZ(file):

    f = pd.read_excel(file, sheet_name=None)

    keys = list(f.keys())
    data = []

    for i in range(0, len(keys) - 1):
        # We don't want the last sheet

        temp_keys = list(f[keys[i]].keys())
        temp_data = []

        for j in range(0, len(temp_keys)):
            temp_data.append({temp_keys[j] : list(f[keys[i]][temp_keys[j]])})

        data.append({keys[i] : temp_data})

    # Save compressed list in local directory
    np.savez_compressed('data.npz', save_list=data)


#=========================================================
BuildNPZ('amiajnl-2013-002512supp_table1.xlsx')