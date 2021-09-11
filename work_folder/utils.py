
import os
import numpy as np

def empty_directory(directory):
    for the_file in os.listdir(directory):
        file_path = os.path.join(directory, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)

        except Exception as e:
            print(e)



def find_replacement(ds1, ds2):

        y2 = ds2['Lat'].to_list()
        x2 = ds2['Lon'].to_list()
        y2 = list(dict.fromkeys(y2))
        x2 = list(dict.fromkeys(x2))
        x1 = ds1['Lon'].to_list()
        x1 = list(dict.fromkeys(x1))
        y1 = ds1['Lat'].to_list()
        y1 = list(dict.fromkeys(y1))
        replacement = {}

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx]

        for i in y2:
            val = find_nearest(y1, i)
            replacement[i] = val
        for i in x2:
            val = find_nearest(x1, i)
            replacement[i] = val
        return replacement