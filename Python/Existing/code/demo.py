import pandas as pd
import json
from utils_munge import *
#from convertbng.util import convert_lonlat

points = pd.read_csv(r'C:\Users\vu86683\Desktop\simple-viewer-app\hack\data\result\result.csv')

with open(r'C:\Users\vu86683\Desktop\simple-viewer-app\hack\data\result/input.json') as f:
    inputs = json.load(f)
    
result_path = r'C:\Users\vu86683\Desktop\simple-viewer-app\hack\data\result'
    
X = inputs['X']
Y = inputs['Y']
radius=10
n_points=10

query_result = get_nearest_in_area(X, Y, points, radius, n_points, False, True)

print('result exported')
with open(result_path + '/query.json', 'w') as f:
    json.dump(json.loads(query_result), f, indent=4)

#result = convert_lonlat(X, Y)
#X_wgs, Y_wgs = [elem[0] for elem in result]
#
#cover_level = getElevation(Y_wgs, X_wgs, resolution=30)
#print(cover_level)
#
#query_result['depth'] = query_result['CoverLevel_down'] - query_result['elevation']