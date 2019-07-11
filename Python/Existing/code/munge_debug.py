import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from convertbng.util import convert_bng
from time import time
import re
from utils_munge import *
import json

# columns = ['FID', 'SHAPE@WKT', 'UpDepth', 'DownDepth', 'DownNode', 'UpNode', 'FlowType', 'Height', 'Width', 'Material', 'ReviewAge', 'ReviewDiam', 'ReviewDept', 'YearLaidBa', 'SewerUsage', 'InfCovLev', 'ID']

# columns_node = ['FID','Shape','ChamberTyp','CoverLevel','InvertDept','NodeRefere','NodeType','SewerUsage', 'InfCovLev', 'SHAPE@WKT']


# data = [
#     (1155,
#     'MULTILINESTRING ZM ((429455.72600000002 542537.36899999995 89.127927591912936 NAN, 429457.67200000002 542550.23800000001 88.896891801470503 NAN))', '1.54', '1.39', 'NZ29424507', 'NZ29424508', 'GRV', '150', ' ', 'VC', '1935', '150', '1.40', ' ', 'C', ' ', 'STE-S188379'),
#     (1152,
#     'MULTILINESTRING ZM ((429455.72600000002 542537.36899999995 89.127927591912936 NAN, 429457.67200000002 542550.23800000001 88.896891801470503 NAN))', '1.54', '1.39', 'NZ29424507', 'NZ29424508', 'GRV', '150', '50', 'VC', '1935', '150', '1.40', '1990', 'C', '14', 'STE-S188379')
#     ]

# data_node = [
#     ('0', '(429318.958, 543720.971)', 'MH',	'78.02', '3.12', 'NZ29433704', 'CH', 'S', '78.82', 'POINT ZM (429318.95799999998 543720.97100000002 79.244317868420438 NAN)')
#     ]

# frame = pd.DataFrame(data, columns=columns)
# nodes = pd.DataFrame(data_node, columns=columns_node)

nodes = pd.read_csv(r'C:\Users\vu86683\Desktop\Zobz\Depth-Perception/nodes.csv', index_col=0)
frame = pd.read_csv(r'C:\Users\vu86683\Desktop\Zobz\Depth-Perception/links.csv', index_col=0)