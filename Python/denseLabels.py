#!/usr/bin/env python

import arcpy
import pandas as pd
import numpy as np
import re


def getNodesFromPipe(fid, table="NWGIS.SEWER"):
    fields = ["UpNode", "DownNode"]
    query = "FID = " + str(fid)
    return queryTable(table, fields, query)


def getNodeCoords(nodeRef, table="NWGIS.SEWER_NO"):
    fields = ["SHAPE@XY", "CoverLevel", "InvertDept", "InfCovLev"]
    query = "NodeRefere = '{0}'".format(nodeRef)
    return queryTable(table, fields, query)


def getTable(table):
    return queryTable(table, ["*"], single=False)


def convertTable(table):
    fields = ['*', 'SHAPE@WKT']
    cursor = queryTable(table, fields, single=False)
    array = [*cursor]
    return cursor.fields, array


def queryTable(table, fields, query=None, single=True):
    arcpy.SelectLayerByAttribute_management(table, "CLEAR_SELECTION")
    cursor = arcpy.da.SearchCursor(table, fields, query)
    if(single):
        for row in cursor:
            return row
    else:
        return cursor


def multiLS_to_LS(table):
    arcpy.AddMessage('multiLS_to_LS()')
    return arcpy.MultipartToSinglepart_management(table)


def table_to_dataframe(table_name):
    columns, data = convertTable(table_name)
    frame = pd.DataFrame(data, columns=columns)
    return frame


def load_data(pipe_table, node_table):
    return table_to_dataframe(pipe_table), table_to_dataframe(node_table)


def geometrystring_to_list(s, point=False):
    if point:
        pattern = r'.+\((.+)\)'
    else:
        pattern = r'.+\(\((.+)\)\)'

    r = re.compile(pattern)

    # given string s
    ss = r.match(s).group(1).split(',')

    xy = [item.strip().split(' ')[0:2] for item in ss]

    return xy


def clean_coordinates(frame, point=False):
    frame['SHAPE@WKT'] = frame['SHAPE@WKT'].apply(geometrystring_to_list, point=point)

    if point:
        frame[['X', 'Y']] = frame.apply(lambda x: x['SHAPE@WKT'][0], axis=1, result_type='expand')
    else:
        frame = frame[frame['SHAPE@WKT'].apply(lambda x: len(x)) == 2]
        frame[['from', 'to']] = frame.apply(lambda x: x['SHAPE@WKT'], axis=1, result_type='expand')
        frame[['fromX', 'fromY']] = frame.apply(lambda x: x['from'], axis=1, result_type='expand')
        frame[['toX', 'toY']] = frame.apply(lambda x: x['to'], axis=1, result_type='expand')
        frame.drop(['from', 'to', 'SHAPE@WKT'], axis=1, inplace=True)

    return frame


def convert_dtype(frame):
    frame = frame.replace(r'^\s*$', np.nan, regex=True)

    for col in frame:
        try:
            frame[col] = frame[col].astype(float)
        except:
            pass

    return frame


def convert_dtypes(frame, col_list):
    """
    Convert column data types to float
    """
    frame[col_list] = frame[col_list].astype(float)
    return frame


def impute_depth(frame):
    """
    Impute missing depth data
    """

    # Fill depth of up and down stream nodes with review depth data
    # According to Clive -mentor, this is actual measurement so reliable
    for col in ['UpDepth', 'DownDepth']:
        frame[col] = frame[col]\
            .fillna(frame['ReviewDept'])

    # Invert Dept is chamber depth. When there's no information,
    # assume that pipes are plugged in to the middle of the chamber
    # which is worst case scenario, because it makes pipes closer to the
    # ground, and pipes are usually plugged to the bottom of the chamber

    frame['UpDepth'] = frame['UpDepth'].fillna(frame['InvertDept_up'] / 2)
    frame['DownDepth'] = frame['DownDepth'].fillna(frame['InvertDept_down'] / 2)

    # When all things fail, assume that the depth of the up and down
    # node to be the same.

    frame['UpDepth'] = frame['UpDepth'].fillna(frame['DownDepth'])
    frame['DownDepth'] = frame['DownDepth'].fillna(frame['UpDepth'])

    # Drop records with absolutely no information on Depth
    frame = frame.dropna(subset=['UpDepth', 'DownDepth'], how='all')

    # Drop unused columns
    frame = frame.drop(['InvertDept_up', 'InvertDept_down'], axis=1)

    return frame


def impute_height(frame):
    """
    Imputing pipe height, which is measurement from top to bottom
    also means diameter
    """

    # Impute height with diametre
    frame['Height'] = frame['Height'].fillna(frame['ReviewDiam'])

    # Calculate average height by material and pipe usage
    mat = frame\
        .groupby(['Material', 'SewerUsage'], as_index=False)\
        .agg({'Height': np.nanmean})\
        .rename(columns={'Height': 'mat_usage_avg'})

    mat2 = frame\
        .groupby(['Material'], as_index=False)\
        .agg({'Height': np.nanmean})\
        .rename(columns={'Height': 'mat_avg'})

    # First, impute by matching material and sewer usage
    frame = frame\
        .merge(mat, on=['Material', 'SewerUsage'])\
        .merge(mat2, on=['Material'])

    # Then matching by only material
    frame['Height'] = frame['Height']\
        .fillna(frame['mat_usage_avg'])\
        .fillna(frame['mat_avg'])

    # All pipes heights were imputed after this step, so no need for
    # futher imputation

    # Drop unused columns
    frame = frame.drop(['mat_usage_avg', 'mat_avg'], axis=1)

    return frame


def merge_nodes(frame, nodes):
    """
    Merge pipe data with nodes data
    """
    frame = frame\
        .merge(
            nodes,
            how='left',
            left_on='UpNode',
            right_on='NodeRefere')\
        .drop(
            'NodeRefere',
            axis=1)\
        .merge(
            nodes,
            how='left',
            left_on='DownNode',
            right_on='NodeRefere',
            suffixes=('_up', '_down'))\
        .drop(
            'NodeRefere',
            axis=1)

    return frame


def compute_length(frame):
    frame['length_m'] = np.sqrt((frame['fromX'] - frame['toX'])**2 + (frame['fromY'] - frame['toY'])**2)
    return frame


def interpolate(frame, precision_meter=5):
    """
    Interpolate points along the pipe
    """

    # Setup some vars
    selected_cols = [*frame.columns]
    remove_cols = ['fromX', 'fromY', 'toX', 'toY']

    for x in remove_cols:
        selected_cols.remove(x)

    frame_list = []

    for i, row in frame.iterrows():
        # Generate X 1 metre apart
        Xs = np.linspace(
            start=float(row['fromX']),
            stop=float(row['toX']),
            num=int(row['length_m'] / precision_meter)
        )

        # Generate Y 1 metre apart
        Ys = np.linspace(
            start=float(row['fromY']),
            stop=float(row['toY']),
            num=int(row['length_m'] / precision_meter)
        )

        # Convert to dataframe
        x = pd.DataFrame(np.array([Xs, Ys]).T)
        x.index = [i] * len(x)

        # Sore
        frame_list.append(x)

    # Combine all points coordinates
    # At this points, we end up with a dataframe of only Xs and Ys and row id
    points = pd.concat(frame_list)
    points.columns = ['X', 'Y']

    # Join on index - Hacky and lazy solution due to lack of sleep. But works.
    points[selected_cols] = frame[selected_cols].copy()

    # Clean up index
    points = points.reset_index(drop=True)

    return points


def compute_elevation_pipe(frame):
    """
    Compute elevation of up node and down node, taking into account pipe diam
    """
    frame['elevation_up'] = frame['CoverLevel_up'] - frame['UpDepth'] + frame['Height']
    frame['elevation_down'] = frame['CoverLevel_down'] - frame['DownDepth'] + frame['Height']

    return frame


def compute_elevation_point(points):
    """
    Compute elevation and depth of the interpolated points
    as the weighted average of the 2 ends
    """
    # distance from both ends
    points['from_X_up'] = np.abs(points['X'] - points['X_up'])
    points['from_X_down'] = np.abs(points['X_down'] - points['X'])

    # percentage of X distance along the line. Ex: 4 metre from X upstream, 6
    # metre from X downstream, means that 40% distance from X upstream
    points['pct_from_X_up'] = points['from_X_up'] / points[['from_X_up', 'from_X_down']].sum(axis=1)

    # elevation and cover level as weighted average
    points['elevation'] = points['elevation_up'] * (1 - points['pct_from_X_up'])\
        + points['elevation_down'] * (points['pct_from_X_up'])

    points['CoverLevel'] = points['CoverLevel_up'] * (1 - points['pct_from_X_up'])\
        + points['CoverLevel_down'] * (points['pct_from_X_up'])

    # Depth equals to cover level minus elevation
    points['depth'] = points['CoverLevel'] - points['elevation']

    # Drop unnecessary columns
    points = points.drop(['from_X_up', 'from_X_down', 'pct_from_X_up'], axis=1)

    return points


def mainToCSV(frame, nodes):
    arcpy.AddMessage('mainToCSV()')

    # clean coordinates
    frame = clean_coordinates(frame, point=False)
    nodes = clean_coordinates(nodes, point=True)

    # convert dtype
    frame = convert_dtype(frame)
    nodes = convert_dtype(nodes)

    # drop infcovlev
    frame = frame.drop('InfCovLev', axis=1)  # All null

    # Convert back to metre
    frame[['Height', 'Width', 'ReviewDiam']] /= 1000
    # nodes[['InvertDept', 'CoverLevel', 'InfCovLev']] /= 1000

    # Impute cover level
    nodes['CoverLevel'].fillna(nodes['InfCovLev'], inplace=True)
    nodes = nodes[['X', 'Y', 'NodeRefere', 'InvertDept', 'CoverLevel']]

    # coordinate start is always from upnode
    # merge nodes
    frame = merge_nodes(frame, nodes)

    # Impute missing
    # Depth
    frame = impute_depth(frame)

    # Height
    frame = impute_height(frame)

    # Compute elevation
    frame = compute_elevation_pipe(frame)

    # Compute pipe length
    compute_length(frame)

    # interpolate points
    points = interpolate(frame)

    # Compute points' elevations
    points = compute_elevation_point(points)

    # Dump
    dirName = "Data/Generated/"
    if not os.path.exists(dirName):
        os.mkdirs(dirName)
    points.to_csv('Data/Generated/interpolated_points.csv', index=False)

    # with open(result_path + '/query.json', 'w') as f:
    #    json.dump(json.loads(selected), f)


def csvToLayer(path, name):
    arcpy.AddMessage('csvToLayer()')
    return arcpy.XYTableToPoint_management(path, name, 'X', 'Y', coordinate_system=arcpy.SpatialReference('British National Grid'))


def getDepthFromTif(tifPath, table):
    arcpy.AddMessage('getDepthFromTif()')
    arcpy.AddSurfaceInformation_3d(table, tifPath, "Z")
    arcpy.AddField_management(table, "Depth", "DOUBLE")
    fields = ["Z", "elevation", "Depth"]
    with arcpy.da.UpdateCursor(table, fields) as cursor:
        for row in cursor:
            if row[1] is None:
                row[2] = -10
            else:
                row[2] = row[0] - row[1]
            cursor.updateRow(row)


def generate_results(sewerLayer, sewerNodesLayer, elevationData, result, **kwarg):
    exploded = multiLS_to_LS(table=sewerLayer)

    frame, nodes = load_data(
        pipe_table=exploded,
        node_table=sewerNodesLayer
    )

    mainToCSV(frame, nodes)

    # Path ensured from mainToCSV
    lyr = csvToLayer('Data/Generated/interpolated_points.csv', result)
    getDepthFromTif(elevationData, result)


def main():
    arg = dict([(p.name, p.valueAsText) for p in arcpy.GetParameterInfo()])
    arcpy.AddMessage('Arg: ' + str(arg))
    generate_results(**arg)


main()
