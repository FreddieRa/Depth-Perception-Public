# -*- coding: utf-8 -*-

import arcpy
import os
import pandas as pd
import numpy as np
import re


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Depth Perception"
        self.alias = ""

        # List of tool classes associated with this toolbox
        self.tools = [splitByData, generatePoints]


class splitByData(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Split By Data"
        self.description = ""
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        sewerLayer = arcpy.Parameter(
            displayName="Sewer Layer",
            name="sewerLayer",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input")
        sewerLayer.value = "NWGIS.SEWER"

        sewerShape = arcpy.Parameter(
            displayName="Sewer Shapefile",
            name="sewerShape",
            datatype="DEShapefile",
            parameterType="Optional",
            direction="Input")

        depthuffix = arcpy.Parameter(
            displayName="Depth Suffix",
            name="depthuffix",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        depthuffix.value = ".Depth"

        noDepthuffix = arcpy.Parameter(
            displayName="No-Depth Suffix",
            name="noDepthuffix",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        noDepthuffix.value = ".noDepth"

        params = [sewerLayer, sewerShape, depthuffix, noDepthuffix]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    
    def main(self, sewerLayer, sewerShape, depthuffix, noDepthuffix, **kwarg):
        noDepthCol = [220,20,20,100]
        depthCol = [40,200,40,100]

        aprx = arcpy.mp.ArcGISProject("CURRENT"); # 0.
        nwgisMap = aprx.listMaps()[0];

        if not(sewerLayer or sewerShape):
            arcpy.AddError("Error: Either a layer or shape file are required")
            return None

        if(sewerShape):
            sewerLayer = nwgisMap.addDataFromPath(sewerShape)
            sewerLayer = sewerLayer.name

        arcpy.AddMessage('sewerLayer: ' + sewerLayer)
        noDepthLayer = sewerLayer + noDepthuffix
        depthLayer = sewerLayer + depthuffix

        # If layers are already there, delete them first
        self.tryRemoveLayer(nwgisMap, noDepthLayer)
        self.tryRemoveLayer(nwgisMap, depthLayer)

        # Now create them

        arcpy.MakeFeatureLayer_management(sewerLayer, noDepthLayer, "UpDepth = '' And DownDepth = ''"); # 1,2,3,4,5.
        self.addLayerToMap(nwgisMap, noDepthLayer)

        arcpy.MakeFeatureLayer_management(sewerLayer, depthLayer, "UpDepth <> '' Or DownDepth <> ''"); # 6,7,8.
        self.addLayerToMap(nwgisMap, depthLayer)

        # Adjust appearance

        nwgisSewer = nwgisMap.listLayers(sewerLayer)[0];
        nwgisSewernoDepth = nwgisMap.listLayers(noDepthLayer)[0];
        nwgisSewerDepth = nwgisMap.listLayers(depthLayer)[0];

        nwgisSewer.visible = False; # 9.
        self.colourLayer(nwgisSewernoDepth, noDepthCol)
        self.colourLayer(nwgisSewerDepth, depthCol)


    def tryRemoveLayer(self, mapName, layerName):
        try:
            mapName.removeLayer(mapName.listLayers(layerName)[0])
            arcpy.AddMessage('Deleted layer ' + layerName)
        except:
            pass


    def colourLayer(self, layerName, colour):
        sym = layerName.symbology
        sym.renderer.symbol.color = {"RGB": colour};
        layerName.symbology = sym;


    def layerFileName(self, layerName):
        dirName = "Data/Generated/"
        if not os.path.exists(dirName):
            os.mkdirs(dirName)
        return dirName + layerName + '.lyrx'


    def addLayerToMap(self, mapName, layerName):
        arcpy.SaveToLayerFile_management(layerName, self.layerFileName(layerName), True)
        savedFile = arcpy.mp.LayerFile(self.layerFileName(layerName))
        mapName.addLayer(savedFile)
        arcpy.AddMessage('Added layer ' + layerName)


    def suffixify(self, layerName, suffix):
        return self.layerName.replace('.lyrx', suffix + '.lyrx')

    def execute(self, parameters, messages):
        """The source code of the tool."""
        arg = dict([(p.name, p.valueAsText) for p in parameters])
        arcpy.AddMessage('Arg: ' + str(arg))
        self.main(**arg)
        return



class generatePoints(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Generate Points"
        self.description = ""
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        sewerLayer = arcpy.Parameter(
            displayName="Sewer Layer",
            name="sewerLayer",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input")
        sewerLayer.value = "NWGIS.SEWER"

        sewerShape = arcpy.Parameter(
            displayName="Sewer Shapefile",
            name="sewerShape",
            datatype="DEShapefile",
            parameterType="Optional",
            direction="Input")
        
        sewerNodesLayer = arcpy.Parameter(
            displayName="Sewer Nodes Layer",
            name="sewerNodesLayer",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input")
        sewerNodesLayer.value = "NWGIS.SEWER_NO"

        sewerNodesShape = arcpy.Parameter(
            displayName="Sewer Nodes Shapefile",
            name="sewerNodesShape",
            datatype="DEShapefile",
            parameterType="Optional",
            direction="Input")

        elevationData = arcpy.Parameter(
            displayName="Elevation Data (.tif)",
            name="elevationData",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input")

        precision = arcpy.Parameter(
            displayName="Precision",
            name="precision",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input")
        precision.value = 5

        result = arcpy.Parameter(
            displayName="Result",
            name="result",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Output")
        result.value = "NWGIS.GENERATED"

        params = [sewerLayer, sewerShape, sewerNodesLayer, sewerNodesShape, elevationData, precision, result]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    
    def getNodesFromPipe(self, fid, table="NWGIS.SEWER"):
        fields = ["UpNode", "DownNode"]
        query = "FID = " + str(fid)
        return self.queryTable(table, fields, query)


    def getNodeCoords(self, nodeRef, table="NWGIS.SEWER_NO"):
        fields = ["SHAPE@XY", "CoverLevel", "InvertDept", "InfCovLev"]
        query = "NodeRefere = '{0}'".format(nodeRef)
        return self.queryTable(table, fields, query)


    def getTable(self, table):
        return self.queryTable(table, ["*"], single=False)


    def convertTable(self, table):
        fields = ['*', 'SHAPE@WKT']
        cursor = self.queryTable(table, fields, single=False)
        array = [*cursor]
        return cursor.fields, array


    def queryTable(self, table, fields, query=None, single=True):
        arcpy.SelectLayerByAttribute_management(table, "CLEAR_SELECTION")
        cursor = arcpy.da.SearchCursor(table, fields, query)
        if(single):
            for row in cursor:
                return row
        else:
            return cursor


    def multiLS_to_LS(self, table):
        arcpy.AddMessage('multiLS_to_LS()')
        return arcpy.MultipartToSinglepart_management(table)


    def table_to_dataframe(self, table_name):
        columns, data = self.convertTable(table_name)
        frame = pd.DataFrame(data, columns=columns)
        return frame


    def load_data(self, pipe_table, node_table):
        return self.table_to_dataframe(pipe_table), self.table_to_dataframe(node_table)


    def geometrystring_to_list(self, s, point=False):
        if point:
            pattern = r'.+\((.+)\)'
        else:
            pattern = r'.+\(\((.+)\)\)'

        r = re.compile(pattern)

        # given string s
        ss = r.match(s).group(1).split(',')

        xy = [item.strip().split(' ')[0:2] for item in ss]

        return xy


    def clean_coordinates(self, frame, point=False):
        frame['SHAPE@WKT'] = frame['SHAPE@WKT'].apply(self.geometrystring_to_list, point=point)

        if point:
            frame[['X', 'Y']] = frame.apply(lambda x: x['SHAPE@WKT'][0], axis=1, result_type='expand')
        else:
            frame = frame[frame['SHAPE@WKT'].apply(lambda x: len(x)) == 2]
            frame[['from', 'to']] = frame.apply(lambda x: x['SHAPE@WKT'], axis=1, result_type='expand')
            frame[['fromX', 'fromY']] = frame.apply(lambda x: x['from'], axis=1, result_type='expand')
            frame[['toX', 'toY']] = frame.apply(lambda x: x['to'], axis=1, result_type='expand')
            frame.drop(['from', 'to', 'SHAPE@WKT'], axis=1, inplace=True)

        return frame


    def convert_dtype(self, frame):
        frame = frame.replace(r'^\s*$', np.nan, regex=True)

        for col in frame:
            try:
                frame[col] = frame[col].astype(float)
            except:
                pass

        return frame


    def convert_dtypes(self, frame, col_list):
        """
        Convert column data types to float
        """
        frame[col_list] = frame[col_list].astype(float)
        return frame


    def impute_depth(self, frame):
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


    def impute_height(self, frame):
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


    def merge_nodes(self, frame, nodes):
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


    def compute_length(self, frame):
        frame['length_m'] = np.sqrt((frame['fromX'] - frame['toX'])**2 + (frame['fromY'] - frame['toY'])**2)
        return frame


    def interpolate(self, frame, precision):
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
                num=int(row['length_m'] / precision)
            )

            # Generate Y 1 metre apart
            Ys = np.linspace(
                start=float(row['fromY']),
                stop=float(row['toY']),
                num=int(row['length_m'] / precision)
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


    def compute_elevation_pipe(self, frame):
        """
        Compute elevation of up node and down node, taking into account pipe diam
        """
        frame['elevation_up'] = frame['CoverLevel_up'] - frame['UpDepth'] + frame['Height']
        frame['elevation_down'] = frame['CoverLevel_down'] - frame['DownDepth'] + frame['Height']

        return frame


    def compute_elevation_point(self, points):
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


    def mainToCSV(self, frame, nodes, precision):
        arcpy.AddMessage('mainToCSV()')

        # clean coordinates
        frame = self.clean_coordinates(frame, point=False)
        nodes = self.clean_coordinates(nodes, point=True)

        # convert dtype
        frame = self.convert_dtype(frame)
        nodes = self.convert_dtype(nodes)

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
        frame = self.merge_nodes(frame, nodes)

        # Impute missing
        # Depth
        frame = self.impute_depth(frame)

        # Height
        frame = self.impute_height(frame)

        # Compute elevation
        frame = self.compute_elevation_pipe(frame)

        # Compute pipe length
        self.compute_length(frame)

        # interpolate points
        points = self.interpolate(frame, precision)

        # Compute points' elevations
        points = self.compute_elevation_point(points)

        # Dump
        points.to_csv('Data/Generated/interpolated_points.csv', index=False)

        # with open(result_path + '/query.json', 'w') as f:
        #    json.dump(json.loads(selected), f)


    def csvToLayer(self, path, name):
        arcpy.AddMessage('csvToLayer()')
        return arcpy.XYTableToPoint_management(path, name, 'X', 'Y', coordinate_system=arcpy.SpatialReference('British National Grid'))


    def getDepthFromTif(self, tifPath, table):
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


    def main(self, sewerLayer, sewerShape, sewerNodesLayer, sewerNodesShape, elevationData, precision, result, **kwarg):
        if not(sewerLayer or sewerShape):
            arcpy.AddError("Error: Either a sewer layer or shape file are required")
            return None
        
        if not(sewerNodesLayer or sewerNodesShape):
            arcpy.AddError("Error: Either a sewer nodes layer or shape file are required")
            return None

        aprx = arcpy.mp.ArcGISProject("CURRENT"); # 0.
        nwgisMap = aprx.listMaps()[0];

        if (sewerShape):
            sewerLayer = nwgisMap.addDataFromPath(sewerShape)
            sewerLayer = sewerLayer.name
        
        if (sewerNodesShape):
            sewerNodesLayer = nwgisMap.addDataFromPath(sewerNodesShape)
            sewerNodesLayer = sewerNodesLayer.name

        exploded = self.multiLS_to_LS(table=sewerLayer)

        frame, nodes = self.load_data(
            pipe_table=exploded,
            node_table=sewerNodesLayer
        )

        self.mainToCSV(frame, nodes, precision)

        lyr = self.csvToLayer('Data/Generated/interpolated_points.csv', result)
        self.getDepthFromTif(elevationData, result)


    def execute(self, parameters, messages):
        """The source code of the tool."""
        arg = dict([(p.name, p.valueAsText) for p in parameters])
        arcpy.AddMessage('Arg: ' + str(arg))
        self.main(**arg)
        return





class genericTool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Generate Points"
        self.description = ""
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        param = arcpy.Parameter(
            displayName="Param",
            name="param",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")
        param.value = "DEFAULT"


        params = [param]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def main(self, otherParamters):
        """Put the main code in here"""
        return
    
    def execute(self, parameters, messages):
        """Decoding the arguments into a dictionary to be passed into main()"""
        arg = dict([(p.name, p.valueAsText) for p in parameters])
        arcpy.AddMessage('Arg: ' + str(arg))
        self.main(**arg)
        return