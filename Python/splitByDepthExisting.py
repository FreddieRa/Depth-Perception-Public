#!/usr/bin/env python

import arcpy
import os


def splitByExisting(sewerLayer, sewerShape, depthSuffix, nodepthSuffix, **kwarg):
    noDepthsCol = [220,20,20,100]
    depthsCol = [40,200,40,100]

    aprx = arcpy.mp.ArcGISProject("CURRENT"); # 0.
    nwgisMap = aprx.listMaps()[0];

    if not(sewerLayer or sewerShape):
        arcpy.AddError("Error: Either a layer or shape file are required")
        return None

    if(sewerShape):
        sewerLayer = nwgisMap.addDataFromPath(sewerShape)
        sewerLayer = sewerLayer.name

    arcpy.AddMessage('sewerLayer: ' + sewerLayer)
    nodepthsLayer = sewerLayer + nodepthSuffix
    depthsLayer = sewerLayer + depthSuffix

    # If layers are already there, delete them first
    tryRemoveLayer(nwgisMap, nodepthsLayer)
    tryRemoveLayer(nwgisMap, depthsLayer)

    # Now create them

    arcpy.MakeFeatureLayer_management(sewerLayer, nodepthsLayer, "UpDepth = '' And DownDepth = ''"); # 1,2,3,4,5.
    addLayerToMap(nwgisMap, nodepthsLayer)

    arcpy.MakeFeatureLayer_management(sewerLayer, depthsLayer, "UpDepth <> '' Or DownDepth <> ''"); # 6,7,8.
    addLayerToMap(nwgisMap, depthsLayer)

    # Adjust appearance

    nwgisSewer = nwgisMap.listLayers(sewerLayer)[0];
    nwgisSewerNODEPTHS = nwgisMap.listLayers(nodepthsLayer)[0];
    nwgisSewerDEPTHS = nwgisMap.listLayers(depthsLayer)[0];

    nwgisSewer.visible = False; # 9.
    colourLayer(nwgisSewerNODEPTHS, noDepthsCol)
    colourLayer(nwgisSewerDEPTHS, depthsCol)


def tryRemoveLayer(mapName, layerName):
    try:
        mapName.removeLayer(mapName.listLayers(layerName)[0])
        arcpy.AddMessage('Deleted layer ' + layerName)
    except:
        pass


def colourLayer(layerName, colour):
    sym = layerName.symbology
    sym.renderer.symbol.color = {"RGB": colour};
    layerName.symbology = sym;


def layerFileName(layerName):
    dirName = "Data/Generated/"
    if not os.path.exists(dirName):
        os.mkdirs(dirName)
    return dirName + layerName + '.lyrx'


def addLayerToMap(mapName, layerName):
    arcpy.SaveToLayerFile_management(layerName, layerFileName(layerName), True)
    savedFile = arcpy.mp.LayerFile(layerFileName(layerName))
    mapName.addLayer(savedFile)
    arcpy.AddMessage('Added layer ' + layerName)


def suffixify(layerName, suffix):
    return layerName.replace('.lyrx', suffix + '.lyrx')


def main():
    arg = dict([(p.name, p.valueAsText) for p in arcpy.GetParameterInfo()])
    arcpy.AddMessage('Arg: ' + str(arg))
    splitByExisting(**arg)


main()
