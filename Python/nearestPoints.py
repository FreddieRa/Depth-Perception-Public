#!/usr/bin/env python

import arcpy
import os


#def bufferIntersect(startingPoint, bufferFeatureClass, distance, inputPoints, outputFeatureClass, **kwarg):
#    arcpy.Buffer_analysis(startingPoint, bufferFeatureClass, distance)
#    arcpy.Intersect_analysis((bufferFeatureClass, inputPoints), outputFeatureClass)

def bufferIntersect(x, y, inputPoints, distance, outputFeatureClass, **kwarg):
    bufferFeatureClass = "Buffer"
    point = "Location"
    XYCoordToPoint(x, y, point)
    arcpy.Buffer_analysis(point, bufferFeatureClass, distance)
    arcpy.Intersect_analysis((bufferFeatureClass, inputPoints), outputFeatureClass)


def XYCoordToPoint(x, y, name, path='/Data/Generated/Point.csv'):
    with open(path, "w") as f:
        string = str(x) + "," + str(y)
        f.write("x,y\n"+string)
    arcpy.XYTableToPoint_management(path, name, "x", "y", coordinate_system=arcpy.SpatialReference("British National Grid"))


def main():
    arcpy.AddMessage('CWD: ' + os.getcwd())
    arg = dict([(p.name, p.valueAsText) for p in arcpy.GetParameterInfo()])
    arcpy.AddMessage('Arg: ' + str(arg))
    bufferIntersect(**arg)


main()
