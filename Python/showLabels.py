#!/usr/bin/env python

import arcpy

def showLabels(layer, field):
    lyr = arcpy.mp.ArcGISProject("CURRENT").listMaps()[0].listLayers(layer)[0]

    for lc in lyr.listLabelClasses():
        if(field == "depth"):
            lc.expression = "'Depth: ' + Round($feature.depth, 3)".format(str(field).title(), field)
        else:
            lc.expression = "'{0}: ' + $feature.{1}".format(str(field).title(), field)
    lyr.showLabels=True


def main():
    arg = dict([(p.name, p.valueAsText) for p in arcpy.GetParameterInfo()])
    arcpy.AddMessage('Arg: ' + str(arg))
    showLabels(**arg)


main()
