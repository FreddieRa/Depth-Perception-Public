This is a project created by Freddie Rawlins, Teo Nistor, and Quy Vu for the Northumbrian Water Innovation Festival 2019.
It was adapted from an earlier work done for Northumbrian Water at Infrahack 2019.

Depth Perception is a series of ArcGIS tools built to visualise the data around the sewage network.

It has 4 in built tools packaged together in a toolbox:
    - SplitByDepthsExisting
    - Create Dense Labels
    - Show Labels
    - Nearest Points

They are designed to be used in order, but each tool is a single python file and as such is self-contained.


--------------------------
SplitByDepthsExisting
--------------------------

Paramters:
 - SewerLayerName
    - DESCRIPTION: The name of the layer containing the sewer pipes (not the nodes)
    - TYPE: STRING
    - DEFAULT: "NWGIS.SEWER"
- Depths Suffix
    - DESCRIPTION: What the first new layer containing pipes with data will be called based on input e.g. "NWGIS.SEWER.DEPTHS"
    - TYPE: STRING
    - DEFAULT: ".DEPTHS
- No Depths Suffix
    - DESCRIPTION: What the second new layer containing pipes with no data will be called based on input e.g. "NWGIS.SEWER.DEPTHS"
    - TYPE: STRING
    - DEFAULT: ".DEPTHS

Return:
 - Creates 2 new layers
 - Returns "None"


This is a basic tool that takes a shapefile of sewer pipes (usually called NWGIS.SEWER) loaded in as a layer in ArcGIS
and creates two new layers, one with all of the pipes where we have one or more of UpDepth and DownDepth, and one where
we have neither. 

This is immediately useful for identifying where data is missing, and also patterns in where it is missing, e.g certain locations,
depths, lengths, etc.



--------------------------
CreateDenseLabels
--------------------------

Paramters:
 - SewerLayer
    - DESCRIPTION: The name of the layer containing the sewer pipes (not the nodes)
    - TYPE: STRING
    - DEFAULT: "NWGIS.SEWER"
 - SewerNodesLayer
    - DESCRIPTION: The name of the layer containing the sewer nodes (based on cover locations)
    - TYPE: STRING
    - DEFAULT: "NWGIS.SEWER_NO"
 - ElevationData
    - DESCRIPTION: The path to the elevation (.tif) data for the region
    - TYPE: FILE PATH
 - Result
    - DESCRIPTION: The name of the layer containing the new interpolated depths
    - TYPE: STRING
    - DEFAULT: "NWGIS_CreateDenseLabels"

Return:
 - Creates 1 new layer
 - Returns "None"


This is the fundamental tool of Depth Perception. This takes in the data and from it interpolates a series of estimated depths
every 5 metres along the pipe network. It gives them all of the required attributes so they can be associated with a pipe in 
ArcGIS or other tools, and manipulated at will. The accuracy of the depths is influenced by their distance from known/recorded 
depths, and the accuracy of the elevation (.tif) data provided. Very little can be determined from pipes with missing data,
but where feasible we have given what information we can.



--------------------------
showLabels
--------------------------

Paramters:
 - Layer
    - DESCRIPTION: The layer you want to be labelled
    - TYPE: STRING
 - Field
    - DESCRIPTION: The field you want to show a label of
    - TYPE: STRING
    - DEFAULT: "depth"

Return:
 - Adds a label to every element in a layer
 - Returns "None"


This is used to show the depths on the generated points. It has been separated out since it is a useful function in itself,
and it cannot be run on the generated points until the layer is added to the map, not just stored in memory.



--------------------------
nearestPoints
--------------------------

Paramters:
 - X
    - DESCRIPTION: The X coordinate you want to measure from 
    - TYPE: STRING
 - Y
    - DESCRIPTION: The Y coordinate you want to measure from 
    - TYPE: STRING
 - Input Points
    - DESCRIPTION: The layer containing the generated points 
    - TYPE: STRING
 - Distance to Point
    - DESCRIPTION: The radius of the circle from the coordinates within which you are looking
    - TYPE: FLOAT
    - DEFAULT: 10

Return:
 - Creates a layer containing the points from the dense layer within range
 - Returns "None"


This is an example of the sort of tool that could be run on-site to show only the useful nearby depths, allowing everything else
to be hidden and not clutter a small screen.