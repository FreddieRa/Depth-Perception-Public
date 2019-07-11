from utils_munge import munge
from Tool import load_data


exploded = multiLS_to_LS(table='NWGIS.SEWER')

frame, nodes = load_data(
    pipe_table=exploded,
    node_table='NWGIS.SEWER_NO'
)

munge(frame, nodes)
