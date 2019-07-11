import requests
import json

# Magic value
_key = '8wN4p5aXqbVsbK3-O5R9ekbSN-5d2a'
_url = 'https://elevation-api.io/api/elevation?key=%s&resolution=%s&points=(%f,%f)'
# _url = 'https://elevation-api.io/api/elevation?resolution=%s&points=(%f,%f)'
_defaultResolution = '30'

def queryElevationApi (latitude, longitude, resolution=None, verbose=False):

    fmtUrl = _url % (_key, resolution or _defaultResolution, latitude, longitude)
    # fmtUrl = _url % (resolution or _defaultResolution, latitude, longitude)

    if verbose:
        print("Querying", fmtUrl)
    response = requests.get(fmtUrl)

    if verbose:
        print(response)
    return response

def getElevation (latitude, longitude, resolution=None, verbose=False):
    return json.loads(queryElevationApi(latitude,longitude,resolution,verbose).text)['elevations'][0]['elevation']
