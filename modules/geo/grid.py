class Grid():
    """
    Class that will extend apis with grid functionality
    Including:
        Forming named tuple points grid for better accuracy
        Reading geo json files and forming them to points grid
    """
    @staticmethod
    def point(*args):
        """ Returns Point class. Takes in Longitude and Latitude in exact order"""

        from collections import namedtuple

        Point = namedtuple('Point', ['lon', 'lat'])
        if type(args[0]) is list:
            lo = args[0][0]
            la = args[0][1]
        else:
            lo = args[0]
            la = args[1]
        return Point(lon = round(lo,3), lat = round(la,3))

    @staticmethod
    def generate_grid(top, bottom, left, right, resolution = .25):
        """
        Generates grid given borders of parallelepiped
        
        Arguments:
            top {float} -- [description]
            bottom {float} -- [description]
            left {float} -- [description]
            right {float} -- [description]
        
        Keyword Arguments:
            resolution {float} -- [description] (default: {.25})
        Returns:
            'grid': List wit awhere.points
            'area' : Area passed when forming the grid
        """

        import numpy as np

        lat_range = np.arange(Grid.round_coordinates(bottom, resolution), Grid.round_coordinates(top, resolution)+resolution, resolution)
        lon_range = np.arange(Grid.round_coordinates(left, resolution), Grid.round_coordinates(right, resolution)+resolution, resolution)
        
        grid = []

        for la in lat_range:
            for lo in lon_range:
                grid.append(Grid.point(lo, la))

        return grid 

    @staticmethod
    def grid_from_geojson(filename):
        import json
        with open(filename, 'r') as f:
            obj = json.load(f)
        grid = []
        for feature in obj.get('features'):
            grid.append(Grid.point(feature.get('geometry').get("coordinates")))
        return grid

    @staticmethod    
    def round_coordinates(self, value, resolution = 0.25):
        return round((value/resolution)*resolution,3)


    
