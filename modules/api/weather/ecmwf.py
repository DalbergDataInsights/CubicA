import json
import os
import sys
from datetime import datetime, timedelta

import netCDF4 as nc
import numpy as np
import pandas as pd
from pip._internal import main
from tqdm import tqdm
from modules.geo.grid import Grid


class Ecmwf():
    """
    Class for working with ECMWF module
    For full functionality you have to request a token and put it in your %username% directory
    You can request and make it manualy or user embedded initialize_module function first. (only once) 
    You have to install netCDF4 library or it will be installed automatically when you call initialize_module function (only once)
    """
    data_path = os.getcwd() + "/data/ecmwf/"
    extracts_path = os.getcwd() + "/data/extracts/"
    
    def initialize_module(self, token=None, **kwargs):
        main(['install', 'netCDF4'])
        token_path = os.path.expanduser('~')
        token = { ## paste your credenttials here if you dont want to initialize it every time
            "url"   : "https://api.ecmwf.int/v1",
            "key"   : "place your key here" if token is None else token,
            "email" : "denys.sementsov@dalberg.com" if kwargs.get('email', None) is None else kwargs.get('email')
                }
        with open(os.path.join(token_path, ".ecmwfapirc"), "w") as f:
            f.writelines(json.dumps(token))
            f.close()

    def get_files(self, date_start, days, forecast = True):
        """
        Simple wrapper for requesting files from ECMWF with default parameters
        ! Do not request more than 40 files in one go (limitation of ECMWF) 
        
        Arguments:
            date_start {datetime} -- Start date for files request
            days {int} -- Amount of days to request from start day
        
        Keyword Arguments:
            forecast {bool} -- Whether to request forecast data or actual observations (default: {True})

        """

        time_frame = [date_start + timedelta(days=x) for x in range(0, days)]
        for date in tqdm(time_frame):
            self.retrieve_data(date, forecast)

    def __cath_index(self, l, in1, in2):
        try:
            return l[in1][in2]
        except IndexError:
            print("INDEX ERROR, RETURNING DEFAULT VALUE")
            return 

    def get_variable(self, lat, lon, date, grid = 0.25, kelvin = True, filename=None, variable= "t2m", **kwargs):
        """
        Function will get you temperature from the date and coordinates specified
        
        Function will also download required files if necessary

        You can influence the ECMWF request by passing additional arguments 
        (see ecmwf.equest_data for additional argument names)

        Arguments:
            lat {`float`} -- Latitude (will be rounded to a closest grid coefficient)
            lon {`float`} -- Longitude (will be rounded to a closest grid coefficient)
            date {`str`} -- Date in YYYmmdd format
        
        Keyword Arguments:
            grid {`float`} -- Grid (roughly 0.01=1km on the surface) (default: {0.25})
            kelvin {`bool`} -- Temperature units (kelvin/Celcius) (default: {True})
        
        Returns:
            `float` -- Temperature for given coordinates and date
        """

        # filename = type_of_dataset+"_"+date+"_time_"+time+"_step_"+step+".nc"
        if not filename:
            filename = self.retrieve_data(date, **kwargs)
        # convert to datetime
        # date = datetime.strptime(date, "%Y%m%d")
        # date = datetime(int(date[:4]), int(date[4:6]), int(date[6:]))
        # open dataset
        try:
            ds = nc.Dataset(filename)
            # datetime into an index
            time = ds.variables['time']
            dates = nc.date2index(dates= date, nctime=time)
            # lon and lat 
            # get index of a grid that contains your lon and lat
            # round coordinates to appropriate resolution
            x = Grid.round_coordinates(lat, grid)
            y = Grid.round_coordinates(lon, grid)
            # get index of the coordinates
            x = self.__cath_index(np.argwhere(ds['latitude'][:] == np.float(x)), 0, 0)
            y = self.__cath_index(np.argwhere(ds['longitude'][:] == np.float(y)), 0, 0)
            # get temperature in kelvin
            t = ds[variable][dates,x, y]
            # TODO return list object
            if not kelvin and variable=="t2m":
                t = t-273.15
        except:
            print("Something went wrong!")
            print(filename)
            t = 0
        return round(t, 4)

    def retrieve_data(self, date_start, date_stop=None, time = "00:00:00/06:00:00/12:00:00/18:00:00", forecast = True, step = "0/3/6/12/18/24/48/72/96",  area = "4.5/29/-2/35", grid = "0.25/0.25", overwrite = False):
        """
        Creates request to ECMWF and gets relevant files if there is none already
        !! Do not request more that 40 files in one cycle. This is a limitation of ECMWF servers.
        
        Arguments:
            date_start {`datetime`} -- (start) Date to get the file for in YYYYmmdd format
        
        Keyword Arguments:
            date_stop {`datetime`} -- end date to get the weather data (default : {date_start})
            time {`str`} -- Time to get the data for (default: {"00:00:00/06:00:00/12:00:00/18:00:00"})
            forecast {`bool`} -- Should the module request forecast or actual data (default: {False})
            step {`str`} -- Step for forecast in hours divided by / (default: {'0/3/6/12/18/24/48/72/96'})
            area {`str`} -- Area to get weather data for in {toplat}/{leftlon}/{bottomlat}/{rightlon} format (default: Uganda {"4.5/29/-2/35"})
            grid {`str`} -- Resolution to make a request for (1/100km) / (default: {'0.25/0.25'})

        Returns:
            `str` -- Path to the file to parse
        """

        # {toplat}/{leftlon}/{bottomlat}/{rightlon}

        type_of_dataset = "fc" if forecast else "an"

        date_start = date_start.strftime("%Y%m%d")

        start = date_start[:4] + "-" + date_start[4:6] + "-" + date_start[6:]
        if date_stop:
            date_stop = date_stop.strftime("%Y%m%d")
            stop = date_stop[:4] + "-" + date_stop[4:6] + "-" + date_stop[6:]
            date_to_get = start + "/to/" + stop
            filename = type_of_dataset + "_" + start + "_to_" + stop +".nc"
        else:
            date_to_get = start + "/to/" + start
            filename = type_of_dataset+"_"+start+".nc"
        
        file = os.path.join(self.data_path, filename)
        # return filename if the file exists
        if os.path.exists(file) and not overwrite:
            return file
        
        # import module only when running the request
        from ecmwfapi import ECMWFDataServer

        server = ECMWFDataServer()

        if forecast == False:
            request = {
            "class": "mc",
            "dataset": "cams_nrealtime",
            "date": date_to_get,
            "expver": "0001",
            "levtype": "sfc",
            'grid'      : grid,
            "param": "164.128/167.128",
            "step": "0",
            "stream": "oper",
            'time'      : time,
            "type": "an",
            "target": file,
            'format'    : "netcdf"
            }

        else:
            time = "00:00:00"
            request = {
            "class": "mc",
            "dataset": "cams_nrealtime",
            "date": date_to_get,
            "expver": "0001",
            "levtype": "sfc",
            "param": "142.128/164.128/167.128",
            "step": "0/3/6/12/24/48",
            "stream": "oper",
            "grid": grid,
            # 'time'      : time,
            "time": time,
            "type": "fc",
            "target": file,
            'format'    : "netcdf"
            }

        if forecast == True and step:
            request['step'] = step

        server.retrieve(request)
        return file

    def get_actual_grid(self, date_start, days, filename = None, grid_step = 0.25, top = None, left = None, bottom = None, right = None, **kwargs):
        """
        Wraper function to get the grid of temperature values
        Gets actual temperature over defined period over defined grid
        
        Arguments:
            date_start {datetime} -- Start of the period of interest
            days {int} -- Length in days of period of interest
        
        Keyword Arguments:
            filename {str} -- Specify filename if you want to use specific nc file (default: {None})
            grid_step {float} -- Grid resolution (default: {0.25})
            top {float} -- Topmost lattitude (default: {None})
            left {float} -- Leftmost longitude (default: {None})
            bottom {float} -- Bottommost lattitude (default: {None})
            right {float} -- Rightmost longitude (default: {None})
        
        Returns: 
            if {filename} is defined writes a [file] and returns a [dataframe]
            if {filename} is not defined returns [dataframe] 

        Tips:
            Pass overwrite = True if you when you request files if you have KeyErrors
        """
        kelvin = kwargs.get('kelvin', False)
        kwargs['forecast'] = False

        data = {    
            "date" : [],
            "lat" : [],
            "lon" : [],
            "0_h_t2m" : [],
            "6_h_t2m" : [],
            "12_h_t2m" : [],
            "18_h_t2m" : [],
            "0_h_tcc" : [],
            "6_h_tcc" : [],
            "12_h_tcc" : [],
            "18_h_tcc" : []
        }
        
        if not top or not bottom or not left or not right:
            return ValueError("Please define borders of the grid")

        lat = np.arange(self.__round_coordinates(bottom, grid_step), self.__round_coordinates(top, grid_step)+grid_step, grid_step)
        lon = np.arange(self.__round_coordinates(left, grid_step), self.__round_coordinates(right, grid_step)+grid_step, grid_step)
        date_range = [date_start + timedelta(days=x) for x in range(0, days)]
        # iterate over dates
        for date in tqdm(date_range):
            for la in tqdm(lat):
                for lo in tqdm(lon):
                    data["date"].append(date.strftime("%Y%m%d"))
                    data["lat"].append(la)
                    data["lon"].append(lo)
                    data["0_h_t2m"].append(self.get_variable(la, lo, date,  
                                                            grid = grid_step,
                                                            kelvin = kelvin, 
                                                            variable = "t2m", **kwargs))
                    data["6_h_t2m"].append(self.get_variable(la, lo, date+timedelta(hours=6), 
                                            grid =  grid_step, 
                                            kelvin = kelvin,
                                            variable = "t2m", **kwargs))
                    data["12_h_t2m"].append(self.get_variable(la, lo, date+timedelta(hours=12),  
                                            grid = grid_step, 
                                            kelvin = kelvin,  
                                            variable = "t2m", **kwargs))
                    data["18_h_t2m"].append(self.get_variable(la, lo, date+timedelta(hours=18),  
                                            grid = grid_step, 
                                            kelvin = kelvin,  
                                            variable = "t2m", **kwargs))
                    data["0_h_tcc"].append(self.get_variable(la, lo, date, 
                                            grid = grid_step, 
                                            variable = "tcc", 
                                            **kwargs))
                    data["6_h_tcc"].append(self.get_variable(la, lo, date+timedelta(hours=6), 
                                            grid =  grid_step, 
                                            variable = "tcc", **kwargs))
                    data["12_h_tcc"].append(self.get_variable(la, lo, date+timedelta(hours=12), 
                                            grid = grid_step, 
                                            variable = "tcc", **kwargs))
                    data["18_h_tcc"].append(self.get_variable(la, lo, date+timedelta(hours=18), 
                                            grid = grid_step, 
                                            variable = "tcc", **kwargs))
        df = pd.DataFrame(data)
        if filename:
            df.to_csv(os.path.join(os.getcwd(), filename), sep=",", encoding="utf-8")
            print("File saved under" + os.path.join(os.getcwd(), filename))
        return df

    def get_forecast_grid(self,date_start, days, filename = None, grid_step = 0.25, top = None, left = None, bottom = None, right = None, **kwargs):
        """
        Gets you information and returns file with default forecasts for requested date period over defined location
        
        Arguments:
            date_start {[datetime]} -- Start of period of interest
            days {[int]} -- Length of the period of interest
        
        Keyword Arguments:
            filename {[str]} -- If defined the dataframe will be saved as csv file in current working directory (default: {None})
            grid_step {float} -- Requested grid resolution(should be alligned with requested files) (default: {0.25})
            top {[type]} -- [description] (default: {None})
            left {[type]} -- [description] (default: {None})
            bottom {[type]} -- [description] (default: {None})
            right {[type]} -- [description] (default: {None})
        
        Returns:
            [type] -- [description]
        
        Tips:
            Pass overwrite = True if you when you request files if you have KeyErrors
        """

        kelvin = kwargs.get('kelvin', False)

        data = {
            "date" : [],
            "lat" : [],
            "lon" : [],
            "0_h_t2m" : [],
            "6_h_t2m" : [],
            "12_h_t2m" : [],
            "24_h_t2m" : [],
            "48_h_t2m" : [],
            "0_h_tcc" : [],
            "6_h_tcc" : [],
            "12_h_tcc" : [],
            "24_h_tcc" : [],
            "48_h_tcc" : [],
            "0_h_lsp" : [],
            "6_h_lsp" : [],
            "12_h_lsp" : [],
            "24_h_lsp" : [],
            "48_h_lsp" : []
        }

        lat = np.arange(self.__round_coordinates(bottom, grid_step), self.__round_coordinates(top, grid_step) + grid_step, grid_step)
        lon = np.arange(self.__round_coordinates(left, grid_step), self.__round_coordinates(right, grid_step) + grid_step, grid_step)
        date_range = [date_start + timedelta(days=x) for x in range(0, days)]
        # iterate over dates
        for date in tqdm(date_range):
            for la in tqdm(lat):
                for lo in tqdm(lon):
                    data["date"].append(date.strftime("%Y%m%d"))
                    data["lat"].append(la)
                    data["lon"].append(lo)
                    data["0_h_t2m"].append(self.get_variable(la, lo, date,  grid= grid_step, kelvin = kelvin,  variable = "t2m", **kwargs))
                    data["6_h_t2m"].append(self.get_variable(la, lo, date+timedelta(hours=6), grid = grid_step, kelvin = kelvin, variable = "t2m", **kwargs))
                    data["12_h_t2m"].append(self.get_variable(la, lo, date+timedelta(hours=12),  grid= grid_step, kelvin = kelvin, variable = "t2m", **kwargs))
                    data["24_h_t2m"].append(self.get_variable(la, lo, date+timedelta(hours=24),  grid= grid_step, kelvin = kelvin, variable = "t2m", **kwargs))
                    data["48_h_t2m"].append(self.get_variable(la, lo, date+timedelta(hours=48),  grid= grid_step, kelvin = kelvin, variable = "t2m", **kwargs))
                    data["0_h_tcc"].append(self.get_variable(la, lo, date, grid= grid_step, kelvin = kelvin,  variable = "tcc", **kwargs))
                    data["6_h_tcc"].append(self.get_variable(la, lo, date+timedelta(hours=6),  grid= grid_step, kelvin = kelvin, variable = "tcc", **kwargs))
                    data["12_h_tcc"].append(self.get_variable(la, lo, date+timedelta(hours=12), grid= grid_step, kelvin = kelvin, variable = "tcc", **kwargs))
                    data["24_h_tcc"].append(self.get_variable(la, lo, date+timedelta(hours=24), grid= grid_step, kelvin = kelvin, variable = "tcc", **kwargs))
                    data["48_h_tcc"].append(self.get_variable(la, lo, date+timedelta(hours=48), grid= grid_step, kelvin = kelvin, variable = "tcc", **kwargs))
                    data["0_h_lsp"].append(self.get_variable(la, lo, date, grid= grid_step, kelvin = kelvin, variable = "lsp", **kwargs))
                    data["6_h_lsp"].append(self.get_variable(la, lo, date+timedelta(hours=6),  grid= grid_step, kelvin = kelvin, variable = "lsp", **kwargs))
                    data["12_h_lsp"].append(self.get_variable(la, lo, date+timedelta(hours=12), grid= grid_step, kelvin = kelvin, variable = "lsp", **kwargs))
                    data["24_h_lsp"].append(self.get_variable(la, lo, date+timedelta(hours=24), grid= grid_step, kelvin = kelvin, variable = "lsp", **kwargs))
                    data["48_h_lsp"].append(self.get_variable(la, lo, date+timedelta(hours=48), grid= grid_step, kelvin = kelvin, variable = "lsp", **kwargs))

        df = pd.DataFrame(data)
        if filename:
            df.to_csv(os.path.join(self.extracts_path, filename), sep=",", encoding="utf-8")
            print("File saved under" + os.path.join(self.extracts_path, filename))
        return df