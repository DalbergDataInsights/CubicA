import base64
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from modules.geo.grid import Grid

from collections import namedtuple

class AWhere():

    ## Properties 
    @property
    def api_header(self):
        return {'Authorization' : 'Bearer ' + self.generate_token()}

    ## Static variables
    awhere_uri = "https://api.awhere.com/oauth/token"
    api_host = "https://api.awhere.com/"

    data_path = os.getcwd() + "/data/awhere/"
    extracts_path = os.getcwd() + "/data/extracts/"
    shapes_path = os.getcwd() + "/data/shapes/"

    ## Constructor
    def __init__(self, key = None, secret = None):
        """
        Initializes module with default parameters and consumer key/secret
        
        Keyword Arguments:
            key {str} -- Key is generated on aWhere website (default: {None})
            secret {str} -- Secret is generated on aWhere website (default: {None})
        """ 
        if not key:
            consumer_key = "place your key here"
        else:
            consumer_key = key
        if not secret:
            consumer_secret = "place your secret here"
        else:
            consumer_secret = secret

        # Encoding key for the module
        to_encode  = consumer_key+ ":" + consumer_secret
        self.key = base64.b64encode(to_encode.encode("utf-8"))
        
        # Recetting token expiration date for new keys
        self.token_expire_time = datetime(1970,1,1,0,0)
        
    ## Functions
    def generate_token(self):
        """
        Will generate new API token if current one is missing/expired
        
        Raises:
            Exception -- When API returns not OK code
        
        Returns:
            API token -- API token is inserted to headings of API request
        """

        if datetime.now() > self.token_expire_time:
            # Fixing header for token request
            request_header  = {'Content-Type': 'application/x-www-form-urlencoded', 
                                'Authorization': b'Basic '+self.key}
            request_body = { 'grant_type':'client_credentials'}
            # Setting new expiration time for token
            self.token_expire_time = datetime.now() + timedelta(hours=1)
            # Requesting new token
            response = requests.post(self.awhere_uri, data=request_body, headers=request_header)
            if response.ok:
                self.token = response.json().get('access_token')
            else:
                raise Exception("Api call returned error with {0} status code".format(response.status_code))

        return self.token

    def current_conditions_by_loc(self, lat=None, lon=None):
        """
        Will return current weather conditions
        
        Keyword Arguments:
            lat {float} -- Latitude (default: {None})
            lon {float} -- Longitude (default: {None})
        
        Raises:
            KeyError -- lat and lon should be specified
            Exception -- returns exception error if response is not returned with code 200/OK
        
        Returns:
            Pandas df -- Dataframe with current conditions for given lat and lon
        """

        if lat is None or lon is None:
            raise KeyError("Please specify Latitude and Longitude")

        # For api request
        request = self.api_host+f"/v2/weather/locations/{str(lat)},{str(lon)}/currentconditions"
        # Get response, validate with header
        response = requests.get(request, headers = self.api_header)
        if response.ok:
            # data structure 
            data = {
                "date": "",
                "time" : "",
                "lat": 0.0,
                "lon": 0.0,
                "conditions_code" : "",
                "cloud_conditions": "",
                "wind_conditions" : "",
                "rain_conditions" : "",
                "conditions_text" : "",
                "temp" : 0,
                "precipitation" : 0,
                "solar" : 0,
                "relative_humidity": 0,
                "wind_amount" : 0,
                "wind_bearind" : 0,
                "wind_direction" : ""
            }

            response_data = response.json()
            date_time = datetime.fromisoformat(response_data.get('dateTime')).replace(tzinfo=timezone(timedelta(0)))
            data['date'] = date_time.date()
            data['time'] = date_time.time()
            data['lat'] = response_data.get("location").get("latitude")
            data['lon'] = response_data.get("location").get("longitude")
            data['conditions_code'] = response_data.get("conditionsCode")
            data['cloud_conditions'] = data['conditions_code'][0]
            data['rain_conditions'] = data['conditions_code'][1]
            data['wind_conditions'] = data['conditions_code'][2]
            data['conditions_text'] = response_data.get("conditionsText")
            data['temp'] = response_data.get("temperature", {}).get("amount")
            data['precipitation'] = response_data.get("precipitation", {}).get("amount")
            data['solar'] = response_data.get("solar", {}).get("amount")
            data['wind_amount'] = response_data.get("wind", {}).get("amount")
            data['wind_bearing'] = response_data.get("wind", {}).get("bearing")
            data['wind_direction'] = response_data.get("wind", {}).get("direction")
            
            return data
        else:
            raise Exception("Api call returned error with {0} status code".format(response.status_code))

    def __get_observations_by_loc(self, lat = None, lon=None, start_date = datetime.now(), end_date = None):
        """
        Gets observations in given timeframe given single set of latitude and longitude
        
        Keyword Arguments:
            lat {float} -- Latitude (default: {None})
            lon {float} -- Longitude (default: {None})
            start_date {datetime} -- Start Date for the observation to return  (default: {datetime.now()})
            end_date {datetime} -- Stop Date for the observations to return (default: {None})
        
        Returns:
            [pandas DataFrame] -- Dataframe with requested information
        """

        # Catching errors

        assert(lat is not None), "Please specify Latitude"
        assert(lon is not None), "Please specify Longitude"
        assert(start_date)
        if lat == None or lon == None:
            raise KeyError("Please specify Latitude and Longitude")

        if start_date > datetime.now():
            raise ValueError("Date specified is after the current date!")

        ## HELPER FUNCTION ##
        def sort_dict(data, response_data):
            for date in response_data:
                data['date'].append(date.get('date'))
                data['lat'].append(date.get("location").get("latitude"))
                data['lon'].append(date.get("location").get("longitude"))
                data['temp_min'].append(date.get("temperatures", {}).get('min'))
                data['temp_max'].append(date.get("temperatures", {}).get('max'))
                data['precipitation'].append(date.get("precipitation", {}).get("amount"))
                data['solar'].append(date.get("solar", {}).get("amount"))

                data['relative_humidity_min'].append(date.get("relativeHumidity", {}).get("min"))
                data['relative_humidity_max'].append(date.get("relativeHumidity", {}).get("max"))

                data['wind_average'].append(date.get("wind", {}).get("average"))
                data['wind_morning_max'].append(date.get("wind", {}).get("morningMax"))
                data['wind_day_max'].append(date.get("wind", {}).get("dayMax"))
        ## END    FUNCTION ##
        
        # defining the time frame
        time_frame = start_date.strftime("%Y-%m-%d") + "," + end_date.strftime("%Y-%m-%d") if end_date else start_date.strftime("%Y-%m-%d")
        # Forming and sending initial request
        request = self.api_host + f"/v2/weather/locations/{lat},{lon}/observations/{time_frame}"
        response = requests.get(request, headers= self.api_header)

        if response.ok:
            response_data = response.json().get('observations', [response.json()])
            data = {
                "date" : [],
                "lat" : [],
                "lon" : [],
                "temp_min" : [],
                "temp_max" : [],
                "precipitation" : [],
                "solar" : [],
                "relative_humidity_min" : [],
                "relative_humidity_max" : [],
                "wind_morning_max" : [],
                "wind_day_max" : [],
                "wind_average" : []
            }
            
            # Process first response
            sort_dict(data, response_data)
            # If there is more data avaliable for this request
            while "next" in response.json().get('_links').keys():
                # Form a next request and process it
                request = self.api_host + response.json().get("_links").get("next", {}).get("href")
                response = requests.get(request, headers = self.api_header)
                response_data = response.json().get('observations', [response.json()])
                sort_dict(data, response_data)
            # Finally form a dataframe
            df = pd.DataFrame(data)
            return df
        else:
            # raise Exception('Request was returned with error ' + str(response.status_code) + f'for point {lon}, {lat}')
            print('Request was returned with error ' + str(response.status_code)  + f'for point {lon}, {lat}')
            return pd.DataFrame()
        
    def get_observations_by_points(self, points, from_date = None, to_date = None, resolution=0.25, to_file=False):
        """
        Small helper function that will help you to get historical data for the list of points
        
        Arguments:
            Points to fetch weather conditions
        
        Keyword Arguments:
            from_date {datetime} -- Starting date for the historical data (default: {None})
            to_date {datetime} -- End date for the historical data (default: {None})
            resolution {float} -- Grid resolution (default: {0.25})
            to_file {bool} -- Extracts to a file if possible (default : {False})
        
        Returns:
            Pandas DataFrame with requested data
        """

        output = None
        
        for point in tqdm(points):
            df  = self.__get_observations_by_loc(lat = point.lat, lon = point.lon, start_date = from_date, end_date= to_date)
            if output is None:
                output = df.copy(deep=True)
                # writing file
                if to_file:
                    if to_date is None:
                        to_date = from_date
                    filename = "observations_" + from_date.strftime("%Y%m%d") + "_to_" + to_date.strftime("%Y%m%d") + "_fetched_" + datetime.now().strftime("%Y%m%d %HH:%MM") + ".csv"
                    file_path = os.path.join(self.data_path, filename)
                    output.to_csv(file_path, index=False,  sep=",", encoding="utf-8")
            else:
                output = output.append(df.copy(deep=True), ignore_index = True)
                if to_file:
                    df.to_csv(file_path, index=False, header= False, mode='a', sep=",", encoding="utf-8")
        return output
    
    def __get_forecasts_by_loc(self, lat, lon, start_date = None, end_date = None):
        """
        Gets forecasts in given timeframe given single set of latitude and longitude
        
        Keyword Arguments:
            lat {float} -- Latitude (default: {None})
            lon {float} -- Longitude (default: {None})
            start_date {datetime} -- Start Date for the observation to return  (default: {datetime.now()})
            end_date {datetime} -- Stop Date for the observations to return (default: {None})
        
        Returns:
            [pandas DataFrame] -- Dataframe with requested information
        """

        def sort_dict(data, response_data):
            for day in response_data:
                for forecast in day.get('forecast', {}):    
                    data['date'].append(day.get("date"))
                    data['time_start'].append(forecast.get("startTime"))
                    data['time_end'].append(forecast.get("endTime"))
                    data['lat'].append(day.get("location").get("latitude"))
                    data['lon'].append(day.get("location").get("longitude"))
                    data['conditions_text'].append(forecast.get("conditionsText"))
                    data['conditions_cloud'].append(forecast.get("conditionsCode")[0])
                    data['conditions_rain'].append(forecast.get("conditionsCode")[1])
                    data['conditions_wind'].append(forecast.get("conditionsCode")[2])
                    data['temp_max'].append(forecast.get("temperatures").get("max"))
                    data['temp_min'].append(forecast.get("temperatures").get("min"))
                    data['precipitation_chance'].append(forecast.get("precipitation").get("chance"))
                    data['precipitation_amount'].append(forecast.get("precipitation").get("amount"))
                    data['cloud_cover'].append(forecast.get("sky").get("cloudCover"))
                    data['sunshine'].append(forecast.get("sky").get("sunshine"))
                    data['solar'].append(forecast.get("solar").get("amount"))
                    data['relative_humidity_min'].append(forecast.get("relativeHumidity").get("min"))
                    data['relative_humidity_avg'].append(forecast.get("relativeHumidity").get("average"))
                    data['relative_humidity_max'].append(forecast.get("relativeHumidity").get("max"))
                    data['wind_min'].append(forecast.get("wind").get("min"))
                    data['wind_avg'].append(forecast.get("wind").get("average"))
                    data['wind_max'].append(forecast.get("wind").get("max"))
                    data['dew_point'].append(forecast.get("dewPoint").get("amount"))
                    soil = forecast.get("soilTemperatures")[0]
                    data['soil_temp_01m_min'].append(soil.get("min"))
                    data['soil_temp_01m_max'].append(soil.get("max"))
                    data['soil_temp_01m_avg'].append(soil.get("average"))
                    soil = forecast.get("soilTemperatures")[1]
                    data['soil_temp_04m_min'].append(soil.get("min"))
                    data['soil_temp_04m_max'].append(soil.get("max"))
                    data['soil_temp_04m_avg'].append(soil.get("average"))
                    soil = forecast.get("soilTemperatures")[2]
                    data['soil_temp_1m_min'].append(soil.get("min"))
                    data['soil_temp_1m_max'].append(soil.get("max"))
                    data['soil_temp_1m_avg'].append(soil.get("average"))
                    soil = forecast.get("soilTemperatures")[3]
                    data['soil_temp_2m_min'].append(soil.get("min"))
                    data['soil_temp_2m_max'].append(soil.get("max"))
                    data['soil_temp_2m_avg'].append(soil.get("average"))
                    soil = forecast.get("soilMoisture")[0]
                    data['soil_moist_01m_min'].append(soil.get("min"))
                    data['soil_moist_01m_max'].append(soil.get("max"))
                    data['soil_moist_01m_avg'].append(soil.get("average"))
                    soil = forecast.get("soilMoisture")[1]
                    data['soil_moist_04m_min'].append(soil.get("min"))
                    data['soil_moist_04m_max'].append(soil.get("max"))
                    data['soil_moist_04m_avg'].append(soil.get("average"))
                    soil = forecast.get("soilMoisture")[2]
                    data['soil_moist_1m_min'].append(soil.get("min"))
                    data['soil_moist_1m_max'].append(soil.get("max"))
                    data['soil_moist_1m_avg'].append(soil.get("average"))
                    soil = forecast.get("soilMoisture")[3]
                    data['soil_moist_2m_min'].append(soil.get("min"))
                    data['soil_moist_2m_max'].append(soil.get("max"))
                    data['soil_moist_2m_avg'].append(soil.get("average"))
        
        if start_date is None:
            response = requests.get(self.api_host + f"/v2/weather/locations/{lat},{lon}/forecasts", headers = self.api_header)
        elif end_date is None:
            response = requests.get(self.api_host+ f"/v2/weather/locations/{lat},{lon}/forecasts/{start_date.strftime('%Y-%m-%d')}", headers = self.api_header)
        else:
            response = requests.get(self.api_host + f"/v2/weather/locations/{lat},{lon}/forecasts/{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}", headers = self.api_header)
        if response.ok:
 
            data = {
                "date": [],
                "time_start": [],
                "time_end" : [],
                "lat": [],
                "lon" : [],
                "conditions_text" : [],
                "conditions_cloud" : [],
                "conditions_rain" : [],
                "conditions_wind": [],
                "temp_max": [],
                "temp_min": [],
                "precipitation_chance": [],
                "precipitation_amount" : [],
                "cloud_cover": [],
                "sunshine": [],
                "solar" : [],
                "relative_humidity_min" : [],
                "relative_humidity_avg" : [],
                "relative_humidity_max" : [],
                "wind_min" : [],
                "wind_avg" : [],
                "wind_max" : [],
                "dew_point" : [],
                "soil_temp_01m_min" : [],
                "soil_temp_01m_max" : [],
                "soil_temp_01m_avg" : [],
                "soil_temp_04m_min" : [],
                "soil_temp_04m_max" : [],
                "soil_temp_04m_avg" : [],
                "soil_temp_1m_min" : [],
                "soil_temp_1m_max" : [],
                "soil_temp_1m_avg" : [],
                "soil_temp_2m_min" : [],
                "soil_temp_2m_max" : [],
                "soil_temp_2m_avg" : [],
                "soil_moist_01m_min" : [],
                "soil_moist_01m_max" : [],
                "soil_moist_01m_avg" : [],
                "soil_moist_04m_min" : [],
                "soil_moist_04m_max" : [],
                "soil_moist_04m_avg" : [],
                "soil_moist_1m_min" : [],
                "soil_moist_1m_max" : [],
                "soil_moist_1m_avg" : [],
                "soil_moist_2m_min" : [],
                "soil_moist_2m_max" : [],
                "soil_moist_2m_avg" : []
            }

            response_data = response.json().get("forecasts", [response.json()])
            sort_dict(data, response_data)

            while "next" in response.json().keys():
                # Form a next request and process it
                request = response.json().get("next", {})
                response = requests.get(request, headers = self.api_header)
                response_data = response.json().get('observations', [response.json()])
                sort_dict(data, response_data)

            df = pd.DataFrame(data)

            return df
        else:
            print('Request was returned with error ' + str(response.status_code)  + f'for point {lon}, {lat}')
            return pd.DataFrame()

    def get_forecasts_by_points(self, points, from_date = None, to_date = None, resolution=0.25, to_file=False):
        """
        Small helper function that will return weather observations for specified list of points
        
        Arguments:
            List of awhere.points to fetch
        
        Keyword Arguments:
            from_date {datetime} -- Start date for the period of interes (default: {None})
            to_date {datetime} -- End date for the period of interest (default: {None})
            resolution {float} -- Grid resolution (in 100 km) (default: {0.25})
            to_file {bool} -- Whether output should be saved to a file (default: {False})
        
        Returns:
            Pandas DataFrame -- with weather observations for requested period
        """

        assert(from_date < datetime.now() + timedelta(days=8)), "Forecasts are available only up to 7 days in the future!"
        if to_date is not None:
            assert(to_date < datetime.now() + timedelta(days=8)), "Forecasts are available only up to 7 days in the future!"

        output = None

        for point in tqdm(points):
            df  = self.__get_forecasts_by_loc(lat = point.lat, lon = point.lon, start_date = from_date, end_date= to_date)
            if output is None:
                output = df.copy(deep=True)
                if to_file:
                    if to_date is None:
                        to_date = from_date
                    filename = "forecast_" + from_date.strftime("%Y%m%d") + "_to_" + to_date.strftime("%Y%m%d")  + "_fetched_" + datetime.now().strftime("%Y%m%d %HH:%MM") +  ".csv"
                    file_path = os.path.join(self.data_path, filename)
                    output.to_csv(file_path, index=False,  sep=",", encoding="utf-8")
            else:
                output = output.append(df.copy(deep=True), ignore_index = True)
                if to_file:
                    df.to_csv(file_path, index=False, mode='a', header=False, sep=",", encoding="utf-8")

        return output

    def get_by_geojson(self, filename, from_date = None, to_date = None, to_file = False, grid_name= "Unknown country", forecast = None):
        """
        Will get you weather information from your geojson
        
        Arguments:
            filename {geJSON} -- gerJSON file path
        
        Keyword Arguments:
            from_date {datetime} -- Start date for weather (default: {None})
            to_date {datetime} -- End date for weather (default: {None})
            to_file {bool} -- Wheather the output should be in file (default: {False})
        """
        
        if from_date is None:
            from_date = datetime.today()
        if not to_date is None:
            assert(to_date - from_date >= timedelta(days=0)), "End date should be after start date"


        grid = Grid.grid_from_geojson(filename)
        
        output = None
        for point in tqdm(grid):
            if forecast is None:
                df  = self.current_conditions_by_loc(lat = point.lat, lon = point.lon)
            if forecast:
                df  = self.__get_forecasts_by_loc(lat = point.lat, lon = point.lon, start_date = from_date, end_date= to_date)
            else:    
                df  = self.__get_observations_by_loc(lat = point.lat, lon = point.lon, start_date = from_date, end_date= to_date)   
            if output is None:
                output = df.copy(deep=True)
                if to_file:
                    if to_date is None:
                        to_date = from_date
                    fname = grid_name +  "_" + from_date.strftime("%Y%m%d") + "_to_" + to_date.strftime("%Y%m%d") +  ".csv"
                    file_path = os.path.join(self.data_path, fname)
                    output.to_csv(file_path, index=False,  sep=",", encoding="utf-8")
            else:
                output = output.append(df.copy(deep=True), ignore_index = True)
                if to_file:
                    df.to_csv(file_path, index=False, mode='a', header=False, sep=",", encoding="utf-8")

        return output

    def get_by_country(self, country, resolution = '075', from_date = None, to_date=None, forecast = None, to_file=None):

        filename = self.shapes_path + country.lower() + "_" + resolution + ".geojson"

        return self.get_by_geojson(filename = filename, from_date = from_date, to_date = to_date, to_file = to_file, forecast = forecast, grid_name=country)
        
    # !TODO
    def get_norms_by_loc(self, lat, lon):
        pass

    def get_norms_by_grid(self):
        pass

    def get_agrovalues_by_loc(self):
        pass

    def get_agrovalues_by_grid(self):
        pass

    def get_agronorms_by_loc(self):
        pass

    def get_agronorms_by_grid(self):
        pass


