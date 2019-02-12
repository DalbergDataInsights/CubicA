# CubicA

## The Agriculture Advisory App

## Recommenders module

### Connection class

## APIs Library

### Weather APIs

Each API represented as a class. By creating an Instance of a Class you initialize key and secret pair for the instance that can request and receive weather information.

Each source has different availability for the information, so usage of the modules should be combined with general knowledge of
The API data structure of a particular source.

Please look through jupyter_examples for the module of interest.

#### ECMWF CAMS

#### ECMWF CAMS general information

- Highest resolution .25, best accuracy resolution .75
- Instantiating the module and calling *initialize_module* will create a .keyecmwf file in your %username% folder
- The key is usually user-specific, but the module contains key that you can use already (For better stability please create a key urself)
- The data is not available right away, you need to download files from ECMWF servers first, but the module does that for you.
- Since the source code is not containing any data, first requests will be longer than sequential requests for the same period of time.

#### EMCWF CAMS module usage

```python
from api.weather import ecmwf

module = ecmwf.Ecmwf()
# if you dont have a key/token
module.initialize_module()
# if you have a token
module.initialize_modile(token)
# for the single variable/coordinate use
module.get_variable()
# for the grid of coordinates with all available variables use:
module.get_actual_grid(**kwargs) # for actual information/observations
module.get_forecast_grid(**kwargs) # for forecasts
# consult function description for the list of parameters
```

### aWhere

#### aWhere General information

- Highest resolution unspecified, best accuracy resolution .075
- There is a limit of requests per month -> use with caution
- There are some keys/secrets embedded in the module, but you can get your keys via apps.awhere.com
- There are a variety of additional variables available (consult jupyter_notebook example or webpage)
- The data is available right away via GET request
- The module has an internal unique token generation system that will update your token every hour
- Forecasts are available up to 7 days in the future hourly
- There is no foreseeable limit for the observation data at the moment (more than 5 years tested) (Free account only has 30 days of historical data limit)

#### aWhere module usage

##### Preferred way to use the module is by geo-json or csv file

```python
from api.weather import awhere
module = awhere.AWhere(key, secret)
module.get_by_geojson(filename, from_date, to_date,to_file, grid_name, forecast)
```

##### If you need to get weather information for specified grid, consider using get_grid function

```python
module = awhere.AWhere(key, secret)
df = module.get_observations_by_points(point = module.generate_grid(top, bottom, left, right))
```

```python
from api.weather import awhere

module = awhere.AWhere(key, secret) # both parameters are optinal for test. Please create or ask for key/secret if you are using the module in production
# getting actual information
module.current_conditions_by_loc(**kwargs)
# getting observations
module.get_observations_by_points(**kwargs)
# getting forecasts
module.get_forecasts_by_points(**kwargs)
# please consult function descriptions for the list of arguments
```