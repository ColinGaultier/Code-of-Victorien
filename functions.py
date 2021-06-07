"""Functions.py: functions used for fitting/modelling of solar-energy prediction."""

import pandas as pd
import numpy as np
import pytz as tz

from astrotool.date_time import fix_date_time
    


def read_knmi_weather_data(fileName):
    """Read a KNMI data file with hourly measurements, and return them with some adaptations to mimick WeerPlaza data."""
    
    # Read the data file:
    # Select columns to read: date, time, cloud cover, rain, temperature, pressure, wind direction and speed, relative humidity
    selCols = ['YYYYMMDD','HH', 'N', 'RH', 'T', 'P', 'DD','FH', 'U']
    Weather = pd.read_csv(fileName, skiprows=29, index_col=False, header=1, usecols=selCols, sep=r'\s*,\s*',
                          engine='python')
    
    # We need to fix several things below:
    #   1) hours are 1-24, convert to 0-23.  Note that this means e.g 2021-12-31 24h -> 2020-01-01 0h!
    #   2) UTC -> CET
    #   3) create separate year, month, day columns
    #   4) remove obsolete columns
    #   5) rename some columns to give them the same name as in the WP data
    #   6) convert some units to ~SI
    
    
    # Add a UTC datetime column to the dataframe:
    Weather = KNMIdatehour2datetime(Weather)
    
    # Convert datetime object to CET:
    cet = tz.timezone('Europe/Amsterdam')
    Weather.datetime = Weather.datetime.dt.tz_convert(cet)         # Works when already converted to UT by to_datetime and if datetime is not an index!
    
    
    # Create separate year, month, day, hour columns:
    years  = Weather.loc[:, 'datetime'].dt.year
    months = Weather.loc[:, 'datetime'].dt.month
    days   = Weather.loc[:, 'datetime'].dt.day
    hours  = Weather.loc[:, 'datetime'].dt.hour
    
    Weather.insert(1, 'year', years)
    Weather.insert(2, 'mo',   months)
    Weather.insert(3, 'dy',   days)
    Weather.insert(4, 'hr',   hours)
    
    
    # Remove unnecessary columns:
    del Weather['YYYYMMDD']  # Remove the 'YYYYMMDD' column
    del Weather['HH']        # Remove the old 'HH' column; have 'hr' now
    del Weather['datetime']  # No longer need the datetime column here
    
    
    # Rename some columns to line up with the WP data and use ~SI units:
    # Rename the clouds column, keep only lines where clouds (N) is a number (not NaN), and convert to percentage:
    Weather.rename(columns={"N":"clouds"}, inplace=True)  # Rename clouds column
    Weather = Weather[Weather.clouds.notna()]             # Keep only lines where 'clouds' is a number (not NaN) - https://stackoverflow.com/a/13413845/1386750)
    Weather.loc[:, 'clouds'] *= (100/8)                   # Cloud cover: octets (0-8) -> percentage (0-100)
    
    # Rename the rain column, replace values -1 with 0.25, and convert to mm(/hr):
    Weather.rename(columns={"RH":"rain"},inplace=True)    # Rename rain column
    Weather.loc[Weather.rain==-1, 'rain'] = 0.25          # Rain value '-1' means 0mm < rain < 0.05mm -> use 0.025mm = 0.25 x 0.1 mm - https://stackoverflow.com/a/28541443/1386750
    Weather.loc[:, 'rain'] /= 10                          # Rain (0.1 mm(/hr)) -> mm(/hr)
    
    # Rename the temperature column and convert to °C:
    Weather.rename(columns={'T':'temp'},inplace=True)     # Rename temperature column
    Weather.loc[:, 'temp'] /= 10                          # Temperature (0.1°C) -> °C
    
    # Rename the pressure column and convert to mbar/hPa:
    Weather.rename(columns={'P':'press'},inplace=True)    # Rename pressure column
    Weather.loc[:, 'press'] /= 10                         # Pressure (0.1 mbar) -> mbar
    
    # Rename the wind-direction and wind-speed columns, and convert to m/s:
    Weather.rename(columns={'DD':'wd'},inplace=True)      # Rename wind-direction column
    Weather.rename(columns={'FH':'ws'},inplace=True)      # Rename wind-speed column
    Weather.loc[:, 'ws'] /= 10                            # Wind speed (0.1 m/s) -> m/s
    
    # Rename the relative-humidity column:
    Weather.rename(columns={'U':'rh'},inplace=True)       # Rename relative-humidity column
    
    
    return Weather




def KNMIdatehour2datetime(knmi_data):
    """Convert the KNMI date and hour columns to a single datetime column.
    
    The KNMI date is expressed as an integer formatted as YYYYMMDD, while the hours run from 1-24 rather than
    from 0-23.  This causes problems when converting to Python or Pandas datetime objects.
    
    Parameters:
      knmi_data (Pandas df):  KNMI weather dataframe.
    
    Returns:
      (Pandas df):  KNMI weather dataframe.
    
    """
    
    # Split the YYYYMMDD column into separate numpy arrays:
    ymd     = knmi_data['YYYYMMDD'].values  # Numpy array
    years   = np.floor(ymd/1e4).astype(int)
    
    months  = np.floor((ymd - years*1e4)/100).astype(int)
    days    = np.floor(ymd - years*1e4 - months*100).astype(int)
    
    # Create numpy arrays for the time variables:
    hours   = knmi_data['HH'].values  # Numpy array
    minutes = np.zeros(hours.size)
    seconds = np.zeros(hours.size) + 0.001  # 1 ms past the hour, to ensure no negative round-off values occur (e.g. 2021,1,1, 0,0,-1e-5 -> 2020,12,31, 23,59,59.99999)
    
    # Fix the dates, e.g. turning 2020-12-31 24:00:00 to 2021-01-01 00:00:00:
    years,months,days, hours,minutes,seconds = fix_date_time(years,months,days, hours,minutes,seconds)
    
    # Combine the 1D numpy arrays into a single 2D array with the original arrays as COLUMNS, and convert it to a Pandas df:
    dts = pd.DataFrame(np.vstack([years,months,days,hours]).transpose(), columns=['year','month','day','hour'])
    dts = pd.to_datetime(dts, utc=True)    # Turn the columns in the df into a single datetime64[ns] column
    
    # Add the datetime column to the KNMI weather dataframe:
    knmi_data['datetime'] = dts
    
    return knmi_data


