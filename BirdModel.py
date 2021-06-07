#!/bin/env python3

import colored_traceback
colored_traceback.add_hook()

import numpy as np
import pandas as pd
from pvlib.clearsky import bird
from pvlib.atmosphere import get_relative_airmass
from pvlib.atmosphere import gueymard94_pw
import matplotlib.pyplot as plt
import solarenergy as se
import time
from dataclasses import dataclass

from functions import read_knmi_weather_data


r2d = se.r2d  # Multiplication factor to convert radians to degrees
d2r = se.d2r  # Multiplication factor to convert degrees to radians


@dataclass
class SolarPanels:
    """Dataclass containing solar-panel data."""
    
    # Geographic location:
    geoLon:  float = 0.0
    geoLat:  float = 0.0
    
    # Orientation:
    az:      float = 0.0  # 'Azimuth'; 0=S (rad)
    incl:    float = 0.0  # 'Zenith angle'; 0=horizontal (rad)
    
    # Size and capacity:
    area:    float = 0.0  # Surface area of solar panels (m2)
    eff:     float = 0.0  # Efficiency of solar panels (0-1)
    Pmax:    float = 0.0  # Maximum power of solar panels (kW)
    
    
def main():
    """Main function"""
    # Measure run time:
    tc0 = time.process_time()  # CPU time
    
    # Read solar-panel data:
    print("\nReading data...")
    SolPan  = pd.read_csv('data/Pl14_SP_alldata_hourly.csv', sep=r'\s*,\s*', engine='python')
    
    # Read WeerPlaza weather data:
    # Weather = pd.read_csv('data/wp_weer_36h_all.csv', sep=r'\s*,\s*', engine='python')
    
    # Read KNMI weather data:
    Weather = read_knmi_weather_data('data/knmi_uurgeg_Lent.csv')
    # print(Weather)
    
    data = pd.merge(SolPan, Weather)
    nDat = len(data.Pmean)
    print("Total number of data points overlapping between solar-panel data and weather data: ", nDat)
    
    # Select data:
    print("Selecting data:  ", end="")
    # 20190611-13: no weather data
    
    # 20190610 has strong variation in clouds and rain
    # 20190616 has strong variation in clouds and rain
    # 20190619 has rain
    # 20190623 starts clear, ends clouded
    # 20190625 is ~80% clouded (~constant)
    # 20190629 is clear
    
    # year  = 2019
    # month =    6
    # day   =   10
    # year  = 2020
    # month =    3
    # day   =   19
    # data  = data[ (data.year==year) & (data.mo==month) & (data.dy==day) ]
    # data  = data[ (data.year==year) & (data.mo==month) & (data.dy==day) & (data.hr==12) ]
    # data  = data[ (data.year==year) & (data.mo==month) ]
    # data  = data[ (data.year==2019) ]  # (all of) 2019 only - 4429 data points, ~85s
    # data  = data[ (data.year==2019) & (data.hr>12) & (data.hr<16) ]  # 2019; 13-15h only - 1080 data points, ~20s
    # data  = data[ (data.year == 2019) & (data.mo>3) & (data.mo<10) & (data.hr>8) & (data.hr<22) ]  # April-September 2019; 13-15h - 540 points, ~10s
    # data  = data[ (data.year == 2019) & (data.mo>3) & (data.mo<10) & (data.hr>11) & (data.hr<15) ]  # April-September 2019; 13-15h - 540 points, ~10s
    # data  = data[ (data.year == 2019) & (data.mo>4) & (data.mo<9) & (data.hr>8) & (data.hr<22) ] #May -August 2019 ; 9h-21h ; 1584 pts ~ 13 s
    # data  = data[ (data.year == 2019) & (data.mo==6) & (data.dy==21) & (data.hr>8) & (data.hr<22) ] #21 June 2019 ; 9h-21h ;
    
    # data  = data[ (data.year == 2019) & (data.mo<4) & (data.mo>9) & (data.hr>9) & (data.hr<20) ] #october -march 2019 ; 9h-21h ;
    
    # data  = data[ (data.year == 2019) & (data.mo>4) & (data.mo<9) & (data.hr>8) & (data.hr<22) & (data.clouds != 0) & (data.Pmean > 0.001)] #May -August 2019 ; 9h-21h ; only clouds ! ; 1584 pts ~ 13 s
    
    # data  = data[ (data.year == 2019) & (data.mo>4) & (data.mo<9) & (data.hr>8) & (data.hr<22) & (data.clouds != 0) & (data.Pmean > 0.01) & (data.rain >0)] #May -August 2019 ; 9h-21h ; 293 pts ~ 13 s& (data.rain > 0)
    
    # data  = data[ (data.year == 2019) & (data.mo>4) & (data.mo<9) & (data.hr>8) & (data.hr<22) & (data.clouds != 0) & (data.Pmean > 0.01) ] #May -August 2019 ; 9h-21h ; 293 pts ~ 13 s
    
    
    # data  = data[ (data.year == 2019) & (data.hr>8) & (data.hr<22) &                (data.Pmean > 0.01) & (data.clouds > 0)] #Year 2019 ; 9h-21h ; only clouds ;  3606 pts
    # data  = data[ (data.year == 2019) & (data.hr>8) & (data.hr<22) &                (data.Pmean > 0.01) & (data.rain > 0)] #Year 2019 ; 9h-21h ; only rain ;  992 pts
    # data  = data[(data.hr>8) & (data.hr<22) & (data.Pmean > 0.01) & (data.clouds > 0)] # 13953 points
    # data  = data[(data.hr>8) & (data.hr<22) & (data.clouds > 0) & (data.Pmean > 0.01) & (data.rain >0)] # 3458 points
    # data  = data[ (data.Pmean > 0.01) & (data.clouds > 0) & ((data.mo>10) | (data.mo<4))] 
    
    # data  = data[ (data.Pmean > 0.01) & (data.clouds > 0) & (data.mo>3) & (data.mo<10) & (data.rain >0)]#Summers rain
    # data  = data[ (data.Pmean > 0.01) & (data.clouds > 0) & (data.mo>3) & (data.mo<10)]#Summers
    # data  = data[ (data.Pmean > 0.01) & (data.clouds > 0)]
    data  = data[ (data.Pmean > 0.01) & (data.clouds > 0)& (data.rain >0)]
    
    nDat = len(data.Pmean)
    print("number of data points left: ", nDat)
    
    
    # Solar-panel data:
    global sp
    sp = SolarPanels()
    sp.geoLon  =  5.950270*d2r   # Geographical longitude of solar panels
    sp.geoLat  =  51.987380*d2r  # Geographical latitude of solar panels
    sp.az      = -2*d2r          # 'Azimuth'; 0=S (rad)
    sp.incl    =  28*d2r         # 'Zenith angle'; 0=horizontal (rad)
    sp.area    =  15*1.6         # Surface area of solar panels (m2)
    sp.eff     =  0.12           # Efficiency of solar panels (0-1)
    sp.Pmax    =  3000           # Maximum power of solar panels (or inverter) (W)
    
    print(data)
    computeCloudspower(data, cls=False, sunaltitude=False, cloudcover=False, rain=True, humidity=False)
    
    # Print run time:
    tc1 = time.process_time()
    print()
    print('CPU time:   %0.2f s' % (tc1-tc0))
    
    
def computeCloudspower(data, cls, sunaltitude, cloudcover,  rain, humidity):
    """
    data     : the DataFrame
    cls      : True includes cloud cover, cls = False does not include cloud cover
    
    """
    
    global sp
    
    # Sun altitude part-------------------------------------------------------
    
    if(sunaltitude):
        SunAltData     = []
        CloudsPowerSA  = []
        CorrectionSASineSunAlt = []
        if cls:
            CorrectionSAClouds  = []
            CorrectionSAClsExtF = []
        CorrectionSAExtF = []
        
        for iRow in range(len(data)):
            row = data.iloc[iRow]
            sunAz,sunAlt,sunDist = se.sun_position_from_date_and_time(sp.geoLon, sp.geoLat,
                                                                      row.year, row.mo, row.dy, row.hr,
                                                                      timezone='Europe/Amsterdam')
            # if((row.year==2019) & (row.mo==12) & (row.dy==21)):
            #     print("%4i-%2.2i-%2.2i %2.2ih  %8.3f  %8.3f  %8.1f" % (row.year, row.mo, row.dy, row.hr, sunAlt*r2d, row.Pmean, row.press))
                
            if(sunAlt > 5*d2r):
                SunAltData.append(sunAlt*r2d)
                
                zenith    = 90 - sunAlt*r2d
                airmass   = get_relative_airmass(zenith,model='kastenyoung1989')
                aod380    = 0.3538
                aod500    = 0.2661
                precipitable_water = gueymard94_pw(row.temp, row.rh)  # Temperature in °C and relative humidity in %
                
                BirdArray = bird(zenith, airmass, aod380, aod500, precipitable_water,
                                 ozone=0.3, pressure=101325., dni_extra=1364.,
                                 asymmetry=0.85, albedo=0.2)
                BirdPower = pd.Series(BirdArray)
                
                # Power yield  by the DNI
                cosTheta       = se.cos_angle_sun_panels(sp.az,sp.incl, sunAz,sunAlt)
                PowerBirdDni   = max(BirdPower.dni * max(cosTheta,0),0) * sp.area *sp.eff
                PowerBirdDni   = min(PowerBirdDni,sp.Pmax)
                
                
                # Power yield  by the DHI
                PowerBirdDhi   = max(BirdPower.dhi * (1 + np.cos(sp.incl))/2, 0) * sp.area * sp.eff
                PowerBirdDhi   = min(PowerBirdDhi,sp.Pmax)
                
                # Compution Clouds DHI power
                ClsPowSA    = max(row.Pmean*1000 - (1 - row.clouds/100) * (PowerBirdDni + PowerBirdDhi),0)  # Clouds power = Power on the panel - a blue sky DHI % - DNI
                CloudsPowerSA.append(ClsPowSA)
                
                # Compensate for sun altitude influence:
                ClsPowSA    = ClsPowSA / np.sin(sunAlt)
                
                CorrectionSASineSunAlt.append(ClsPowSA)
                
                # Compensate for clouds influence:
                if cls:
                    ClsPowSACls  = ClsPowSA / (row.clouds/100)  # looks like a linear function with a lot of scatter
                    CorrectionSAClouds.append(ClsPowSACls)
                    
                    # Compensate for extinction factor influence:
                    extFac      = se.extinction_factor(airmass)
                    ClsPowSACls = ClsPowSACls * extFac
                    
                    CorrectionSAClsExtF.append(ClsPowSACls)
                    
                # Compensate for extinction factor influence:
                extFac   = se.extinction_factor(airmass)
                ClsPowSA = ClsPowSA * extFac
                
                CorrectionSAExtF.append(ClsPowSA)
                
        # return  # Don't plot
        
        plt.close('all')  # in case previous savings were not closed
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(SunAltData, CloudsPowerSA, s=3)
        ax.set_ylabel('Clouds power (W)')
        ax.set_xlabel('Sun altitude (°)')
        ax.grid(True)
        r = np.corrcoef(SunAltData, CloudsPowerSA)
        corr = r[0,1]
        Pearson = 'Pearson correlation coefficient: '+str(corr)
        print(Pearson)
        plt.figtext(0.55, 0.9,Pearson,horizontalalignment="center",verticalalignment="center",color="red")
        plt.show()
        # fig.savefig('PwCls.pdf')
        # fig.savefig('SummersPwCls.pdf')
        plt.close()
        
        # plt.grid()
        # plt.scatter(SunAltData, CorrectionSASineSunAlt, s = 3)
        # plt.xlabel('Sun altitude (°)')
        # plt.ylabel('Corrected clouds power')
        # plt.savefig('SinSunAlt.pdf')
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(SunAltData, CorrectionSASineSunAlt, s=3)
        ax.set_ylabel('Corrected clouds power')
        ax.set_xlabel('Sun altitude (°)')
        ax.grid(True)
        r = np.corrcoef(SunAltData, CorrectionSASineSunAlt)
        corr = r[0,1]
        Pearson = 'Pearson correlation coefficient: '+str(corr)
        print(Pearson)
        plt.figtext(0.55, 0.9,Pearson,horizontalalignment="center",verticalalignment="center",color="red")
        # fig.savefig('SinSunAlt.pdf')
        # fig.savefig('SummersSinSunAlt.pdf')
        plt.show()
        
        plt.close()
        
        if cls:
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(SunAltData, CorrectionSAClouds, s=3)
            ax.set_ylabel('Corrected clouds power')
            ax.set_xlabel('Sun altitude (°)')
            ax.grid(True)
            
            r = np.corrcoef(SunAltData, CorrectionSAClouds)
            corr = r[0,1]
            Pearson = 'Pearson correlation coefficient: '+str(corr)
            print(Pearson)
            plt.figtext(0.55, 0.9,Pearson,horizontalalignment="center",verticalalignment="center",color="red")
            
            # fig.savefig('SinSulAlt_CloudCover.pdf')
            # fig.savefig('SummersSinSulAlt_CloudCover.pdf')
            plt.close()
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(SunAltData, CorrectionSAClsExtF, s=3)
            ax.set_ylabel('Corrected clouds power')
            ax.set_xlabel('Sun altitude (°)')
            ax.grid(True)
            
            r = np.corrcoef(SunAltData, CorrectionSAClsExtF)
            corr = r[0,1]
            Pearson = 'Pearson correlation coefficient: '+str(corr)
            print(Pearson)
            plt.figtext(0.55, 0.9,Pearson,horizontalalignment="center",verticalalignment="center",color="red")
            
            # fig.savefig('ExtinctionFactor.pdf')
            # fig.savefig('SummersExtinctionFactor.pdf')
            plt.close()
            
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(SunAltData, CorrectionSAExtF, s=3)
        ax.set_ylabel('Corrected clouds power')
        ax.set_xlabel('Sun altitude (°)')
        ax.grid(True)
        
        r = np.corrcoef(SunAltData, CorrectionSAExtF)
        corr = r[0,1]
        Pearson = 'Pearson correlation coefficient: '+str(corr)
        print(Pearson)
        plt.figtext(0.55, 0.9,Pearson,horizontalalignment="center",verticalalignment="center",color="red")
        
        # fig.savefig('ExtinctionFactor_Nocls.pdf')
        # fig.savefig('SummersExtinctionFactor_Nocls.pdf')
        # plt.show()
        plt.show()
        plt.close()
        
        
# Rainfall Part---------------------------------------------------------

    if(rain):
        RainData      = []
        CorrectionRExtF    = []
        # CorrectionRLn      = []
        
        for iRow in range(len(data)):
            row = data.iloc[iRow]
            sunAz,sunAlt,sunDist = se.sun_position_from_date_and_time(sp.geoLon, sp.geoLat,
                                                                      row.year, row.mo, row.dy, row.hr,
                                                                      timezone='Europe/Amsterdam')
            if(sunAlt > 5*d2r):
                RainData.append(row.rain)
                
                zenith    = 90 - sunAlt*r2d
                airmass   = get_relative_airmass(zenith,model='kastenyoung1989')
                aod380    = 0.3538
                aod500    = 0.2661
                precipitable_water = gueymard94_pw(row.temp, row.rh)  # Temperature in °C and relative humidity in %
                
                BirdArray = bird(zenith, airmass, aod380, aod500, precipitable_water,
                                 ozone=0.3, pressure=101325., dni_extra=1364.,
                                 asymmetry=0.85, albedo=0.2)
                BirdPower = pd.Series(BirdArray)
                
                # Power yield  by the DNI
                cosTheta       = se.cos_angle_sun_panels(sp.az,sp.incl, sunAz,sunAlt)
                PowerBirdDni   = max(BirdPower.dni * max(cosTheta,0),0) * sp.area *sp.eff
                PowerBirdDni   = min(PowerBirdDni,sp.Pmax)
                
                
                # Power yield  by the DHI
                PowerBirdDhi   = max(BirdPower.dhi * (1 + np.cos(sp.incl))/2, 0) * sp.area * sp.eff
                PowerBirdDhi   = min(PowerBirdDhi,sp.Pmax)
                
                # Compution Clouds DHI power
                ClsPowR    = max(row.Pmean*1000 - (1 - row.clouds/100) * (PowerBirdDni + PowerBirdDhi),0)  # Clouds power = Power on the panel - a blue sky DHI % - DNI
                
                # Compensate for sun altitude influence:
                ClsPowR    = ClsPowR / np.sin(sunAlt)
                
                # Compensate for cloud cover:
                ClsPowR    = ClsPowR / (row.clouds/100)
                
                # Compensate for extinction factor influence:
                extFac    = se.extinction_factor(airmass)
                ClsPowR   = ClsPowR * extFac
                CorrectionRExtF.append(ClsPowR)
                
                # Correction by the shape function
                
                # ClsPowR   = ClsPowR * np.log(row.rain + 1)
                # ClsPowR   = ClsPowR * np.log(row.rain + 1)**2
                # ClsPowR   = ClsPowR * row.rain**2
                # CorrectionRLn.append(ClsPowR)
                
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(RainData, CorrectionRExtF, s=3)
        ax.set_ylabel('Corrected clouds power')
        ax.set_xlabel('Rainfall (mm/h)')
        ax.grid(True)
        
        r = np.corrcoef(RainData, CorrectionRExtF)
        corr = r[0,1]
        Pearson = 'Pearson correlation coefficient: '+str(corr)
        print(Pearson)
        plt.figtext(0.55, 0.9,Pearson,horizontalalignment="center",verticalalignment="center",color="red")
        
        fig.savefig('FullCorrection_AsFRain.pdf')
        # fig.savefig('SummersFullCorrection_AsFRain.pdf')
        plt.close()
        
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.scatter(RainData, CorrectionRLn, s = 3)
        # ax.set_ylabel('Corrected clouds power')
        # ax.set_xlabel('Rainfall (mm/h)')
        # ax.grid(True)
        
        # r = np.corrcoef(SunAltData, CorrectionRLn)
        # corr = r[0,1]
        # Pearson = 'Pearson correlation coefficient: '+str(corr)
        # print(Pearson)
        # plt.figtext(0.55, 0.9,Pearson,horizontalalignment ="center",verticalalignment ="center",color ="red")
        
        # #fig.savefig('CorrectionLn_AsFRain.pdf')
        # plt.close()
        
# Cloud cover part---------------------------------------------------------
    if(cloudcover):
        CloudsData  = []
        CloudsPowerCC    = []
        for iRow in range(len(data)):
            row = data.iloc[iRow]
            sunAz,sunAlt,sunDist = se.sun_position_from_date_and_time(sp.geoLon, sp.geoLat,
                                                                      row.year, row.mo, row.dy, row.hr,
                                                                      timezone='Europe/Amsterdam')
            if(sunAlt>5*d2r):
                CloudsData.append(row.clouds)
                
                zenith    = 90 - sunAlt*r2d
                airmass   = get_relative_airmass(zenith,model='kastenyoung1989')
                aod380    = 0.3538
                aod500    = 0.2661
                precipitable_water = gueymard94_pw(row.temp, row.rh)  # Temperature in °C and relative humidity in %
                
                BirdArray = bird(zenith, airmass, aod380, aod500, precipitable_water,
                                 ozone=0.3, pressure=101325., dni_extra=1364.,
                                 asymmetry=0.85, albedo=0.2)
                BirdPower = pd.Series(BirdArray)
                
                # Power yield  by the DNI
                cosTheta       = se.cos_angle_sun_panels(sp.az,sp.incl, sunAz,sunAlt)
                PowerBirdDni   = max(BirdPower.dni * max(cosTheta,0),0) * sp.area *sp.eff
                PowerBirdDni   = min(PowerBirdDni,sp.Pmax)
                
                
                # Power yield  by the DHI
                PowerBirdDhi   = max(BirdPower.dhi * (1 + np.cos(sp.incl))/2, 0) * sp.area * sp.eff
                PowerBirdDhi   = min(PowerBirdDhi,sp.Pmax)
                
                # Compution Clouds DHI power
                ClsPowCC    = max(row.Pmean*1000 - (1 - row.clouds/100) * (PowerBirdDni + PowerBirdDhi),0)  # Clouds power = Power on the panel - a blue sky DHI % - DNI
                CloudsPowerCC.append(ClsPowCC)
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(CloudsData, CloudsPowerCC, s=3)
        ax.set_ylabel('Clouds power (W)')
        ax.set_xlabel('Cloud Cover (%)')
        ax.grid(True)
        # fig.savefig('ClsPow_Cls.pdf')
        # fig.savefig('SummersClsPow_Cls.pdf')
        plt.close()
        
# Humidity part -------------------------------------------------------------
    if(humidity):
        HumidityData      = []
        CorrectionHExtF    = []
        
        for iRow in range(len(data)):
            row = data.iloc[iRow]
            sunAz,sunAlt,sunDist = se.sun_position_from_date_and_time(sp.geoLon, sp.geoLat,
                                                                      row.year, row.mo, row.dy, row.hr,
                                                                      timezone='Europe/Amsterdam')
            if(sunAlt > 5*d2r):
                HumidityData.append(row.rh)
                
                zenith    = 90 - sunAlt*r2d
                airmass   = get_relative_airmass(zenith,model='kastenyoung1989')
                aod380    = 0.3538
                aod500    = 0.2661
                precipitable_water = gueymard94_pw(row.temp, row.rh)  # Temperature in °C and relative humidity in %
                
                BirdArray = bird(zenith, airmass, aod380, aod500, precipitable_water,
                                 ozone=0.3, pressure=101325., dni_extra=1364.,
                                 asymmetry=0.85, albedo=0.2)
                BirdPower = pd.Series(BirdArray)
                
                # Power yield by the DNI
                cosTheta       = se.cos_angle_sun_panels(sp.az,sp.incl, sunAz,sunAlt)
                PowerBirdDni   = max(BirdPower.dni * max(cosTheta,0),0) * sp.area *sp.eff
                PowerBirdDni   = min(PowerBirdDni,sp.Pmax)
                
                
                # Power yield by the DHI
                PowerBirdDhi   = max(BirdPower.dhi * (1 + np.cos(sp.incl))/2, 0) * sp.area * sp.eff
                PowerBirdDhi   = min(PowerBirdDhi,sp.Pmax)
                
                # Compution Clouds DHI power
                ClsPowH    = max(row.Pmean*1000 - (1 - row.clouds/100) * (PowerBirdDni + PowerBirdDhi),0)  # Clouds power = Power on the panel - a blue sky DHI % - DNI
                
                # Compensate for sun altitude influence:
                ClsPowH    = ClsPowH / np.sin(sunAlt)
                
                # Compensate for cloud cover:
                ClsPowH    = ClsPowH / (row.clouds/100)
                
                # Compensate for extinction factor influence:
                extFac    = se.extinction_factor(airmass)
                ClsPowH   = ClsPowH * extFac
                CorrectionHExtF.append(ClsPowH)
                
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(HumidityData, CorrectionHExtF, s=3)
        ax.set_ylabel('Corrected clouds power')
        ax.set_xlabel('Relative humidity (%)')
        ax.grid(True)
        
        r = np.corrcoef(HumidityData, CorrectionHExtF)
        corr = r[0,1]
        Pearson = 'Pearson correlation coefficient: '+str(corr)
        print(Pearson)
        plt.figtext(0.55, 0.9,Pearson,horizontalalignment="center",verticalalignment="center",color="red")
        
        fig.savefig('FullCorrection_AsFHumidity.pdf')
        # fig.savefig('SummersFullCorrection_AsFHumidity.pdf')
        plt.close()
    
    
    
    return


if(__name__ == "__main__"): main()



