# -*- coding: utf-8 -*-
"""
Visualising the Stonex R20 measurement points 
=============================================
Most of the bottom screws of the LAN ports in Z723 were surveyed. 
The door was set as North (0 azimuth). 

Created on Sun Dec 21 17:18:34 2025

@author: theja
"""
import matplotlib.pyplot as plt 
import scipy 
import numpy 
import pandas as pd
import glob

df = pd.concat([pd.read_csv(each, delimiter=',') for each in glob.glob('*.TXT')]).reset_index(drop=True)

df.columns = ['PtID','x','y','z','Code']
station_pts = df[df['Code']=='STATION']
plt.figure()
a0 = plt.subplot(111,projection='3d')
plt.plot(df.loc[:,'x'], df.loc[:,'y'], df.loc[:,'z'],'*')
plt.plot(station_pts.loc[:,'x'], station_pts.loc[:,'y'], station_pts.loc[:,'z'],'r*')
a0.set_zlim(-1.5,1.5)


