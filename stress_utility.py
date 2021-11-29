import numpy as np
import pandas as pd
import math

def GetMeasureRegionData(data, region):
    # get contact position
    data['cxi'] = 0.5*(data.iloc[:,3] + data.iloc[:,5])
    data['cyi'] = 0.5*(data.iloc[:,4] + data.iloc[:,6])
    # measurecontacts = [] # contact id within the region
    x1, x2, y1, y2 = region[0], region[1], region[2], region[3]
    Vr = (x2-x1)*(y2-y1)*(1.0e-4)
    CL = [] # branch vector list
    CF = [] # contact force list
    maxcid = np.max(data.iloc[:,0])
    maxcid = np.size(data,0)
    for i in range (0,maxcid):
        if (data.iloc[i,10] > x1) & (data.iloc[i,10] < x2):
            if (data.iloc[i,11] > y1) & (data.iloc[i,11] < y2):
                # measurecontacts.append(data.iloc[i,0])
                CL.append([data.iloc[i,5]-data.iloc[i,3], data.iloc[i,6]-data.iloc[i,4]])
                CF.append(data.iloc[i,9])
    return CF, CL, Vr

def compute_stress(CF,CL,Vr):
    xxmstress = 0
    yymstress = 0
    xymstress = 0
    yxmstress = 0
    fx = 0
    fy = 0
    for i in range (0, len(CF)):
        fx = CF[i]*CL[i][0]/np.sqrt(CL[i][0]**2 + CL[i][1]**2)
        fy = CF[i]*CL[i][1]/np.sqrt(CL[i][0]**2 + CL[i][1]**2)
        xxmstress = xxmstress + fx*CL[i][0]
        yymstress = yymstress + fy*CL[i][1]
        xymstress = xymstress + fx*CL[i][1]
        yxmstress = yxmstress + fy*CL[i][0]
    xxmstress = -xxmstress/Vr
    yymstress = -yymstress/Vr
    xymstress = -xymstress/Vr
    yxmstress = -yxmstress/Vr
    stress = [xxmstress, yymstress, xymstress, yxmstress]
    return stress
