import numpy as np
import pandas as pd
import math
from stress_utility import GetMeasureRegionData, compute_stress
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
plt.rcParams["font.family"] = "Times New Roman"

path = '/Users/sichuan/Dropbox (ASU)/My Document/Experiment/2D_Photoelastic_Test/Image Processing Results/Geo-Congress/Force Detections/1st Penetration/'
# print("Please make sure your force file has a format like the following...\n")
# print("contact id, ball_1 id, ball_2 id, ball_1 xi, ball_1 yi, ball_2 xi, ball_2 yi, r1, r2, force_i\n")

filename = ['Force_1', 'Force_2', 'Force_3', 'Force_4', 'Force_5', 'Force_6']
stress = []
meanstress = []
devistress = []
for x in filename:
    input = input = path + x + '.xlsx'
    data = pd.read_excel(input)
    x1, x2, y1, y2 = 23, 29, 16, 22
    region = [x1, x2, y1, y2]
    CF, CL, Vr = GetMeasureRegionData(data, region)
    getstress = compute_stress(CF,CL,Vr)
    stress.append(getstress)
    meanstress.append(-0.5*(getstress[0] + getstress[1])*0.001)
    devistress.append(np.sqrt(0.25*(getstress[0]-getstress[1])**2 + (0.5*(getstress[2]+getstress[3]))**2)*0.001)

path = '/Users/sichuan/Dropbox (ASU)/My Document/Experiment/2D_Photoelastic_Test/Image Processing Results/Geo-Congress/Force Detections/2nd Penetration/'
filename = ['Force_1', 'Force_2']
for x in filename:
    input = input = path + x + '.xlsx'
    data = pd.read_excel(input)
    x1, x2, y1, y2 = 23, 29, 16, 22
    region = [x1, x2, y1, y2]
    CF, CL, Vr = GetMeasureRegionData(data, region)
    getstress = compute_stress(CF,CL,Vr)
    stress.append(getstress)
    meanstress.append(-0.5*(getstress[0] + getstress[1])*0.001)
    devistress.append(np.sqrt(0.25*(getstress[0]-getstress[1])**2 + (0.5*(getstress[2]+getstress[3]))**2)*0.001)

fig, ax = plt.subplots(figsize=(3.0,3.0))
plt.plot(meanstress,devistress, '-x')
# ax.set_aspect(1)
ax.set_xlim([0,800])
ax.set_ylim([0,100])
ax.set_xlabel('mean stress (kPa)')
ax.set_ylabel('deviator stress (kPa)')
fig.subplots_adjust(bottom=0.18, top=0.95, left=0.18, right=0.95, wspace=0.25, hspace=0.05)
plt.savefig('stress path.png', dpi=600)
plt.show()

# filename = 'Force_2'
# print('The input file is ' + filename + '.xlsx')

# input = forcepath + filename + '.xlsx'
# data = pd.read_excel(input)
# # define a rectangular measurement region, unit cm
# x1, x2, y1, y2 = 23, 29, 16, 22
# region = [x1, x2, y1, y2]

# CF, CL, Vr = GetMeasureRegionData(data, region)
# stress = compute_stress(CF,CL,Vr)
# meanstress = 0.5*(stress[0] + stress[1])
# devistress = np.sqrt(0.25*(stress[0]-stress[1])**2 + (0.5*(stress[2]+stress[3]))**2)
# print(stress)
# print(meanstress)
# print(devistress)


