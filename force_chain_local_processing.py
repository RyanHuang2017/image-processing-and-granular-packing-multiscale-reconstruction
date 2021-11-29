#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 09:34:57 2021

@author: sichuan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
plt.rcParams["font.family"] = "Times New Roman"

def close_figwindow():
    plt.close()

def force_processing(filename):

    input1 = filename + '.xlsx'
    outfile1 = filename + '_network.png'
    outfile2 = filename + '_polarhist.png'
    
    data = pd.read_excel(input1)
    x1 = data.iloc[:,3]
    y1 = data.iloc[:,4]
    x2 = data.iloc[:,5]
    y2 = data.iloc[:,6]
    Fn = data.iloc[:,9]
    dx = (x2.iloc[:]-x1.iloc[:]).to_numpy()
    dy = (y2.iloc[:]-y1.iloc[:]).to_numpy()

    # calculate the position of contact centers
    xc = 0.5*(data.iloc[:,3] + data.iloc[:,5])
    yc = 0.5*(data.iloc[:,4] + data.iloc[:,6])

    # calculate the direction of the force chains
    num = np.shape(x1)[0]
    dirFoc = np.zeros(num)
    for i in range(0,num):
        dirFoc[i] = math.atan2(dy[i],dx[i])
    
    # external boundary limits
    xmin = min(min(x1.iloc[:]),min(x2.iloc[:]))
    xmax = max(max(x1.iloc[:]),max(x2.iloc[:]))

    ymin = min(min(y1.iloc[:]),min(y2.iloc[:]))
    ymax = max(max(y1.iloc[:]),max(y2.iloc[:]))

    # local measurement boundary limits
    xcmin = xmin + (xmax - xmin)*0.25
    xcmax = xmax - (xmax - xmin)*0.25

    ycmin = ymin + (ymax - ymin)*0.25
    ycmax = ymax - (ymax - ymin)*0.25

    # determine the maximum contact force and average contact force magnitude
    Fmax = max(Fn.iloc[:])
    Fave = np.average(Fn)

    fig, ax = plt.subplots(figsize=(3.0,3.0))
    timer = fig.canvas.new_timer(interval = 3000) #creating a timer object and setting an interval of 3000 milliseconds
    timer.add_callback(close_figwindow)

    ax.set_xlim([0,60])
    ax.set_ylim([0,60])
    ax.axis('equal')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    # set up the external boundary of the force chains 
    ax.plot([xmin,xmax],[ymin,ymin],'k-')
    ax.plot([xmin,xmax],[ymax,ymax],'k-')
    ax.plot([xmin,xmin],[ymin,ymax],'k-')
    ax.plot([xmax,xmax],[ymin,ymax],'k-')

    # set up the local boundary of the measurement area
    ax.plot([xcmin,xcmax],[ycmin,ycmin],'k--')
    ax.plot([xcmin,xcmax],[ycmax,ycmax],'k--')
    ax.plot([xcmin,xcmin],[ycmin,ycmax],'k--')
    ax.plot([xcmax,xcmax],[ycmin,ycmax],'k--')

    # plot all the force chains
    for i in range(0,num):
        if (Fn.iloc[i]>Fave):
            linethickness = (Fn.iloc[i]/Fmax)*2.0
            ax.plot([x1.iloc[i],x2.iloc[i]],[y1.iloc[i],y2.iloc[i]],'r-',linewidth = linethickness)
    
    plt.savefig(outfile1, dpi = 600)
    timer.start()
    plt.show()
    #--------------------------------------------------------
    # polarhistogram plot
    fig = plt.figure(figsize=(3.0,3.0))
    ax = plt.subplot(111, polar=True)
    
    N = 20
    bottom = 0
    max_height = Fmax

    theta = dirFoc
    radii = Fn.iloc[:]
    width = (2*np.pi) / (4*N)
    bars = ax.bar(theta, radii, width=width, bottom=bottom)
    ax.set_rticks([0,25,50,75])
    
    for r, bar in zip(radii, bars):
        bar.set_facecolor(plt.cm.viridis(r / 10.))
        bar.set_alpha(0.8)
    
    plt.savefig(outfile2, dpi = 600)
    timer.start()
    plt.show()
    return