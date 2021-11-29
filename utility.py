#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 09:34:57 2021

@author: sichuan
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.patches import Circle, PathPatch
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

    num = np.shape(x1)[0]
    dirFoc = np.zeros(num)

    for i in range(0,num):
        dirFoc[i] = math.atan2(dy[i],dx[i])

    xmin = min(min(x1.iloc[:]),min(x2.iloc[:]))
    xmax = max(max(x1.iloc[:]),max(x2.iloc[:]))

    ymin = min(min(y1.iloc[:]),min(y2.iloc[:]))
    ymax = max(max(y1.iloc[:]),max(y2.iloc[:]))

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

    # set up the boundary of the force chains 
    ax.plot([xmin,xmax],[ymin,ymin],'k-')
    ax.plot([xmin,xmax],[ymax,ymax],'k-')
    ax.plot([xmin,xmin],[ymin,ymax],'k-')
    ax.plot([xmax,xmax],[ymin,ymax],'k-')

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

def local_force_processing(filename):
    input1 = filename + '.xlsx'
    outfile1 = filename + '_localnetwork.png'
    outfile2 = filename + '_localpolarhist.png'
    
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

    # calculate the direction of the force chains
    num = np.shape(x1)[0]
    dirFoc = np.zeros(num)
    for i in range(0,num):
        dirFoc[i] = math.atan2(dy[i],dx[i])
    print(type(dirFoc))
    print(type(Fn))

    localFn = pd.DataFrame(columns = ['LocalFn'])
    localDir = pd.DataFrame(columns = ['LocalDir'])

    for i in range (0, num):
        if ((xc.iloc[i] > xcmin) & (xc.iloc[i] < xcmax) & (yc.iloc[i] > ycmin) & (yc.iloc[i] < ycmax)):
            localFn = localFn.append({'localFn': Fn.iloc[i]}, ignore_index = True)
            localDir = localDir.append({'localDir': dirFoc[i]}, ignore_index = True)

    # set up the force chain drawing environment
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

    theta = localDir.iloc[:,1]
    radii = localFn.iloc[:,1]
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

def connecting_contact(pid, data):

    cp = []
    num = np.shape(data)[0]
    for i in range (0, num):
        if (data.iloc[i,1] == pid):
            cp.append(data.iloc[i,0])

    return  cp

def particle_unbalanced_force(cp, data):

    unbalF = np.zeros(2)
    for i in range (0, len(cp)):
        xni = data.iloc[cp[i]-1, 3] - data.iloc[cp[i]-1, 5]
        yni = data.iloc[cp[i]-1, 4] - data.iloc[cp[i]-1, 6]
        dir = np.array([xni,yni])/np.sqrt(xni**2 + yni**2)
        Fi = data.iloc[cp[i]-1, 9] * dir
        unbalF = unbalF + Fi

    return unbalF

def get_particle_info(pid,data):

    num = np.shape(data)[0]
    xp = 0.0
    yp = 0.0
    rp = 0.0
    for i in range (0, num):
        if (data.iloc[i,1] == pid):
            xp = data.iloc[i,3]
            yp = data.iloc[i,4]
            rp = data.iloc[i,7]
            break

    return xp, yp, rp

def get_unbalance_force_field(filename):
    
    input = filename + '.xlsx'
    output = filename + '_unbal.csv'
    data = pd.read_excel(input)
    nump = max(max(data.iloc[:,1]), max(data.iloc[:,2]))
    unbalance_forces = np.zeros((nump,7))

    for i in range (0, nump):
        
        cp = []
        pid = i + 1
        unbalF = np.zeros(2)
        cp = connecting_contact(pid, data)
        unbalF = particle_unbalanced_force(cp, data)
        xp, yp, rp = get_particle_info(pid,data)

        unbalance_forces[i][0] = pid
        unbalance_forces[i][1] = unbalF[0]
        unbalance_forces[i][2] = unbalF[1]
        unbalance_forces[i][3] = np.sqrt(unbalF[0]**2 + unbalF[1]**2)
        unbalance_forces[i][4] = xp
        unbalance_forces[i][5] = yp
        unbalance_forces[i][6] = rp
    
    np.savetxt(output, unbalance_forces, delimiter=",")
    
    return

def plot_unbalance_force_field(filename):

    input = filename + '_unbal.csv'
    output = filename + '_unbal.png'
    data = np.loadtxt(input,delimiter=",")
    n,m = np.shape(data)

    pid  = pd.DataFrame(data[:,0], columns=['ID'])
    ubFx = pd.DataFrame(data[:,1], columns=['Unbalanced Fx'])
    ubFy = pd.DataFrame(data[:,2], columns=['Unbalanced Fy'])
    ubF  = pd.DataFrame(data[:,3], columns=['Unbalanced F'])
    xp   = pd.DataFrame(data[:,4], columns=['ball xpos'])
    yp   = pd.DataFrame(data[:,5], columns=['ball ypos'])
    rp   = pd.DataFrame(data[:,6], columns=['ball radii'])

    fig, ax = plt.subplots(figsize=(4.0,3.0))
    timer = fig.canvas.new_timer(interval = 3000) #creating a timer object and setting an interval of 3000 milliseconds
    timer.add_callback(close_figwindow)

    cm = plt.cm.get_cmap('viridis') # color bar
    z = ubF.iloc[:,0] # color list
    sc = plt.scatter(xp.iloc[:,0], yp.iloc[:,0], c=z, vmin=0, vmax=max(ubF.iloc[:,0]), s=2*np.sqrt(rp.iloc[:,0]), cmap=cm)
    plt.colorbar(sc)
    ax.set_ylim(-5,50)
    ax.set_xlim(0,60)
    ax.invert_yaxis()
    ax.axis('equal')
    plt.savefig(output,dpi = 600)
    plt.tight_layout()
    timer.start()
    plt.show()
    
    return

def draw_particle(xc,yc,r):
    theta = theta = np.linspace(0.0,2.0*np.pi,120)
    a = r * np.cos(theta) + xc * np.ones(120)
    b = r * np.sin(theta) + yc * np.ones(120)
    return a, b

def get_max_unbalance_particle(filename):
    input = filename + '_unbal.csv'
    data = np.loadtxt(input,delimiter=",")
    n,m = np.shape(data)
    id  = pd.DataFrame(data[:,0], columns=['ID'])
    ubF  = pd.DataFrame(data[:,3], columns=['Unbalanced F'])
    max_ubF = max(ubF.iloc[:,0])

    for i in range (0,n):
        if (ubF.iloc[i,0]==max_ubF):
            pid = int(id.iloc[i,0])
            break
    return pid

def mark_unbalance_force_field(pid,filename):

    input = filename + '_unbal.csv'
    output = filename + '_unbal_mark.png'
    data = np.loadtxt(input,delimiter=",")
    n,m = np.shape(data)

    id  = pd.DataFrame(data[:,0], columns=['ID'])
    ubFx = pd.DataFrame(data[:,1], columns=['Unbalanced Fx'])
    ubFy = pd.DataFrame(data[:,2], columns=['Unbalanced Fy'])
    ubF  = pd.DataFrame(data[:,3], columns=['Unbalanced F'])
    xp   = pd.DataFrame(data[:,4], columns=['ball xpos'])
    yp   = pd.DataFrame(data[:,5], columns=['ball ypos'])
    rp   = pd.DataFrame(data[:,6], columns=['ball radii'])

    fig, ax = plt.subplots(figsize=(4.0,3.0))
    timer = fig.canvas.new_timer(interval = 3000) #creating a timer object and setting an interval of 3000 milliseconds
    timer.add_callback(close_figwindow)

    # plt all particles
    for i in range (0,n):
        a, b = draw_particle(xp.iloc[i,0],yp.iloc[i,0],rp.iloc[i,0])
        plt.plot(a,b,'k-',linewidth = 0.5)
    
    # mark the target particle
    a, b = draw_particle(xp.iloc[pid-1,0],yp.iloc[pid-1,0],rp.iloc[pid-1,0])
    plt.plot(a,b,'r-',linewidth = 1.0)
    ax.set_ylim(-5,50)
    ax.set_xlim(0,60)
    ax.invert_yaxis()
    ax.axis('equal')
    plt.savefig(output,dpi = 600)
    plt.tight_layout()
    timer.start()
    plt.show() 

    cp = []
    data = pd.read_excel(filename + '.xlsx')
    num = np.shape(data)[0]
    for i in range (0, num):
        if (data.iloc[i,1] == pid):
            cp.append(data.iloc[i,0])
    
    print("coordination number of the particle is: ", len(cp))
    print("the associated unbalance force is: ", ubF.iloc[pid-1,0])
    return

def contact_map_drawing(filename):

    input1 = filename + '.xlsx'
    outfile1 = filename + '_network.png'
    
    data = pd.read_excel(input1)
    id1= data.iloc[:,1]
    x1 = data.iloc[:,3]
    y1 = data.iloc[:,4]
    r1 = data.iloc[:,7]
    id2= data.iloc[:,2]
    x2 = data.iloc[:,5]
    y2 = data.iloc[:,6]
    r2 = data.iloc[:,8]
    Fn = data.iloc[:,9]
    dx = (x2.iloc[:]-x1.iloc[:]).to_numpy()
    dy = (y2.iloc[:]-y1.iloc[:]).to_numpy()

    num = np.shape(x1)[0]
    dirFoc = np.zeros(num)

    for i in range(0,num):
        dirFoc[i] = math.atan2(dy[i],dx[i])

    x1min = min(x1.iloc[:])
    y1min = min(y1.iloc[:])
    x2min = min(x2.iloc[:])
    y2min = min(y2.iloc[:])

    x1max = max(x1.iloc[:])
    y1max = max(y1.iloc[:])
    x2max = max(x2.iloc[:])
    y2max = max(y2.iloc[:])

    xmin = min(x1min,x2min)
    xmax = max(x1max,x2max)

    ymin = min(y1min,y2min)
    ymax = max(y1max,y2max)

    if (xmin == x1min):
        idxmin = id1.index[x1.iloc[:] == xmin].tolist()
        rxmin = r1.iloc[idxmin[0]]
    else:
        idxmin = id2.index[x2.iloc[:] == xmin].tolist()
        rxmin = r2.iloc[idxmin[0]]
    
    if (xmax == x1max):
        idxmax = id1.index[x1.iloc[:] == xmax].tolist()
        rxmax = r1.iloc[idxmax[0]]
    else:
        idxmax = id2.index[x2.iloc[:] == xmax].tolist()
        rxmax = r2.iloc[idxmax[0]]

    if (ymin == y1min):
        idymin = id1.index[y1.iloc[:] == ymin].tolist()
        rymin = r1.iloc[idxmin[0]]
    else:
        idymin = id2.index[y2.iloc[:] == ymin].tolist()
        rymin = r2.iloc[idymin[0]]
    
    if (ymax == y1max):
        idymax = id1.index[y1.iloc[:] == ymax].tolist()
        rymax = r1.iloc[idymax[0]]
    else:
        idymax = id2.index[y2.iloc[:] == ymax].tolist()
        rymax = r2.iloc[idymax[0]]

    Fmax = max(Fn.iloc[:])
    Fave = np.average(Fn)

    fig, ax = plt.subplots(figsize=(2.1,1.85),dpi=600)
    timer = fig.canvas.new_timer(interval = 3000) #creating a timer object and setting an interval of 3000 milliseconds
    timer.add_callback(close_figwindow)

    for i in range(0,num):
        circle = Circle((x1.iloc[i],y1.iloc[i]), r1.iloc[i], fill=False, edgecolor=(0.3,0.3,0.3), linewidth=0.2, alpha=0.5)
        ax.add_patch(circle)
    
    # ax.set_xlim([0,60])
    # ax.set_ylim([0,60])
    # ax.axis('equal')
    # ax.invert_yaxis()
    # ax.set_xticks([])
    # ax.set_yticks([])

    # # set up the boundary of the force chains 
    # ax.plot([xmin-rxmin,xmax+rxmax],[ymin-rymin,ymin-rymin],'k-')
    # ax.plot([xmin-rxmin,xmax+rxmax],[ymax+rymax,ymax+rymax],'k-')
    # ax.plot([xmin-rxmin,xmin-rxmin],[ymin-rymin,ymax+rymax],'k-')
    # ax.plot([xmax+rxmax,xmax+rxmax],[ymin-rymin,ymax+rymax],'k-')

    ax.set_xlim([xmin-rxmin,xmax+rxmax])
    ax.set_ylim([ymin-rymin,ymax+rymax])
    ax.set_aspect(1)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    # plot all the force chains
    for i in range(0,num):
        if (Fn.iloc[i]>Fave):
            linethickness = (Fn.iloc[i]/Fmax)*2.0
            ax.plot([x1.iloc[i],x2.iloc[i]],[y1.iloc[i],y2.iloc[i]],'r-',linewidth = linethickness)

    fig.subplots_adjust(bottom=0.01, top=0.99, left=0.01, right=0.99, wspace=0.25, hspace=0.05)
    plt.savefig(outfile1, dpi = 600)
    timer.start()
    plt.show()
    return