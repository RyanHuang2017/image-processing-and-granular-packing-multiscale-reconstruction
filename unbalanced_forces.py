import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from utility import particle_unbalanced_force, connecting_contact, get_particle_info, get_unbalance_force_field, plot_unbalance_force_field, mark_unbalance_force_field,get_max_unbalance_particle
plt.rcParams["font.family"] = "Times New Roman"

# data = pd.read_excel('Force_6.xlsx')
# nump = max(max(data.iloc[:,1]), max(data.iloc[:,2]))
# unbalance_forces = np.zeros((nump,7))


# for i in range (0, nump):

#     cp = []
#     unbalF = np.zeros(2)

#     cp = connecting_contact(i, data)
#     unbalF = particle_unbalanced_force(cp, data)
#     xp, yp, rp = get_particle_info(i,data)

#     unbalance_forces[i][0] = i + 1
#     unbalance_forces[i][1] = unbalF[0]
#     unbalance_forces[i][2] = unbalF[1]
#     unbalance_forces[i][3] = np.sqrt(unbalF[0]**2 + unbalF[1]**2)
#     unbalance_forces[i][4] = xp
#     unbalance_forces[i][5] = yp
#     unbalance_forces[i][6] = rp
#     print(i)

# np.savetxt("Force_5_unbal.csv", unbalance_forces, delimiter=",")

# data = np.loadtxt('Force_3_unbal.csv',delimiter=",")
# n,m = np.shape(data)

# pid  = pd.DataFrame(data[:,0], columns=['ID'])
# ubFx = pd.DataFrame(data[:,1], columns=['Unbalanced Fx'])
# ubFy = pd.DataFrame(data[:,2], columns=['Unbalanced Fy'])
# ubF  = pd.DataFrame(data[:,3], columns=['Unbalanced F'])
# xp   = pd.DataFrame(data[:,4], columns=['ball xpos'])
# yp   = pd.DataFrame(data[:,5], columns=['ball ypos'])
# rp   = pd.DataFrame(data[:,6], columns=['ball radii'])


# theta = np.linspace(0.0, 2.0 * np.pi ,150)
# fig, ax = plt.subplots(figsize=(3.0,3.0))

# for i in range (0, n):
#     radius = 0.4
#     a = xp.iloc[i,0] + radius * np.cos(theta)
#     b = yp.iloc[i,0] + radius * np.sin(theta)
#     ax.plot(a,b, 'k-')

# ax.axis('equal')
# ax.invert_yaxis()
# plt.show()
# plot_unbalance_force_field('Force_3_unbal')

# filename = 'Force_1'
# get_unbalance_force_field(filename)
# plot_unbalance_force_field(filename)
# #-------------------------------------
# filename = 'Force_2'
# get_unbalance_force_field(filename)
# plot_unbalance_force_field(filename)
# #-------------------------------------
# filename = 'Force_3'
# get_unbalance_force_field(filename)
# plot_unbalance_force_field(filename)
# #-------------------------------------
# filename = 'Force_4'
# get_unbalance_force_field(filename)
# plot_unbalance_force_field(filename)
# #-------------------------------------
# filename = 'Force_5'
# get_unbalance_force_field(filename)
# plot_unbalance_force_field(filename)
# #-------------------------------------
# filename = 'Force_6'
# get_unbalance_force_field(filename)
# plot_unbalance_force_field(filename)
# #-------------------------------------

filename = 'Force_6'
pid = get_max_unbalance_particle(filename)
print(pid)
mark_unbalance_force_field(200,filename)
#-------------------------------------