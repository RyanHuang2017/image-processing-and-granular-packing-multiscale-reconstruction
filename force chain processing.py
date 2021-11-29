#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 09:34:57 2021

@author: sichuan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utility import force_processing
from utility import local_force_processing
from utility import contact_map_drawing
plt.close("all")

# filename = 'Force_1'
# force_processing(filename)
# #----------------------------------------
# filename = 'Force_2'
# force_processing(filename)
# #----------------------------------------
# filename = 'Force_3'
# force_processing(filename)
# #----------------------------------------
# filename = 'Force_4'
# force_processing(filename)
# #----------------------------------------
# filename = 'Force_5'
# force_processing(filename)
# #----------------------------------------
# filename = 'Force_6'
# force_processing(filename)

# filename = 'Force_1'
# local_force_processing(filename)
# #-------------------------------------------
# filename = 'Force_2'
# local_force_processing(filename)
# #-------------------------------------------
# filename = 'Force_3'
# local_force_processing(filename)
# #-------------------------------------------
# filename = 'Force_4'
# local_force_processing(filename)
# #-------------------------------------------
# filename = 'Force_5'
# local_force_processing(filename)
# #-------------------------------------------
# filename = 'Force_6'
# local_force_processing(filename)
# #-------------------------------------------

filename = 'Force_1'
contact_map_drawing(filename)
filename = 'Force_2'
contact_map_drawing(filename)
filename = 'Force_3'
contact_map_drawing(filename)
filename = 'Force_4'
contact_map_drawing(filename)
filename = 'Force_5'
contact_map_drawing(filename)
filename = 'Force_6'
contact_map_drawing(filename)