# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 08:24:05 2022

Guideline in measurement setup
Monte Carlo Adaption of equation 4

@author: schnei19
"""
import numpy as np
import MC_Tools_PTB_V1_7_19NRM02 as mc
import matplotlib.pyplot as plt

draws = 5000  #The number of monte carlo runs
bins= 100  #The number of bins for the histogramm

"""
Assigning of Values and Drawing of Distributions for uncorrelated Quantities.
The values and the distributions are given and explained in the guideline document
"""
c_P = mc.drawValues(Mean = 1.00024, Stddev = 0.00016, Draws = draws, Type = "normal")
d_PS = mc.drawValues(Mean = 3-0.0005, Stddev = 3+0.0005, Draws = draws, Type = "uniform_interval")
#d_PS = mc.drawValues(Mean = 3, Stddev = 0.0005, Draws = draws, Type = "uniform")
I_vref = mc.drawValues(Mean = 300, Stddev = 1.2, Draws = draws, Type = "normal")
R_ph = mc.drawValues(Mean = 0.0100057, Stddev = 0.0000075, Draws = draws, Type = "normal")
T_ref = mc.drawValues(Mean = 2800-10, Stddev = 2800+10, Draws =draws, Type = "uniform_interval")
#T_ref = mc.drawValues(Mean = 2800, Stddev = 10, Draws =draws, Type = "uniform")
T_A = 2856
c_J = mc.drawValues(Mean = 1.00003, Stddev = 0.00003, Draws = draws, Type = "normal")
R_ref = mc.drawValues(Mean = 0.10004, Stddev= 0.0000035, Draws= draws, Type = "normal")
J_0 = 5.9
#m_P = mc.drawValues(Mean = -0.023-0.006, Stddev = -0.023+0.006, Draws = draws, Type = "uniform_interval")
m_P = mc.drawValues(Mean = -0.023, Stddev = 0.006, Draws = draws, Type = "uniform")
mT_S= mc.drawValues(Mean = 0.72-0.15, Stddev = 0.72+0.15, Draws = draws, Type = "normal")
#mI_S =mc.drawValues(Mean = 6.96-0.66, Stddev = 6.96+0.66, Draws= draws, Type = "uniform_interval")
mI_S =mc.drawValues(Mean = 6.96, Stddev = 0.66, Draws= draws, Type = "uniform")

"""
Drawing of the correlated quantites U_Ph and U_Ref
"""
#Assigning the matching format needed for "drawMultiVariate"
U_Ph_dist = [4.5784,0.0095,"t",9]
U_Ref_dist = [0.590962,0.00012,"t",9]
#Assembling the list of correlated quantities
distributionlist= [U_Ph_dist,U_Ref_dist]
#Defining the correlation matrix
correlationmatrix = np.array([[1.0,0.22],[0.22,1.0]])
#Drawing the correlated values
resultdraw = mc.drawMultiVariate(distributionlist, correlationmatrix, Draws = draws)

U_Ph = resultdraw[:][0]
U_ref= resultdraw[:][1]

"""
For this example the correction factor in the equation of s_v in the Good Practice Guide (GDP) is neglected 
(i.e. set to 1 without uncertainty) to keep the example short.
"""

"""
Calculating s_v strictly according to the measurement equation
"""
s_v = ((c_P*U_Ph*d_PS**2)/(I_vref*R_ph))*((T_ref/T_A)**m_P)*((c_J*U_ref)/(J_0*R_ref))**(m_P*mT_S-mI_S)

"""
Evaluating the s_v data
"""
plt.hist(s_v, bins = bins)
output = mc.sumMC(s_v, printOutput=(True))
