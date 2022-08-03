# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 10:25:23 2021

Spectral calibration evaluation V7 date: 19.04.2022

This file illustrates an implementation of monte carlo caluclation for "Guideline on the Minimum Specifictation for the Spectral Photometric Measurement setups"
The measurement model is in analog to equation 8 of section 3.1.1 for sectral responsivity calibration of photometer heads

The measurement model used is:
    s_Ph(lambda)=(J_Ph/J_ref)*(U_mon,ref/U_mon,Ph)*s_ref(lambda)*A*c_wl(lambda)*c_bw(lambda)*c_unif(lambda)*c_dist
    
    Correction factor for polarization is neglected as all detectors in use have non-significant polarization dependency.
    Correction factor for temperature are neglected as all detectors are temperature stabilized.

    For TULIP setup adjustable photocurrent amplifier and voltagemeters are used, replacing photocurrents with the resistance and voltages.

Measurement Uncertainty Calculation for Calibration of spectral irradiance responsivity for (filter-)radiometers (e.g. photometers) with a tunable laser setup.

Calculation is done for a device under test (detector to be calibrated) referencing to one reference detector.
The calculation of the photometric responsivity is done per default.

The input data as TDMS is readable as variable dictionary once imported to Python.
A Excel converted version is available, but not fit for calculation.

Important explanation steps are marked with:
!!!!!!!!!!

The listed non standard-modules are required:
    MC_Tools_PTB_V1_7_19NRM02
    

Following variables are required to set for each measurement:
    path          - path of the measurement data
    referenz      - Identifier of the reference detector
    DUT           - Identifier of the DUT
    referenzpfad  - path to the reference data for trap detectors and amplifiers
    tdmsdatei     - name of the measurement
    
    unc_dist      - uncertainty of the alignment (distance) of the DUT. Can be vary between 3mm and 0.1mm depending on the DUT
    unc_wlcorr    - uncertainty of the wavelength scale. Depends on the measurement equipment used. For the laser spectrum analyzer 0.003 nm is apliccable
    
Requirements: 
    numpy Version 1.19.4
    scipy Version 1.5.4
    matplotlip Version 3.3.2
    pandas Version 1.1.5

@author: schnei19
"""

import numpy as np
import MC_Tools_PTB_V1_7_19NRM02 as mc
from scipy.interpolate import interp1d
import scipy.interpolate as interpol
import matplotlib.pyplot as plt
from datetime import datetime
import math
import pandas as pd

"""
Returns the current time for monitoring the progress of the script
"""
now = datetime.now()
print('Start =', now.strftime("%H:%M:%S"))

"""
List of variables to be set before calculation is started
"""
draws = 500  # Number of total runs of the MC calculation. 5000 usually is enough for stable results
path = 'P:/Learning_Python/'  # Path of the source file in xlsx format
referenz = 'ENV03B'  # Identifier of the reference detector (the name representation in the source file)
DUT = 'FD18'   # Identifier of the DUT detector (the name representation in the source file)
referenzpfad = 'O:/4-1/4-11/TULIP/Auswertungen/Python Radiometer Auswertung/'   # Path of the reference Excel-Files for amplifiers, detectors and the Vlambda function
xlsxdatei = 'Excel-Import-File.xlsx'  # Name of the Excel file

unc_dist=0.001  # absolute uncertainty of the reference plane of the DUT in m
unc_wlcorr=0.003 # absolute correlated uncertainty of the spectrometer/LSA wavelength scale in nm

saving = False  #Turns saving of result files on/off


"""
!!!!!!!!!!
Start of calculation
Changes here need only to be done to savename or datapaths.

"""
dateiname = xlsxdatei.replace('.xlsx', '')  # Removing the file ending to use filename for savedata
zeitstempel = now.strftime("%d_%m_%Y_%H_%M")
savename = 'sE_'+DUT + '_' + dateiname + ' ' + zeitstempel #Constructing the filename for savedata

# importing the TDMS data file have the complete measurement data stored
dataimport = pd.read_excel(path+xlsxdatei, engine='openpyxl', sheet_name='meas data1')

# Chooses the measurement data section from TDMS file
# More sections consist of configuration data and metadata
measdata = dataimport
"""
!!!!!!!!!!
Import of reference data
The Vlambda function, the reference detector responsivities, the amplifier resistances and the correlation matrices are imported
"""
ndr, traps, vlambda = mc.importref(referenzpfad)
trapcorrENV03B = np.loadtxt(referenzpfad+'traps/corrmatrix-ENV03B.txt')
trapcorrT11 = np.loadtxt(referenzpfad+'traps/corrmatrix-T11.txt')


"""
!!!!!!!!!!
Determining the indices for reference and DUT measurements
Evaluation of the data file to filter the data only for measurements of reference and DUT detector
This function is included here, as for PTB internal use a more extensive wavelength evaluation is included.

"""
# Looking for indices belonging to measurements of reference and DUT detector
def indexdetermination_19NRM02(measdata,referenz,DUT):
    ref_index_list_proto = mc.getIndices(measdata["Identifier"], referenz)
    dut_index_list_proto = mc.getIndices(measdata["Identifier"], DUT)
    # Usually used for importing ad evaluating saved spectra
    # For example reduced to using the wavelengths saved in TDMS files
    lv_wl_list = mc.getValues(
        measdata["LSA_AirCentroidWl"], range(len(measdata["Nr"])))
    bw_list = mc.getValues(measdata["LSA_width"],range(len(measdata["Nr"])))
    Centroid_wl_total_proto = lv_wl_list # adapted to keep variable names as in previous versions

    # Narrowing down the indices lists to correct measurements
    ref_index_list = ref_index_list_proto
    dut_index_list = dut_index_list_proto
    complete_index_list = ref_index_list + dut_index_list
    complete_index_list.sort()
    return ref_index_list,dut_index_list,complete_index_list,Centroid_wl_total_proto,bw_list

ref_index_list,dut_index_list,complete_index_list,Centroid_wl_total_proto,bandwidth_total_proto = indexdetermination_19NRM02(measdata, referenz, DUT)

now = datetime.now()
print('Import done =', now.strftime("%H:%M:%S"))


"""
!!!!!!!!!!
Evaluation of the nominal wavelengths
The results are used for:
    1. summarizing all measurement belonging to the same nominal wavelength
    2. summarizing the values of repeated measurements for each nominal wavelength
    
"""
sollwlextract = measdata['Soll_Wl'] # all nominal wavelength values

sollwlcomplete = sollwlextract[complete_index_list] # the nominal wavelengths for reference and DUT measurements

sollwl = sollwlextract[ref_index_list] # the nominal wavelengths only for reference measurements
"""
Following lists are used for filtering the data in later calculation steps
"""
# creatling list at which indices there are unique nominal wavelengths
extractwl, extractindex, extractcount = np.unique(
    sollwlextract, return_index=True, return_counts=True)
# creating list at which indices there are unique nominal wavelengths for DUT and reference
uniquewl, uniqueindex, uniquecount = np.unique(
    sollwlcomplete, return_index=True, return_counts=True)
# creating list at which indices there are unique nominal wavelengths at the reference detector
refuniquewl, refuniqueindex, refuniquecount = np.unique(
    sollwl, return_index=True, return_counts=True)


"""
!!!!!!!!!!
Reading the measurement data from the data file into separate variables for easier readability

To get the data from the dictionary the keys are required.
This step could be avoided to not overpopulate the variable namespace.
"""
# Getting the used number of measurements (expected to be the same for complete measurement)
nrdgs = mc.getValues(measdata["NRDGS"],ref_index_list)[1]
# Getting signal (deutsch=hell) and dark values and standarddevivations
hell_ref = mc.getValues(measdata["Signal"], ref_index_list)
hell_ref_std = mc.getValues(measdata["SignalStdDev"], ref_index_list)
dark_ref = mc.getValues(measdata["DarkSignal"], ref_index_list)
dark_ref_std = mc.getValues(measdata["DarkSignalStdDev"], ref_index_list)
"""
Extracting the monitor detector values for reference detector
"""
monitorhell_ref = mc.getValues(measdata["Monitor"], ref_index_list)
monitorhell_ref_std = mc.getValues(measdata["MonitorStdDev"], ref_index_list)

"""
Extracting signal (Hell) and dark with standarddeviation for DUT detector
"""
hell_dut = mc.getValues(measdata["Signal"], dut_index_list)
dark_dut = mc.getValues(measdata["DarkSignal"], dut_index_list)
hell_dut_std = mc.getValues(measdata["SignalStdDev"], dut_index_list)
dark_dut_std = mc.getValues(measdata["DarkSignalStdDev"], dut_index_list)
"""
Extracting the monitor detector values for DUT detector
"""
monitorhell_dut = mc.getValues(measdata["Monitor"], dut_index_list)
monitorhell_dut_std = mc.getValues(measdata["MonitorStdDev"], dut_index_list)

"""
Extracting the bandwidths measured with the laser spectrum analyzer (LSA)
"""
bandwidths = mc.getValues(measdata["LSA_width"], ref_index_list)

"""
Extracting the gain setting values required for calculating the resistances of the photocurrent amplifiers
"""
# Getting the identfiers of the amplifieres for reference and DUT
ndr_ref_ident = mc.getValues(measdata["NDR_Identity"], ref_index_list)[1]
ndr_dut_ident = mc.getValues(measdata["NDR_Identity"], dut_index_list)[1]

# Array of gain settings for reference
ndr_ref_gain = np.array(mc.getValues(measdata["NDR_Gain"], ref_index_list), dtype=int)
# Array of gain settings for DUT
ndr_dut_gain = np.array(mc.getValues(measdata["NDR_Gain"], dut_index_list), dtype=int)


"""
!!!!!!!!!!
All data is read into new variables.
Starting here random values are generated.

Generating random values for the resistances of the photocurrentamplifiers

"""
if ndr_ref_ident != "0":  # Checking if the identifier for reference amplifier is set correctly
    # Choosing the matching sheet of imported excel data
    ndrref = ndr[str(ndr_ref_ident)].values
    # Creating an empty array for saving values and uncertainties
    rrefvalues = np.zeros([len(ndr_ref_gain), 2])
    for i in range(len(ndr_ref_gain)):
        # Fills the array with the values and uncertainties used for each measurement
        rrefvalues[i, 0] = ndrref[ndr_ref_gain[i]-1][1]
        rrefvalues[i, 1] = ndrref[ndr_ref_gain[i]-1][2]  
    # Generating the random values with an additional relative uncertainty
    r_ref = mc.drawValues(Mean=rrefvalues[:, 0], Stddev=rrefvalues[:, 1]+rrefvalues[:, 0]*1*10**-4, Draws=draws)

if ndr_dut_ident != "0":  # Checking if the identifier for DUT amplifier is set correctly
    # Choosing the matching sheet of imported excel data
    ndrdut = ndr[str(ndr_dut_ident)].values
    # Creating an empty array for saving values and uncertainties
    rdutvalues = np.zeros([len(ndr_dut_gain), 2])
    for i in range(len(ndr_dut_gain)):
        # Fills the array with the values and uncertainties used for each measurement
        rdutvalues[i, 0] = ndrdut[ndr_dut_gain[i]-1][1]
        rdutvalues[i, 1] = ndrdut[ndr_dut_gain[i]-1][2]
    # Generating the random values with an additional relative uncertainty
    r_dut = mc.drawValues(Mean=rdutvalues[:, 0], Stddev=rdutvalues[:, 1]+rdutvalues[:, 0]*1*10**-4, Draws=draws)
    
"""
!!!!!!!!!!
Generating random values for signal and dark measurements for reference, DUT and monitor detector

As monitor detector is read out in parallel to reference and DUT a monitor correction can be applied.
The voltages are correlated. With enough sample data, the spectrally changing correlation coefficient can be applied.

For this example a single correlation coefficient is chosen, to show the principle of generating correlated numbers.
The pearson correlation coefficient of all the voltages is 0.87 

"""
# Creating empty list to fill with correlated values
hell_ref_list = []
mon_ref_list = []
hell_dut_list = []
mon_dut_list = []
# For all signal measurements the corresponding values are generated
for i in range(len(hell_ref)):
    # Definition of the correlation matrix
    correlation = np.array([[1.0,0.87],[0.87,1.0]])
    # Assembling the list entry with "Mean, Standard Deviation of the Mean, Distribution Identifier, Degrees of Freedom"
    # 20 Measurements are done, therefore Student-T-Distribution with 19 Degrees of Freedom is chosen.
    # The lists are required by the "drawMultiVariate" for value generation.
    ref = [hell_ref[i],hell_ref_std[i]/np.sqrt(nrdgs),'t',nrdgs-1]
    mon_ref = [monitorhell_ref[i],monitorhell_ref_std[i]/np.sqrt(nrdgs),'t',nrdgs-1]
    reflist = [ref,mon_ref]
    dut = [hell_dut[i],hell_dut_std[i]/np.sqrt(nrdgs),'t',nrdgs-1]
    mon_dut = [monitorhell_dut[i],monitorhell_dut_std[i]/np.sqrt(nrdgs),'t',nrdgs-1]
    dutlist = [dut,mon_dut]
    # Multivariate Normal Values are generated
    refpull = mc.drawMultiVariate(reflist, correlation, Draws = draws)
    # Generated values are added to the previously created lists
    hell_ref_list.append(refpull[0,:])
    mon_ref_list.append(refpull[1,:])
    dutpull = mc.drawMultiVariate(dutlist, correlation, Draws = draws)
    hell_dut_list.append(dutpull[0,:])
    mon_dut_list.append(dutpull[1,:])
# The lists are converted to numpy arrays for use in the calculation    
hell_ref_mc = np.array(hell_ref_list)
hell_dut_mc = np.array(hell_dut_list)
ref_mon_mc = np.array(mon_ref_list)
dut_mon_mc = np.array(mon_dut_list)

# The dark readings from reference and DUT are generated uncorrelated
dark_ref_mc = mc.drawValues(Mean=dark_ref, Stddev=dark_ref_std/np.sqrt(nrdgs),
                            Draws=draws, DoF=measdata["NRDGS"][ref_index_list[1]]-1, Type="T")

dark_dut_mc = mc.drawValues(Mean=dark_dut, Stddev=dark_dut_std/np.sqrt(nrdgs),
                            Draws=draws, DoF=measdata["NRDGS"][dut_index_list[1]]-1, Type="T")

"""
!!!!!!!!!!
Start of the calculation

The measurement model is the following:
    s_E = (U_dut*U_Mon_ref / U_ref*U_Mon_dut )* (r_ref / r_dut) * s_ref * A_ref * correctionfactors

The evaluation of the first part: the voltage values and resistances do not depend on wavelength.
Because of this, this first part of the model equation is seperated from the rest of the calculation.

The previously drawn random values in numpy-array format can just be multiplied.
"""
signal_ref_mc = ((hell_ref_mc - dark_ref_mc) /
                 ref_mon_mc)  # Dark and monitor correction for reference

if ndr_ref_ident != "0":  # Check if photocurrent amplifier was used
    i_ref_mc = signal_ref_mc / (r_ref*1000000)  # Calculating the photocurrent from the voltages. 1000000 is used to convert the measured resistances from microOhm to Ohm
else:
    i_ref_mc = signal_ref_mc  # If no amplifier was used the signal is not changed


signal_dut_mc = (hell_dut_mc - dark_dut_mc) / \
    dut_mon_mc  # Dark and monitor correction for DUT

if ndr_dut_ident != "0":  # Check if photocurrent amplifier was used
    i_dut_mc = signal_dut_mc / (r_dut*1000000)  # Calculating the photocurrent from the voltages
else:
    i_dut_mc = signal_dut_mc  # If no amplifier was used the signal is not changed

# Calculation of the photocurrent ratio
# Due to changed connectors of some photodiodes, negative photocurrents can appear
ratio_mc = abs((i_dut_mc) / (i_ref_mc))

"""
!!!!!!!!!!
Calculation of wavelength and bandwidth values and their standard deviations and generating according random numbers.
"""
# Creating empty arrays and lists for wavelength values
wl_mean = np.zeros(len(sollwlextract))
bw_mean = np.zeros(len(sollwlextract))
wl_std = np.zeros(len(sollwlextract))
bw_std = np.zeros(len(sollwlextract))
wllist = []
# loop iterated over all nominal wavelengths (extractcount[0] number of the same nominal wavelength)
for i in range(0, len(Centroid_wl_total_proto), extractcount[0]):
    meanwl = []  # empty list for measured wavelengths belonging to same nominal wavelength
    meanbw = []
    for k in range(extractcount[0]):
        if Centroid_wl_total_proto[i+k] > 0.1: # Check for valid wavelength data
            if np.isnan(Centroid_wl_total_proto[i+k])==False: # Check for NaN Values
                meanwl.append(Centroid_wl_total_proto[i+k])
                meanbw.append(bandwidth_total_proto[i+k])
    if np.mean(meanwl) > 0.1:
        mittelwert = np.mean(meanwl)
        mittelwertbw = np.mean(meanbw)
        # Calculation of mean and standarddeviation
        std = np.std(meanwl, ddof=1)
        wl_mean[i] = mittelwert  # Entering mean values to the array
        wl_std[i] = std  # Entering standard deviation values to the array
        bw_mean[i] = mittelwertbw  # Entering mean bandwidth values to the array
        bw_std[i] = mittelwertbw/4  # Using 1/4 of bandwidths as standard deviation, as the approximation used for correction is for strictly triangular bandpassses
        for l in range(refuniquecount[0]):
            # Assembling the data for generating multivariate values
            # Adding the general correlated wavelength uncertainty of the spectrometer/LSA
            wllist.append([mittelwert, std+unc_wlcorr, "n"]) 

#Removing 0 entries from the standard deviation arrays that can appear
wl_std_reduce = np.delete(wl_std, np.where(wl_std < 0.000001))
bw_std_reduce = np.delete(bw_std, np.where(bw_mean < 0.000001))
            
wlcorr = np.eye(len(wllist))  # Unity matrix as base for correlation matrix
wlcorr[wlcorr==0] = unc_wlcorr/(np.mean(wl_std_reduce)+unc_wlcorr) # Estimating the correlation of the wavelength be calculating the part of the uncertainty, that is assumed to be correlated.
wldraws = mc.drawMultiVariate(wllist, wlcorr, Draws=draws) # Drawing the correlated random values

# Removing 0 entries from the mean arrays that can appear
wl_mean_reduce = np.delete(wl_mean, np.where(wl_mean < 0.000001))
bw_mean_reduce = np.delete(bw_mean, np.where(bw_mean < 0.000001))

# Finding the indices sorting wavelengths
sortarray = wl_mean_reduce.argsort()
wl_mean_sorted = wl_mean_reduce[sortarray]  # Sorting mean wavelength values
wl_std_sorted = wl_std_reduce[sortarray] # Sorting standard deviation wavelength values
bw_mean_sorted = bw_mean_reduce[sortarray]   # Sorting the bandwidth values according to wavelength
bw_std_sorted = bw_std_reduce[sortarray]  # Sorting the standard deviation bandwidth values according to wavelength

"""
!!!!!!!!!!
Looking up reference values, generating accodring random numbers and interpolation

The imported excel data is a source for generating responsivity values.
For correlation two imported matrices are available assuming uniform spectral correlation of 0.65 as no detailed correlation data is available.
"""
ref_wl = np.array(traps[referenz]["wl"])  # Gets the reference values from imported excel dictionary
ref_s = np.array(traps[referenz]["s"])
ref_su = np.array(traps[referenz]["s.u"])

refdistlist = []  # Creating an empty list
for i in range(len(ref_s)):
        # Appends the distribution for multivariate random number generation
    refdistlist.append([ref_s[i], ref_su[i], "n"])

#Check for the reference identifier to use the matching imported correlation matrix
if referenz=='T11':
    refcorr=trapcorrT11
if referenz=='ENV03B':
    refcorr=trapcorrENV03B
refvalues = mc.drawMultiVariate(refdistlist, refcorr, Draws=draws)  # Generating correlated random numbers

refint = []  # Creating an empty list for interpolations
for i in range(draws):
        # Interpolating the generated responsivity values over the nominal wavelength grid
    refint.append(interp1d(ref_wl,refvalues[:,i],kind = 'linear', fill_value='extrapolate'))
        
    # Empty array for the responsivity values at the measured wavelengths
srefwl = np.zeros_like(wldraws)
for i in range(len(wldraws)):
    for j in range(draws):
        # Calculating the responsivity for all wavelengths values
        srefwl[i, j] = refint[j](wldraws[i, j])


wlintegrate = range(360, 960, 1)  # Range of integration for standard luminous responsivity calculation for which interpolation is needed 

    # Empty array for responsivity values at 1 nm intervall
srefwl_int = np.zeros((len(wlintegrate), draws))
for i in range(len(wlintegrate)):
    for j in range(draws):
        # Calculating the responsivites for all 1 nm steps
        srefwl_int[i, j] = refint[j](wlintegrate[i])
# Interpolating the 1 nm steps responsivity values to calculated the derivate (required for correction factors)
ref_int_uk = interpol.InterpolatedUnivariateSpline(ref_wl, np.mean(refvalues, axis=1), ext=0)
ref_Diff2 = ref_int_uk.derivative(n=2)

"""
Generating random numbers for aperture size, uniformity and distance correction factor
"""
# Generating values for the aperture of the reference detectors
T11aperture = mc.drawValues(
    Mean=43.956, Stddev=0.0085, Draws=draws)*0.000001
ENV03aperture = mc.drawValues(
    Mean=15.706, Stddev=0.005, Draws=draws)*0.000001
# Generating correction values for the non-uniformity of the irradiance field
uniformityfield = mc.drawValues(
    Mean=0.99995, Stddev=1.00005, Draws=draws, Type='uniform_interval')
# Generating correction values for the alignment of the detectors at 1.4m measuring distance to the source
distance_uniform = mc.drawValues(Mean=1-math.sqrt((0.0001/1.4)**2+(unc_dist/1.4)**2)*math.sqrt(3), Stddev=1+math.sqrt(
    (2*0.0001/1.4)**2+(2*unc_dist/1.4)**2)*math.sqrt(3), Draws=draws, Type='uniform_interval')
distance = mc.drawValues(Mean=1, Stddev=math.sqrt(
    (0.0001/1.4)**2+(unc_dist/1.4)**2), Draws=draws, Type='normal')

"""
!!!!!!!!!!
Calculating the irradiance responsivity with 2 basic correction factors
The aperture needs to be changes, when other reference detector is used for calculation
"""
s_E_dut = ratio_mc * srefwl * ENV03aperture * distance * uniformityfield

"""
Calculating the means for each quantity to calculate the single MU contributions.
For MC uncertainties, a components analysis requires more steps.
The MC can be done once for each quantity, keeping all other quantities at their respective means.
This evaluates the effect of the uncertainty of one input quantity on the output quantity.
"""
sig_dut_mean=mc.singlevalue_spectral(draws,signal_dut_mc)
sig_ref_mean=mc.singlevalue_spectral(draws,signal_ref_mc)
r_dut_mean=mc.singlevalue_spectral(draws,r_dut)
r_ref_mean=mc.singlevalue_spectral(draws,r_ref)
sref_mean=mc.singlevalue_spectral(draws,srefwl)
blende_mean=np.mean(ENV03aperture)
distance_mean=np.mean(distance)
unif_mean=np.mean(uniformityfield)

"""
Calculation of the MC for each quantity
"""
s_E_stab = (abs((signal_dut_mc / (r_dut_mean*1000000)) / (signal_ref_mc / (r_ref_mean*1000000)))) * sref_mean * blende_mean * distance_mean * unif_mean

s_E_res =  (abs((sig_dut_mean / (r_dut*1000000)) / (sig_ref_mean / (r_ref*1000000)))) * sref_mean * blende_mean * distance_mean * unif_mean

s_E_sref = (abs((sig_dut_mean / (r_dut_mean*1000000)) / (sig_ref_mean / (r_ref_mean*1000000)))) * srefwl * blende_mean * distance_mean * unif_mean

s_E_aperture = (abs((sig_dut_mean / (r_dut_mean*1000000)) / (sig_ref_mean / (r_ref_mean*1000000)))) * sref_mean * ENV03aperture * distance_mean * unif_mean

s_E_distance = (abs((sig_dut_mean / (r_dut_mean*1000000)) / (sig_ref_mean / (r_ref_mean*1000000)))) * sref_mean * blende_mean * distance * unif_mean

s_E_unif = (abs((sig_dut_mean / (r_dut_mean*1000000)) / (sig_ref_mean / (r_ref_mean*1000000)))) * sref_mean * blende_mean * distance_mean * uniformityfield

"""
!!!!!!!!!!
Summarizing the preivously calculated responsivities for each nominal wavelength
Within the measurement, there are 3 repetitions per nominal wavelength.
The summary of the values for each nominal wavelength is done now.
"""
wl_meanall, wl_stdall = mc.sollwl_summary(refuniqueindex, refuniquecount, wldraws)
s_E_meanall, s_E_stdall = mc.sollwl_summary(refuniqueindex, refuniquecount, s_E_dut)


s_E_stab_mean, s_E_stab_std = mc.sollwl_summary(refuniqueindex, refuniquecount, s_E_stab)  
s_E_res_mean, s_E_res_std = mc.sollwl_summary(refuniqueindex, refuniquecount, s_E_res)  
s_E_sref_mean, s_E_sref_std = mc.sollwl_summary(refuniqueindex, refuniquecount, s_E_sref)  
s_E_aperture_mean, s_E_aperture_std = mc.sollwl_summary(refuniqueindex, refuniquecount, s_E_aperture)  
s_E_distance_mean, s_E_distance_std = mc.sollwl_summary(refuniqueindex, refuniquecount, s_E_distance) 
s_E_unif_mean, s_E_unif_std = mc.sollwl_summary(refuniqueindex, refuniquecount, s_E_unif) 


"""
!!!!!!!!!!
Calculating correction factors for bandwidth and wavelength as given in
"UNCERTAINTY ANALYSIS OF A PHOTOMETER CALIBRATION AT THE DSR SETUP OF THE PTB"
by Winter and Sperling in "PROCEEDINGS of the 2nd CIE Expert Symposium on Measurement Uncertainty"

The previously calculated responsivity of the DUT needs to be interpolated and the derivates need to be defined.
This is used in wavelength and bandwidth correction
"""
# Interpolating the calculated responsivities for further corrections
s_E_int_uk = interpol.InterpolatedUnivariateSpline(wl_mean_sorted, np.mean(s_E_meanall, axis=1), ext=0) 
# First derivative for wavelength correction factor
s_E_diff1 = s_E_int_uk.derivative(n=1)
# Second derivative for bandwidth correction factor
s_E_diff2 = s_E_int_uk.derivative(n=2)



# Generating empty lists and arrray for wavelength and bandwidth correction
cbw_list = []
ubw_list = []
cwl_list = []
uwl_list = []
cbw = np.zeros((len(uniquewl), draws))
uwl_list_corr = []

# Loop over all wavelengths
for i in range(len(bw_mean_sorted)):
    # Calculating the factor for numerator of bandwidth correction
    bw_faktor_zähler = (s_E_diff2(wl_mean_sorted[i])/s_E_int_uk(wl_mean_sorted[i]))
    # Calculating the factor for denominator of bandwidth correction
    bw_faktor_nenner = (ref_Diff2(wl_mean_sorted[i])/ref_int_uk(wl_mean_sorted[i]))
    # Generating random bandwidth values from the measured bandwidths
    bw_mc = mc.drawValues(bw_mean_sorted[i]*10**-3, bw_std_sorted[i]*10**-3, Draws=draws)
    vorfaktor_cbw = (1/12)*((bw_mc)**2)  #10**-3 to correct the bandwidths given in pm to nm
     # Calculating the numerator of bandwidth correction
    zähler_cbw = 1-vorfaktor_cbw*bw_faktor_zähler
    # Calculating the denominator of bandwidth correction
    nenner_cbw = 1-vorfaktor_cbw*bw_faktor_nenner
    # Calculating the bandwidth correction factors
    cbw_list=zähler_cbw/nenner_cbw
    cbw_array = np.array(cbw_list)
    cbw[i,:]=cbw_array
    # Calculating the factor for wavelength correction
    wl_faktor = (s_E_diff1(wl_mean_sorted[i])/s_E_int_uk(wl_mean_sorted[i]))
    wl_vorfaktor = wl_std_sorted[i]
    # Appending the lists for generating wavelength correction random values
    uwl_list.append(abs(wl_vorfaktor*wl_faktor))
    cwl_list.append(1)
# Converting the lists to arrays
cwl_array = np.array(cwl_list)
uwl_array = np.array(uwl_list)
# Defining variable and generating wavelength correction 
bw_korr = cbw
wl_korr = mc.drawValues(cwl_array, uwl_array, Draws=draws)

"""
!!!!!!!!!!
Calculating the irradiance responsivity with the additional corrections.

For component analysis all other responsivities are also multiplied with the new corrections

"""
# Calculating the irradiance responsivity with bandwidth and wavelength correction
s_E_mean_korr = s_E_meanall*bw_korr*wl_korr

# Calculating the mean values for MU component analysis
wl_korr_mean=mc.singlevalue_spectral(draws,wl_korr)
bw_korr_mean=mc.singlevalue_spectral(draws,bw_korr)
s_E_mean_mean = mc.singlevalue_spectral(draws,s_E_mean_korr)

s_stab_korr = s_E_stab_mean*bw_korr_mean*wl_korr_mean
s_res_korr = s_E_res_mean*bw_korr_mean*wl_korr_mean
s_sref_korr = s_E_sref_mean*bw_korr_mean*wl_korr_mean
s_aperture_korr = s_E_aperture_mean*bw_korr_mean*wl_korr_mean
s_distance_korr = s_E_distance_mean*bw_korr_mean*wl_korr_mean
s_unif_korr = s_E_unif_mean*bw_korr_mean*wl_korr_mean
s_bw_korr = s_E_mean_mean*bw_korr*wl_korr_mean
s_wl_korr = s_E_mean_mean*bw_korr_mean*wl_korr

"""
!!!!!!!!!!
Interpolating the resulting corrected irradiances to calculate the values for the nominal wavelengths
"""
s_E_int = []
for i in range(draws):
    # interpolation of the irradiances with the measured wavelengths as a grid
    s_E_int.append(interpol.InterpolatedUnivariateSpline(wl_mean_sorted, s_E_mean_korr[:, i], ext=0))

# Creating an array with the nominal wavelength irradiance responsivities
s_E_result = np.zeros((len(uniquewl), draws))
for i in range(len(s_E_result)):
    for j in range(draws):
        s_E_result[i, j] = s_E_int[j](extractwl[i])

# Creating an array with the irradiance responsivities with 1 nm interval for luminous responsivity calculation
s_v_E_result = np.zeros((len(wlintegrate), draws))
for i in range(len(s_v_E_result)):
    for j in range(draws):
        s_v_E_result[i, j] = s_E_int[j](wlintegrate[i])

"""
Calculation of Plancks spectrum for luminous responsivity
"""
spektrum1 = np.array(wlintegrate)
wlrange2 = range(360, 831, 1) # To cover photometric range
spektrum2 = np.array(wlrange2)
pl1 = []
plvl = []

for i in range(len(spektrum1)):
    pl1.append(mc.Planck(wavelength=spektrum1[i]))
planck1 = np.array(pl1)

for i in range(len(spektrum2)):
    plvl.append(mc.Planck(wavelength=spektrum2[i]))
planckvl = np.array(plvl)
KM = 1/683

"""
!!!!!!!!!!
Luminous responsivity calculation for the DUT
As the complete MC values are used, the correlations are automatically included
"""
nenner = np.sum(planckvl*vlambda)
sv = []
for i in range(draws):
    zaehler = np.sum(planck1*s_v_E_result[:, i])
    sv.append(KM*zaehler/nenner)

now = datetime.now()
print('MC runs finished =', now.strftime("%H:%M:%S"))

"""
Evaluation of the MC results for component analysis and plotting
"""
# Output for irradiance responsivity
mc_output1 = mc.mc_analysis(s_E_result,uniquewl)

# Output for irradiance responsivity with 1 nm steps
mc_output2 = mc.mc_analysis(s_v_E_result,spektrum1)

# Monte Carlo results for luminous responsivity
sv_mc = mc.sumMC(sv)
sv_mean = sv_mc[0][0]
sv_unc = sv_mc[0][1]
sv_rel_unc = sv_unc/sv_mean

# Monte Carlo results for the component analysis
uncstab = mc.mc_analysis(s_stab_korr,uniquewl)
uncres = mc.mc_analysis(s_res_korr,uniquewl)
uncref = mc.mc_analysis(s_sref_korr,uniquewl)
uncaperture = mc.mc_analysis(s_aperture_korr,uniquewl)
uncdist = mc.mc_analysis(s_distance_korr,uniquewl)
uncunif = mc.mc_analysis(s_unif_korr,uniquewl)
uncbw = mc.mc_analysis(s_bw_korr,uniquewl)
uncwl = mc.mc_analysis(s_wl_korr,uniquewl)

"""
Evaluation of occuring correlatons
"""
corr1 = mc.correlation(s_E_result)
mc.corrPlotPTB(corr1)

"""
plotting the results
"""
fig, ax = plt.subplots()
ax.plot(mc_output1[:, 0], mc_output1[:, 1],
        linestyle="solid", marker="None", color="blue")
ax.set_xlabel("wavelength / nm")
ax.set_ylabel("s_E")
ax2 = ax.twinx()
ax2.plot(mc_output1[:, 0], mc_output1[:, 2]/mc_output1[:, 1],
         linestyle="solid", marker="None", color="red", label="total unc.")

ax2.plot(uncstab[:, 0], uncstab[:, 2]/uncstab[:, 1],
         linestyle="dashed", marker="None", color="green", label = "stability")

ax2.plot(uncres[:, 0], uncres[:, 2]/uncres[:, 1],
         linestyle="dashed", marker="None", color="orange", label = "resistance")

ax2.plot(uncref[:, 0], uncref[:, 2]/uncref[:, 1],
         linestyle="dashed", marker="None", color="black", label = "reference")

ax2.plot(uncaperture[:, 0], uncaperture[:, 2]/uncaperture[:, 1],
         linestyle="dashed", marker="None", color="cyan", label = "aperture")

ax2.plot(uncdist[:, 0], uncdist[:, 2]/uncdist[:, 1],
         linestyle="dashed", marker="None", color="yellow", label = "alignment")

ax2.plot(uncunif[:, 0], uncunif[:, 2]/uncunif[:, 1],
         linestyle="dashed", marker="None", color="gray", label = "uniformity")

ax2.plot(uncbw[:, 0], uncbw[:, 2]/uncbw[:, 1],
         linestyle="dashed", marker="None", color="magenta", label = "bandwidth")

ax2.plot(uncwl[:, 0], uncwl[:, 2]/uncwl[:, 1],
         linestyle="dashed", marker="None", color="brown", label = "wavelength")
ax2.set_ylabel("u_rel(s_E)")
ax2.legend(loc="upper left",bbox_to_anchor=(1.15, 1) )
ax2.set_yscale("log")
plt.show()

fig, ax = plt.subplots()
ax.plot(mc_output2[:, 0], mc_output2[:, 1],
        linestyle="None", marker="o", color="blue")
ax.set_xlabel("wavelength / nm")
ax.set_ylabel("s_E")
ax2 = ax.twinx()
ax2.plot(mc_output2[:, 0], mc_output2[:, 2],
         linestyle="None", marker="o", color="red")
ax2.set_ylabel("u(s_E)")
ax2.set_yscale("log")
plt.show()

"""
!!!!!!!!!!
Saving the results for textfile
Here the savenames can be changed if required.

The saved data is:
    luminous responsivity and uncertainty
    irradiance responsivity at nominal wavelengths in 5 nm steps
    the MC run results for irradiance responsivity in 5 nm steps
    irradiance responsivity at nominal wavelengths in 1nm steps
"""

svarray=[sv_mean,sv_unc,sv_rel_unc]
if saving==True:
    # Saving the luminous responsivity results
    np.savetxt(path+savename+'_ref_'+referenz+'_sv.txt', svarray)
    # Saving the irradiance responsivity results in 5nm steps
    np.savetxt(path+savename+'_ref_'+referenz+'_5nm.txt', mc_output1)
    # Saving all MC values for irradiance responsivity in 5nm steps
    np.savetxt(path+savename+'_ref_'+referenz+'_5nm_MCruns.txt', s_E_result)
    # Saving the irradiance responsivity results in 1nm steps
    np.savetxt(path+savename+'_ref_'+referenz+'_1nm.txt', mc_output2)
