# -*- coding: utf-8 -*-
"""
MC_Tools_PTB V1.7 19NRM02
Created on Mon Feb  1 12:07:30 2021

MC_Tools_PTB summarizes functions from Scipy and Numpy to new functions useful for monte carlo simulations.
An additional function for summarizing the output of a monte carlo simulation for a single output quantity is given.

V 1.0 PTB internal version with specific functions for data handling in TDMS and evaluating LSA spectra
V 1.1 PTB internal version with full english commentary
V 1.2 (NMISA) Removed PTB specific data handling and spectral evaluation
V 1.3 PTB internal version with improved correlation plot
V 1.4 PTB internal version with uniformity approach
V 1.5 PTB internal version with removed uniformity
V 1.6 added Student-T to Multivariate Draws
V 1.7 added convenience functions for MU analysis and TDMS import functionality
---------------------------------
Overview:

    -drawValues         Draws values from a given distribution returning a numpy array of values
    -drawFromArray      Draws values from an array of measurement data according to a Student-T-distribution returning a numpy array of values
    -sumMC              Summarizes the monte carlo draws for a single quantity
    -drawMultiVariate   Draws values from a multivariate distribution taking a correlation matrix into account
    -correlation        Calculates the correlation matrix
    -corrPlotPTB        Improved correlations matrix plot
    -getIndices         Extracs a list of indices given by a condition
    -getValues          Results an array from a part of a list, given by a list of indices
    -LSAspectra         Imports LSA spectra and calculates the centroid wavelengths
    -Planck             Calculates the spectral radiance based on wavelength and temperature
    -uniformity-field   Calculates the field uniformity for an identifier
    -importref          Imports the reference datasets for transimpedance amplifiers and reference detectors
    -singlevalue_spectral 
                        Summary of the MC values of a quantity
    -sollwl_summary     Summary of the values belonging to the same nominal wavelength
    -mc_analysis        Summary of the sumMC outputs for a spectral quantity
    -clock              stopwatch routine
    -read_tdms          Routine for importing TDMS data

---------------------------------


@author: schnei19
"""


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import glob,os,math,datetime,time
import pandas as pd
from nptdms import TdmsFile as tdms



def drawValues(Mean, Stddev, Draws = 1000, DoF = 1, Type = "normal"):
    """
    Generates Draws number of values from several usually used distribution types.
    As Stddev the absolute standard measurement uncertainty has to be given.

    If called with Numpy-Arrays of Means and StdDevs a Matrix is returned.

    Example:   drawValues(Mean=1, Stddev=0.5, Size=1000, DoF=1, Type="normal")

    Defaultvalues:
        Size = 1000
        DoF = 1
        Type = "normal" 

    Implemented are these distribution types: "normal", "T", "uniform" and "triangle"
    
    For uniform_interval distributions Mean is used as the lower boundary and Stddev is used as upper boundary.
    For triangle_interval distribution Mean is used as the lower boundary, Stddev as the upper boundary and DoF [0..1] is used as symmetry factor.
    """

    if type(Mean) != np.ndarray:
        if Type == "normal":
            nxi = stats.norm.rvs(loc=Mean, scale=Stddev, size=Draws)
            return(nxi)
        if Type == "T":
            txi = stats.t.rvs(loc=Mean, scale=Stddev, df=DoF, size=Draws)
            return(txi)
        if Type == "uniform_interval":
            uxi = stats.uniform.rvs(loc=Mean, scale=(Stddev-Mean), size=Draws)
            return(uxi)
        if Type == "triangle_interval":
            trxi = stats.triang.rvs(loc=Mean, scale=Stddev, c=DoF, size=Draws)
            return(trxi)
        if Type == "uniform":
            uxi = stats.uniform.rvs(loc=Mean-Stddev*math.sqrt(3), scale = 2*Stddev*math.sqrt(3), size=Draws)
            return(uxi)
        if Type == "triangle":
            txi = stats.triang.rvs(loc=Mean-Stddev*math.sqrt(6), scale=2*Stddev*math.sqrt(6), size=Draws, c=DoF)
    else:
        result = np.zeros([len(Mean), Draws])
        for i in range(len(Mean)):
            if Type == "normal":
                result[i] = stats.norm.rvs(loc=Mean[i], scale=Stddev[i], size=Draws)
            if Type == "T":
                result[i] = stats.t.rvs(loc=Mean[i], scale=Stddev[i], df=DoF, size=Draws)
            if Type == "uniform_interval":
                result[i] = stats.uniform.rvs(loc=Mean[i], scale=(Stddev[i]-Mean[i]), size=Draws)
            if Type == "triangle_interval":
                result[i] = stats.triang.rvs(loc=Mean[i], scale=Stddev[i], c=DoF, size=Draws)
            if Type == "uniform":
                result[i] = stats.uniform.rvs(loc=Mean-Stddev*math.sqrt(3), scale = 2*Stddev*math.sqrt(3), size=Draws)    
            if Type == "triangle":
                result[i] = stats.triang.rvs(loc=Mean-Stddev*math.sqrt(6), scale=2*Stddev*math.sqrt(6), size=Draws, c=DoF)
        return(result)


def drawFromArray(List, Draws):
    """
    Draws values from an array representing a Student-T-distribution
    Can be utilized to draw values from a sample of measurement results to generate values for montecarlo propragation.

    Example:    drawFromArray(List = array[1,2,3,4,3,5,2,3,6,7,0], Draws = 1000)

    List is needed to be a numpy.array of values
    """
    mean = np.mean(List)
    std = np.std(List)
    dof = len(List)
    txi = drawValues(Mean=mean, Stddev=std, Draws=Draws, DoF=dof-1, Type="T")
    return(txi)


def sumMC(InputValues, Coverage=0.95, printOutput=False):
    """
    Based on InputValues for one quantity and the given Coverage the measurement uncertainty based on montecarlo results is calculated.

    Output is returned as: [[Mean, absolute Standarduncertainty],[lower coverage boundary, upper coverage boundary]]

    Example:    sumMC([Numpy array], Coverage = 0.99)

    Defaultvalue:
        Coverage = 0.95 (k=2 for normal distributions)
    """
    # Sorting of the input values
    Ys = sorted(InputValues)
    # Calculating the number of draws
    Ylen = len(InputValues)
    # Calculating the number of draws covering the given coverage
    q = int(Ylen * Coverage)
    # Calculating the draw representing the lower coverage intervall boundary
    r = int(0.5 * (Ylen - q))
    # Calculating the mean of the input values
    ymean = np.mean(InputValues)
    # Calculating standard deviation of the input values as absolute standard uncertainty
    yunc = np.std(InputValues)
    # Summarizing mean and uncertainty
    values = [ymean, yunc]
    # Calculating the values of the draws for lower and upper boundary of the coverage intervall
    ylow = Ys[r]
    yhigh = Ys[r + q]
    # Summarizing the coverage intervall
    interval = [ylow, yhigh]
    # Summarizing the total output
    output = [values, interval]
    # Printing the output values
    if printOutput is True:
        print('Mean: ' + str(values[0]))
        print('Standard uncertainty: ' + str(values[1]))
        print(str(Coverage*100) + '% intervall: ' + str(interval))
    # Returns the output values
    return(output)


def getIndices(List, Condition):
    """
    Extracts all indices from the list "List" where "Condition" is True

    Example:  getIndices(['VL1','T11','ENV03B','VL1','T11','ENV03B'], 'VL1')

    Looks through a list of string and results the indices where the string is 'VL1'. Result is [0,3]
    """
    result = [i for i, value in enumerate(List) if value == Condition]
    return(result)


def getValues(List, Indices):
    """
    Extracts the values from List at all indices position

    Example getValues([1,2,3,4,5,6,7,8,9],[1,4])

    Returns [2,5]
    """
    resultlist = [List[i] for i in Indices]
    if isinstance(resultlist[1], str):
        result = resultlist
    else:
        result = np.array(resultlist)
    return(result)


def drawMultiVariate(Distributions, Correlationmatrix, Draws=100):
    """
    Draws values from a multivariate standard distribution according to the given correlation matrix.

    Returns an array with the dimensions (Numer of Distributions, Number of Draws).

    Example:    drawMultiVariate(List[[Mean, total standard uncertainty, type],..], correlation matrix)

    Within the distribution list for type "n" represents standard distribution and "u" represents uniform distribution.
    "t" can be used for drawing from a student t distribution

    As Distributions a list is needed.
    Example for a standard and uniform distribution: Distributions = [[1,0.1,"n"],[5,1,"u"]]
    Example for a Student T distribution: Distributions = [[1,0.1,"t",5],[5,1,"t",10]]

    As Correlationmatrix a positive semidefinite Matrixarray as needed:
    Example for two quantities with correlation rho:   numpy.array([[1.0,rho],[rho,1.0]])
    """
    dimension = len(Correlationmatrix)
    copula = stats.multivariate_normal(np.zeros(dimension), Correlationmatrix)

    z = copula.rvs(Draws)
    x = np.zeros(shape=(dimension, Draws))
    for i in range(dimension):
        xi = stats.norm.cdf(z[:, i])
        if Distributions[i][2] == "n":
            xidist = stats.norm.ppf(xi, loc=Distributions[i][0], scale=Distributions[i][1])
        if Distributions[i][2] == "u":
            xidist = stats.uniform.ppf(xi, loc=Distributions[i][0], scale=Distributions[i][1])
        if Distributions[i][2] == "t":
            xidist = stats.t.ppf(xi, loc=Distributions[i][0], scale=Distributions[i][1], df=Distributions[i][3])
        x[i] = xidist
    return(x)


def correlation(Distributions):
    """
    Calculates a correlation matrix based on the values of the distributions.

    Example:    correlation(Distributions)

    The distributions are needed as a numpy array with each line representing the values of one quantity/distribution

    Example for 3 distributions with 5 values each: numpy.array([[1,1.1,0.9,0.8,1.2],[2,3,4,5,2],[4,3,3,6,1]])
    """
    matrix = np.corrcoef(Distributions)
    return(matrix)



def corrPlotPTB(Matrix):
    """
    Test für besseren Plot der Matrix für TULIP
    
    Example: corrPlotPTB(Correlationmatrix)    
    """
    plt.xlabel('wavelength / nm')
    plt.ylabel('wavelength / nm')
    plt.imshow(Matrix, vmin=-1, vmax=1, cmap="jet", interpolation="nearest", extent=[360, 960 ,960 ,360 ])
    cbar = plt.colorbar(ticks=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    cbar.set_label('correlation coefficient')
    plt.show()


def Planck(wavelength, Temp=2856):
    """
    Calculates the spectral radiance based on the given temperature
    
    wavelength: numpy array with the wavelengths that are to be calculated
    Temp: integer with the Temperature
   
    Returns the value for the requested wavelengths (in nm) and temperature.
    Default Value: Temp = 2856 K
    """
    h = 6.62607015*(10**-34)
    c = 299792458
    k = 1.380649*(10**-23)
    f = c / (wavelength*10**-9)
    factor1 = (2*h*f**3)/(c**2)
    factor2 = math.e**((h*f)/(k*Temp))-1
    result = factor1/factor2
    return(result)


def LSAspectra(Path, upperwavelength=1200.):
    """
    Calculates the Centroid Wavelengths for all LSA spectra within a specified folder.
    Returns a list of tuples containing the Labview Centroid and Calculated Centroid
    Only calculates up to upperwavelength

    Example LSASpectra('P:/Learning_Python/data/spectra/')
    Returns [(192,192.1),(193,193.1),(194,194.1)]
    """
    wavelengths = []   # Empty List for results
    files = glob.glob(Path + '*.txt')   # List all files in directory 'Path' ending on .txt
    files.sort(key=os.path.getmtime)  # sorts the spectra for creation time
    print('Importing spectral data from '+Path+' ...')
    for i, file in enumerate(files):   # For all files
        data = np.loadtxt(file)    # Import the current spectrum
        filename = file[-14:]       # Reducing the filename
        lvcentroid = float(filename[:-4])    # Saving the wavelength within the filename to lvcentroid variable

        wl = data[:, 0]
        signal = data[:, 1]   #C reating arrays with the imported wavelength and signal values
        minwl = wl[0]
        maxwl = wl[-1]
        peakindex = np.argmax(signal)   # Index of maximum Signal value
        peakwl = wl[peakindex]    # Wavelength corresponsing to maximum Signal value
        peaksignal = signal[peakindex]    # Maximum Signal value

        hm = peaksignal/2   # Halfvalue of the maximum Signal

        indices = np.where(signal > hm, wl, 0)   # Part of the wavelengths where Signal is above half value
        reducedwl = []   # Empty array
        for index in indices:
            if index > 0:
                reducedwl.append(index)   # Reducing the array
        fwhm = reducedwl[-1]-reducedwl[0]     # Calculating the full width half maximum FWHM

        intmin = peakwl - 3*fwhm    # Integrationrange from peak to + and - 3 times FWHM
        intmax = peakwl + 3*fwhm

        spectrum = interp1d(wl, signal, fill_value='extrapolate')   # Interpolating the spectra for centroid calculation
        spectrum2 = interp1d(wl, signal*wl, fill_value='extrapolate')

        wertepre = range(int(intmin*10000), int(intmax*10000), 1)
        werte = []
        for k in range(len(wertepre)):
            werte.append(wertepre[k]/10000)   # creating the wavelength values for sum calculation of the centroid

        wertepeaktestpre = range(int(minwl*10000), int(maxwl*10000), 1)
        wertepeaktest = []
        for m in range(len(wertepeaktestpre)):
            wertepeaktest.append(wertepeaktestpre[m]/10000) # creating wavelength values to test if there are sidepeaks

        result = spectrum(werte)
        result2 = spectrum2(werte)

        peaktestresult = spectrum(wertepeaktest)
        peaktestresult2 = spectrum2(wertepeaktest)

        sumresult = sum(result)  # Sum calculation of the peak within +-3 fwhm
        sumresult2 = sum(result2)

        sumpeaktest = sum(peaktestresult) # sum calculation of the peak within the whole spectrum
        sumpeaktest2 = sum(peaktestresult2)

        centroidpeaktest = sumpeaktest2/sumpeaktest
        centroid = sumresult2/sumresult
        peaktest = math.isclose(centroid, centroidpeaktest, abs_tol=1)  # testing if the peak of the whole spectrum agrees with the main peak within 1 nm to check for sidepeaks
        if centroid > upperwavelength:
            peaktest = False
        wavelengths.append((lvcentroid, centroid, peaktest, centroidpeaktest, abs(centroidpeaktest-centroid)))

        #if i % 250 == 0:
        #    print('Importing file %d : wavelength = %.2f'%(i, centroid)) # Prints every 100th wavelength to let the user know the script is working
    print('Done')  #Prints "done" to show the spectral import and calculation is finished
    return(wavelengths)


def importref(referenzpfad):
    """
    Importing referencedata excel-files
    
    Returns:
        ndr: amplifier data as Pandas DataFrame
        traps: trap detector data as Pandas DataFrame
        vlambda: numpy array with the 2° observer Vlambda function in 1nm steps
    """

    # Imports the photocurrent amplifier data from excel file (Excel Sheets named according to amplifier ID)
    ndr = pd.read_excel(referenzpfad + "NDR/NDR.xlsx",
                    sheet_name=None, engine='openpyxl')

    # Import the reference detector data as a dictionary from excel file (Excel Sheets named according to Identifier)
    traps = pd.read_excel(referenzpfad + "Traps/traps.xlsx",
                      sheet_name=None, engine='openpyxl')
    # Imports the data for the Vlambda function
    vlambda = np.array(pd.read_csv(referenzpfad + "CIE_y_data.csv",
                               sep=",", index_col=False, decimal='.', header=None)[1])
    return ndr, traps, vlambda

def singlevalue_spectral(draws,quantity):
    """
    Results calculated means for all values of a quantity and creates array with average values.
    Required for component analysis of MC calculation
    ----------
    draws : int
    quantity : numpy array
    Array with values the size of number of measurements x number of draws 

    Returns
    -------
    result : numpy array
        Array with averaged values for all draws 
    """
    meanvalues=np.mean(quantity, axis=1)
    result = np.array([meanvalues,]*draws).transpose()
    return result

def sollwl_summary(refuniqueindex, refuniquecount, quantity):
    """
    Calculates the mean and standard deviation of values belonging to the same nominal wavelength.
    Returns the values sorted following the indices/wavelengths
    
    refuniqueindex: numpy array of the indices where nominal wavelengths appear the first time for reference detector
    refuniquecount: numpy array with the numbers of measurement per nominal wavelength for reference detector
    quantity: numpy array of the quantity to be summarized. 
    """
    quantitymeanall = []
    quantitystdall = []
    for i in range(len(refuniqueindex)):
        quantitymean = []
        for k in range(refuniquecount[i]):
            quantitymean.append(quantity[refuniqueindex[i]+k])
        mittelwert = np.mean(quantitymean, axis=0)
        std = np.std(quantitymean, axis=0)
        quantitymeanall.append(mittelwert)
        quantitystdall.append(std)
        quantitymean_array = np.array(quantitymeanall)
        quantitystd_array = np.array(quantitystdall)
    return quantitymean_array, quantitystd_array

def mc_analysis(mc_input,wavelengths):
    """
    mc_input : the MC result array as numpy array
    wavelengths : the matching wavelength scale as numpy array
    ----------
    mc_analysis : uses the sumMC function on the MC result for each wavelength.

    Returns
    -------
    output : numpy array
    [wavelength, mean, standard uncertainty] for all wavelengths
    """
    output = np.zeros(shape=(len(mc_input),3))
    for i in range(len(mc_input)):
        analysis = sumMC(mc_input[i])
        output[i,0] = wavelengths[i]
        output[i,1] = analysis[0][0]
        output[i,2] = analysis[0][1]
    return output

def clock(value,verbose=1,praefix='',suffix=''):
    ''' 
    small stop watch function for keeping track of program runtime.
    Author: Riechelmann
    '''
    if value == 0: return time.time()
    if value != 0:
        runtime = time.time() - value
        if verbose == 1:
            if suffix == '': suffix = ' seconds passed'
            print(praefix+' %.3f' % runtime + suffix)
            #print('%.3f' % (runtime/60)+' minutes passed')
        return(time.time())



def read_tdms(tdmsfile,verbose=0):
    '''
    COMPATIBLE with version npTDMS 0.23.0 (25.03.2020)
    opens a TDMS file and returns a dictionary of dictionaries with all data.
    
    Author: Riechelmann
    
    Example
    -------
    >>> data = ptb_tools.read_tdms(filename)
    '''
    starttime = clock(0) # start time for performance check
    tdms_file = tdms.read(tdmsfile) # opens up the TDMS-file with the tdms module
    all_groups = tdms_file.groups() # retrieves the group names contained by the TDMS-file
    tdms_data = {} # creates an empty dictionary
    
    for j in range(len(all_groups)):
        tdms_data[all_groups[j].name] = {}
        group = tdms_file[all_groups[j].name]
        group_channels = group.channels() # gets all group channel names (the data) inside the actual group
        names = [group_channels[ii].path.split("'")[-2] for ii in range(len(group_channels))] # formatting it to a list
        for k in range(len(names)):
            tdms_data[all_groups[j].name][names[k]] = group_channels[k].data #<--- data is added to dictionary
            #print(type(tdms_data[all_groups[j].name][names[k]][0]), group_channels[k].data_type.enum_value)
            if group_channels[k].data_type.enum_value == 68:
                tdms_data[all_groups[j].name][names[k]] = [i.astype(datetime.datetime) for i in tdms_data[all_groups[j].name][names[k]]] # npTDMS version 0.22 compatible
            if group_channels[k].data_type.enum_value == 32:
                try: tdms_data[all_groups[j].name][names[k]] = [i.astype(str).item() for i in tdms_data[all_groups[j].name][names[k]]] # npTDMS version 1.1x compatible
                except AttributeError:
                    tdms_data[all_groups[j].name][names[k]] = [i for i in tdms_data[all_groups[j].name][names[k]]]
    if verbose == 1: print('read_tdms: opened dataset: '+tdmsfile+' (%.3f' % clock(starttime,verbose=0)+' s)')
    return(tdms_data)