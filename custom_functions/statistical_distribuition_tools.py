import pandas as pd
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from custom_functions.freedman_diaconis import freedman_diaconis


def dist_finder(sensor_name):
    data_set = pd.read_csv("labeled_dataset.csv")
    y = data_set[sensor_name]

    # Create an index array (x) for data
    x = np.arange(len(y))
    size = len(y)
    sc=StandardScaler() 
    yy = y.values.reshape(-1,1)
    sc.fit(yy)
    y_std =sc.transform(yy)
    y_std = y_std.flatten()
    y_std
    del yy

    # Set list of distributions to test
    # See https://docs.scipy.org/doc/scipy/reference/stats.html for more

    dist_names = ['beta',
                'f',
                'argus',
                'crystalball',
                'gausshyper',
                'kappa4',
                'kappa3',
                'expon',
                'gamma',
                'lognorm',
                'norm',
                'pearson3',
                'triang',
                'uniform',
                'weibull_min', 
                'weibull_max']

    # Set up empty lists to stroe results
    chi_square = []
    p_values = []

    # Set up 50 bins for chi-square test
    # Observed data will be approximately evenly distrubuted aross all bins
    # percentile_bins = np.linspace(0,100,51)
    # percentile_cutoffs = np.percentile(y_std, percentile_bins)
    # observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
    # cum_observed_frequency = np.cumsum(observed_frequency)

    # Loop through candidate distributions

    for distribution in dist_names:
        # Set up 50 bins for chi-square test
        # Observed data will be approximately evenly distrubuted aross all bins
        min = y.min()
        max = y.max()
        # TODO: Fix percentile bins for chi square
        percentile_bins = np.linspace(min, max, 51)
        percentile_cutoffs = np.percentile(y_std, percentile_bins)
        observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
        cum_observed_frequency = np.cumsum(observed_frequency)


        # Set up distribution and get fitted distribution parameters
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y_std)
        
        # Obtain the KS test P statistic, round it to 5 decimal places
        p = scipy.stats.kstest(y_std, distribution, args=param)[1]
        p = np.around(p, 5)
        p_values.append(p)    
        
        # Get expected counts in percentile bins
        # This is based on a 'cumulative distrubution function' (cdf)
        cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2], 
                            scale=param[-1])
        expected_frequency = []
        for bin in range(len(percentile_bins)-1):
            expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)
        
        # calculate chi-squared
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = sum (((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
        chi_square.append(ss)
            
    # Collate results and sort by goodness of fit (best at top)

    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['chi_square'] = chi_square
    results['p_value'] = p_values
    results.sort_values(['chi_square'], inplace=True)
        
    # Report results

    print ('\nDistributions sorted by goodness of fit:')
    print ('----------------------------------------')
    print (results)

    # Divide the observed data into 100 bins for plotting (this can be changed)
    number_of_bins = freedman_diaconis(y)
    bin_cutoffs = np.linspace(np.percentile(y,0), np.percentile(y,99),number_of_bins)

    # Create the plot
    h = plt.hist(y, bins = bin_cutoffs, color='0.75')

    # Get the top three distributions from the previous phase
    number_distributions_to_plot = 3
    dist_names = results['Distribution'].iloc[0:number_distributions_to_plot]

    # Create an empty list to stroe fitted distribution parameters
    parameters = []

    # Loop through the distributions ot get line fit and paraemters

    for dist_name in dist_names:
        # Set up distribution and store distribution paraemters
        dist = getattr(scipy.stats, dist_name)
        param = dist.fit(y)
        parameters.append(param)
        
        # Get line for each distribution (and scale to match observed data)
        pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])
        scale_pdf = np.trapz (h[0], h[1][:-1]) / np.trapz (pdf_fitted, x)
        pdf_fitted *= scale_pdf
        
        # Add the line to the plot
        plt.plot(pdf_fitted, label=dist_name)
        
        # Set the plot x axis to contain 99% of the data
        # This can be removed, but sometimes outlier data makes the plot less clear
        plt.xlim(0,np.percentile(y,99))

    # Add legend and display plot
    plt.title(sensor_name)
    plt.legend()
    plt.show()

    # Store distribution paraemters in a dataframe (this could also be saved)
    dist_parameters = pd.DataFrame()
    dist_parameters['Distribution'] = (results['Distribution'].iloc[0:number_distributions_to_plot])
    dist_parameters['Distribution parameters'] = parameters

    # Print parameter results
    print ('\nDistribution parameters:')
    print ('------------------------')

    for index, row in dist_parameters.iterrows():
        print ('\nDistribution:', row[0])
        print ('Parameters:', row[1] )