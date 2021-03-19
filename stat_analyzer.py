import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chisquare
from custom_functions.freedman_diaconis import freedman_diaconis

def main():
    # Analyze statistical properties of the data
    sensor_data_table = pd.read_csv("datasets/gas_sensor_array.csv")
    
    #TODO: Create list for each statistical distribution possible
    dists = list()

    # Separate each sensor and plot
    for sensor_name in list(sensor_data_table.columns):
        # Calc suggested number of bins in  histogram based on Freedman-Diaconis rule
        amt_of_bins = freedman_diaconis(sensor_data_table[sensor_name])

        # or as each unique value in sensor
        # amt_of_bins = np.arange(sensor_data_table[sensor_name].min(), sensor_data_table[sensor_name].max())
    
        plt.hist(sensor_data_table[sensor_name], amt_of_bins)
        plt.title(sensor_name)
        plt.show()

        # Compare to other probability distribution
        for prob_dist in dists:
            ch2value = chisquare(sensor_data_table[sensor_name],prob_dist)
    
        # Save probability distribution, max value, min value


    # Output to json file with states


if __name__ == '__main__':
    main()