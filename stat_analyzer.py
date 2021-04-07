import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chisquare
from custom_functions.freedman_diaconis import freedman_diaconis
from custom_functions.statistical_distribuition_tools import dist_finder

def main():
    # Analyze statistical properties of the data
    # sensor_data_table = pd.read_csv("datasets/gas_sensor_array.csv")
    sensor_data_table = pd.read_csv("labeled_dataset.csv")

    # Get amt of clusters from dataset
    num_of_clusters = sensor_data_table['Cluster'].max()
    
    # For each sensor in each cluster
    #TODO: Separate each sensor into new datasets
    for cluster, dataset_cluster in sensor_data_table.groupby('Cluster'):
        print(dataset_cluster)
        # if sensor_data_table.loc[sensor_data_table['Cluster'] == cluster]:


    dist_finder("Temperature")

    # for sensor_name in list(sensor_data_table.columns):
        # dist_finder(sensor_name)

    # Separate each sensor and plot
    # for sensor_name in list(sensor_data_table.columns):
        # Calc suggested number of bins in histogram based on Freedman-Diaconis rule
        # amt_of_bins = freedman_diaconis(sensor_data_table[sensor_name])


        # Compare to other probability distribution
        # for prob_dist in dists:
        #     chi2 = chisquare(sensor_data_table[sensor_name], prob_dist)
        #     print("Chi square value %d", chi2)
    
        # Save probability distribution, max value, min value


    # Output to json file with states
    # or as each unique value in sensor
    # amt_of_bins = np.arange(sensor_data_table[sensor_name].min(), sensor_data_table[sensor_name].max())
    
    # plt.hist(sensor_data_table[sensor_name], amt_of_bins)
    # plt.title(sensor_name)
    # plt.show()

if __name__ == '__main__':
    main()