import pandas as pd
import sys

from custom_functions.statistical_distribuition_tools import dist_finder

def main():
    # Analyze statistical properties of the data
    sensor_data_table = pd.read_csv("labeled_dataset.csv")

    # Get amt of clusters from dataset
    num_of_clusters = sensor_data_table['Cluster'].max()
    ite = 0
    sys.stdout = open('output-gas-sensor-array.txt','wt')

    while ite <= num_of_clusters:

        grouped_by_cluster = sensor_data_table.groupby(sensor_data_table['Cluster'])
        data_one_cluster = grouped_by_cluster.get_group(ite)
        for sensor_name in list(sensor_data_table.columns):
            if sensor_name == "Cluster": continue
            df = dist_finder(sensor_name,data_one_cluster)
            print("state: "+ str(ite))
            print(sensor_name)
            print(df['Distribution'].iloc[0])
            print(df['chi_square'].iloc[0])
            print(df['p_value'].iloc[0])
        ite = ite + 1
 
        

if __name__ == '__main__':
    main()