import pandas as pd
import sys

from custom_functions.statistical_distribuition_tools import dist_finder

def main():
    # Analyze statistical properties of the data
    sensor_data_table = pd.read_csv("labeled_dataset_iot_telemetry_2.csv")

    # Get amt of clusters from dataset
    num_of_clusters = sensor_data_table['Cluster'].max()
    ite = 0

    #TODO: Convert print out from .txt to .json
    sys.stdout = open('output_sensor_sates_information_v2.txt','wt')

    while ite <= num_of_clusters:
        grouped_by_cluster = sensor_data_table.groupby(sensor_data_table['Cluster'])
        data_one_cluster = grouped_by_cluster.get_group(ite)
        print("State: "+ str(ite))
        for sensor_name in list(sensor_data_table.columns):
            # Exclude Time and cluster values since they are not sensor values
            if sensor_name == "Time" or sensor_name == "Cluster": continue
            
            df = dist_finder(sensor_name,data_one_cluster)
            print("Sensor name: " + sensor_name)
            print("Max value: " + str(sensor_data_table[sensor_name].max()))
            print("Min value: " + str(sensor_data_table[sensor_name].min()))
            print("Statistical distribution: " + df['Distribution'].iloc[0])
            print("Distribution parameters: ") 
            print(df['Parameters'].iloc[0])
            print("Chi2 value: " + str(df['chi_square'].iloc[0]))
            print("P value: " + str(df['p_value'].iloc[0]))
        print("End of state: "+ str(ite)+"\n")
        ite = ite + 1
 
        

if __name__ == '__main__':
    main()