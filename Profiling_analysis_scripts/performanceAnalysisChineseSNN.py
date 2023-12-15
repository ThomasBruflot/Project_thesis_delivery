import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
#pd.show_versions(as_json=False)
# Define the column names
columns = ['Self CPU %', 'Self CPU', 'CPU total %', 'CPU total', 'CPU time avg', 'CPU Mem',
           'Self CPU Mem', '# of Calls', 'Total KFLOPs']



#This is from profiling the chinese network.
file_path1 = "Results_25Rows/pyMemLongAnalysisClean.out"

network_list_results_per_column = []
network_list_table_of_inference_dicts_accumulated = []
networks_collected_self_cpu_time_vectors = []


#Holds 4 dicts one for each network, each dict has keys of operators, then each dict holds a dict containing columns which in turn holds value vectors
operator_column_vectors_collected = [] 
CPU_mem_add_total = []
total_tables_ = 690


x_axis = np.array(list(range(1, 10001)))

filenames = [file_path1]


def printTable(myDict, colList=None):
   """ Pretty print a list of dictionaries (myDict) as a dynamically sized table.
   If column names (colList) aren't specified, they will show in random order.
   Author: Thierry Husson - Use it as you want but don't blame me.
   """
   myDict[0] = {key: "{:.2f}".format(value) for key, value in myDict[0].items()}
   if not colList: colList = list(myDict[0].keys() if myDict else [])
   myList = [colList] # 1st row = header
   for item in myDict: myList.append([str(item[col] if item[col] is not None else '') for col in colList])
   colSize = [max(map(len,col)) for col in zip(*myList)]
   formatStr = ' | '.join(["{{:<{}}}".format(i) for i in colSize])
   myList.insert(1, ['-' * i for i in colSize]) # Seperating line
   for item in myList: print(formatStr.format(*item))

def accumulate():
    # Read input file and process tables
    for file_path in filenames:

        aten_operator_column_vectors = {}
        CPU_mem_add = []

        count = 0
        accumulatedValues = {}
        dataList = []

        # Initialize total self CPU time and table count
        total_self_cpu_time = 0
        self_cpu_time_vector = []
        table_count = 0
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                # Check if the line starts with "Self CPU time total:"
                if line.startswith('Self CPU time total:'):
                    # Count tables based on the occurrence of "Self CPU time total:"
                    table_count += 1
                    # Extract and accumulate total self CPU time
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            total_self_cpu_time += float(parts[-1][:-1])  # Extracting the value before "s"
                            self_cpu_time_vector.append(float(parts[-1][:-1]))  # Extracting the value before "s"
                        except ValueError:
                            pass

        # Print total self CPU time and table count
        print(f'Total Self CPU Time: {total_self_cpu_time:.3f} s')
        print(f'Total Tables: {table_count}')
        #print(f'Standard deviation for the CPU Time [ms]: {pd.DataFrame(self_cpu_time_vector).std()}')
        networks_collected_self_cpu_time_vectors.append(self_cpu_time_vector)

        with open(file_path, 'r') as file:
            data = file.readlines()
            for line in data: 

                if ("aten" in line):

                    index_list = []
                    splitList = line.split()
                    for i in range(len(splitList)):
                        if splitList[i] == "Kb":
                            splitList[i-1] = str(float(splitList[i-1])*1000)
                        if splitList[i] == "Mb":
                            splitList[i-1] = str(float(splitList[i-1])*1000*1000)
                        if splitList[i] == "Gb":
                            splitList[i-1] = str(float(splitList[i-1])*1000*1000*1000)
                    splitList = [item for item in splitList if item != "Kb" and item != "b" and item != "Mb" and item != "Gb"]
                    for i in range(len(splitList)):
                        if splitList[i] == "--":
                            splitList[i] = "0.0"
                        if "%" in splitList[i]:
                            splitList[i] = splitList[i].replace("%","")
                        if ("s" in splitList[i]) and ("aten" not in splitList[i]) and ("ms" not in splitList[i])and ("us" not in splitList[i]):
                            splitList[i] = str(float(splitList[i][:-1])*1000*1000) # convert to us
                        if "ms" in splitList[i]: 
                            splitList[i] = str(float(splitList[i][:-2])*1000) # convert to us
                        if "us" in splitList[i]:
                            splitList[i] = str(float(splitList[i][:-2])) # keep us

                    dataList.append(splitList)



        counter = 0
        single_inference_accumulated = {column: 0 for column in columns}
        table_of_inference_dicts_accumulated = {column: [] for column in columns}
        counter_max = 9
        canCount = True
        for values in dataList:
            canCount = True
            for i in range(1,len(values)):
                single_inference_accumulated[columns[i-1]] += float(values[i])
            if counter == counter_max:
                for column_name_key in single_inference_accumulated:
                    table_of_inference_dicts_accumulated[column_name_key].append(single_inference_accumulated[column_name_key])
                single_inference_accumulated = {column: 0 for column in columns}
                counter = 0
                canCount = False
            
            if canCount:
                counter += 1
            
            if values[0] not in accumulatedValues:
                accumulatedValues[values[0]] = {column: 0 for column in columns}
            for i in range(1,len(values)):
                accumulatedValues[values[0]][columns[i-1]] += float(values[i])

            if values[0] not in aten_operator_column_vectors:
                aten_operator_column_vectors[values[0]] = {column: [] for column in columns}

                    
            for i in range(1,len(values)):
                aten_operator_column_vectors[values[0]][columns[i-1]].append(float(values[i])) #Change from adding the numbers to appending to the list.

            if values[0] == "aten::add":
                CPU_mem_add.append(float(values[columns.index("CPU Mem")+1]))

        resultsPerColumn = {}

        for key in accumulatedValues:

            for subkey in accumulatedValues[key]:

                if subkey not in resultsPerColumn:
                    resultsPerColumn[subkey] = float(accumulatedValues[key][subkey])
                else:
                    resultsPerColumn[subkey] += float(accumulatedValues[key][subkey])
        CPU_mem_add_total.append(CPU_mem_add)            
        operator_column_vectors_collected.append(aten_operator_column_vectors)
        network_list_results_per_column.append(resultsPerColumn)
        network_list_table_of_inference_dicts_accumulated.append(table_of_inference_dicts_accumulated)
        

        
accumulate()


"""
The idea behind this script is to take each network and get averages, maximums, and totals for each column
into a single table in the report.
So the output should be a dict for each network. 
Making three dicts per network could also be useful . One for avgs one for maxs and one for totals. 
I can then adjust which I want to avg or take max or whatever.

Firstly: I have a dict containing the columns summed up for all the networks. 
So to just process that one is the most smart thing to do.
"""

def avgDictionaries(number_of_images = 10000):
    """
    Here I take avg
    I also convert from us to ms some places and from b to Kb.
    """
    networks = ["DongEtAl-6400"]
    print("------------------")
    print("Starting averaging...")
    print("Units: ")
    print("% ", "s ", "% ", "s ", "us", "Gb", "Gb")
    print("------------------")
    for idx, network_result_dict in enumerate(network_list_results_per_column):
        print(networks[idx])
        network_result_dict['Self CPU %'] = network_result_dict['Self CPU %']/ total_tables_
        network_result_dict['Self CPU'] = (network_result_dict['Self CPU']/(10**(6)))/total_tables_ #Convert to s
        network_result_dict['CPU total %'] = network_result_dict['CPU total %']/total_tables_
        network_result_dict['CPU total'] = (network_result_dict['CPU total']/(10**(6))) /total_tables_ #Convert to s
        network_result_dict['CPU time avg'] = network_result_dict['CPU time avg']/total_tables_
        network_result_dict['CPU Mem'] = (network_result_dict['CPU Mem']/(10**(9)))/total_tables_ #Convert to Gb
        network_result_dict['Self CPU Mem'] = (network_result_dict['Self CPU Mem']/(10**(9)))/total_tables_ #Convert to Gb
        network_result_dict['# of Calls'] = network_result_dict['# of Calls']/total_tables_
        network_result_dict['Total KFLOPs'] = network_result_dict['Total KFLOPs']/total_tables_ # This is MFLOPS actually
        print(network_result_dict)
        print("------------------")
    
   
avgDictionaries()





def maxDictionaries(number_of_images = 10000):
    """
    Here I take max
    I also convert from us to ms some places and from b to Kb.
    """
    networks = ["DongEtAl-6400"]
    
    maximums_per_network = []

    print("------------------")
    print("Starting maximum finding...")
    print("Units: ")
    print("% ", "s ", "% ", "s ", "us", "Gb", "Gb")
    print("------------------")
    for idx, table_dict in enumerate(network_list_table_of_inference_dicts_accumulated):
        print(networks[idx])
        max_per_inference_table = {column: 0 for column in columns}
    
        max_per_inference_table['Self CPU %'] = max(table_dict['Self CPU %'])
        max_per_inference_table['Self CPU'] = max(table_dict['Self CPU'])/(10**(6)) #Convert to s
        max_per_inference_table['CPU total %'] = max(table_dict['CPU total %'])
        max_per_inference_table['CPU total'] = max(table_dict['CPU total'])/(10**(6)) #Convert to s
        max_per_inference_table['CPU time avg'] = max(table_dict['CPU time avg'])
        max_per_inference_table['CPU Mem'] = max(table_dict['CPU Mem'])/(10**(9)) #Convert to Gb
        max_per_inference_table['Self CPU Mem'] = max(table_dict['Self CPU Mem'])/(10**(9)) #Convert to Gb
        max_per_inference_table['# of Calls'] = max(table_dict['# of Calls'])
        max_per_inference_table['Total KFLOPs'] = max(table_dict['Total KFLOPs']) #This is MFLOPS actually
            
        maximums_per_network.append(max_per_inference_table)
        print(max_per_inference_table)
        print("------------------")
    
#maxDictionaries()



def boxPlotSingleInferences():
    networks = ["DongEtAl-6400"]

    columnName = "Self CPU %"
    scaling = 1#10**(3)
    data_dict = {networks[0]: (np.array(network_list_table_of_inference_dicts_accumulated[0][columnName]))/scaling
                 }

    df = pd.DataFrame(data_dict)

    ax = sns.boxplot(data=df, palette='Greens')
    #plt.ylabel('Log10(Self memory use in Kb)')
    plt.ylabel('Self CPU \%')
    #plt.title("Logarithmic boxplot of self memory for different network configurations of the Diehl & Cook network")
    plt.title("Boxplot of self CPU \% for the Dong et. al. network")
    plt.show()

#boxPlotSingleInferences()
''' 

I need to define what  I want to plot here. 
Let us look at memory allocation for self calls.


'''

def boxPlotSingleInferenceRuntimes():
    networks = ["DongEtAl-6400"]

    data_dict = {networks[0]: np.array(networks_collected_self_cpu_time_vectors[0]) # Convert to seconds
                 }

    df = pd.DataFrame(data_dict)

    ax = sns.boxplot(data=df, palette='Oranges')
    #plt.ylabel('Log10(Self memory use in Kb)')
    plt.ylabel('Self runtime in s')
    #plt.title("Logarithmic boxplot of self memory for different network configurations of the Diehl & Cook network")
    plt.title("Boxplot of self runtime for the Dong et. al. network")
    plt.show()

#boxPlotSingleInferenceRuntimes()


'''
Now I will evaluate which operators use the most operations.

Network names as columns
operators as rows
# of calls in each matrix entry. 

Structure of operator_column_vectors_collected:
Consists of multiple dictionaries called aten_operator_column_vectors that holds as keys operator names
For each operator name it holds a new dictionary that holds lists as the values and as keys it holds column names for performance indicators.

'''


def presentNumCallsPerNetworkPerOperator():
    networks = ["DongEtAl-6400"]
    key_dict = {}
    for idx, aten_operator_column_vector in enumerate(operator_column_vectors_collected):
        print("---------------------------- New network ---------------------")
        print(networks[idx])
        print("\n")
        for key, columnDictionary in aten_operator_column_vector.items():
            print(key)
            print(sum(np.array(columnDictionary['# of Calls'])))
            # This will append for each operator a vector that has all the sums to be easily formatted in latex. 
            if key not in key_dict:
                key_dict[key] = [sum(np.array(columnDictionary['# of Calls']))]
            else:
                key_dict[key].append(sum(np.array(columnDictionary['# of Calls'])))
        for operator in key_dict:
            print(operator, " & ", " & ".join(str(x) for x in key_dict[operator]), " \\", "\\", " hline")

        print("\n")
#presentNumCallsPerNetworkPerOperator()

def boxPlotSingleInferencesMemory():
    networks = ["DongEtAl-6400"]

    columnName = "Self CPU Mem"
    scaling = 10**(9) # Convert to Gb
    data_dict = {networks[0]: (np.array(network_list_table_of_inference_dicts_accumulated[0][columnName]))/scaling
                 }

    df = pd.DataFrame(data_dict)

    ax = sns.boxplot(data=df, palette='Reds')
    #plt.ylabel('Log10(Self memory use in Kb)')
    plt.ylabel('Self Mem [Gb]')
    #plt.title("Logarithmic boxplot of self memory for different network configurations of the Diehl & Cook network")
    plt.title("Boxplot of self memory for the Dong et. al. network")
    plt.show()

#boxPlotSingleInferencesMemory()

def minDictionaries(number_of_images = 10000):
    """
    Here I take min
    I also convert from us to ms some places and from b to Kb.
    """
    networks = ["DongEtAl-6400"]
    
    minimums_per_network = []

    print("------------------")
    print("Starting minimum finding...")
    print("Units: ")
    print("% ", "s ", "% ", "s ", "us", "Gb", "Gb")
    print("------------------")
    for idx, table_dict in enumerate(network_list_table_of_inference_dicts_accumulated):
        print(networks[idx])
        min_per_inference_table = {column: 0 for column in columns}
    
        min_per_inference_table['Self CPU %'] = min(table_dict['Self CPU %'])
        min_per_inference_table['Self CPU'] = min(table_dict['Self CPU'])/(10**(6)) #Convert to s
        min_per_inference_table['CPU total %'] = min(table_dict['CPU total %'])
        min_per_inference_table['CPU total'] = min(table_dict['CPU total'])/(10**(6)) #Convert to s
        min_per_inference_table['CPU time avg'] = min(table_dict['CPU time avg'])
        min_per_inference_table['CPU Mem'] = min(table_dict['CPU Mem'])/(10**(9)) #Convert to Gb
        min_per_inference_table['Self CPU Mem'] = min(table_dict['Self CPU Mem'])/(10**(9)) #Convert to Gb
        min_per_inference_table['# of Calls'] = min(table_dict['# of Calls'])
        min_per_inference_table['Total KFLOPs'] = min(table_dict['Total KFLOPs'])
            
        minimums_per_network.append(min_per_inference_table)
        print(min_per_inference_table)
        print("------------------")
    
minDictionaries()