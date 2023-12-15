import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
pd.show_versions(as_json=False)
# Define the column names
columns = ['Self CPU %', 'Self CPU', 'CPU total %', 'CPU total', 'CPU time avg', 'CPU Mem',
           'Self CPU Mem', '# of Calls', 'Total KFLOPs']

#These have 100 and 50 neurons but only 10 rows out for profiling
file_path1 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/diehlAndCookExperiments/cleanedOutFiles/diehlAndCookStandardCPUClean.out"
file_path2 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/diehlAndCookExperiments/cleanedOutFiles/diehlAndCookReducedCPUClean.out"
file_path3 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/diehlAndCookExperiments/cleanedOutFiles/diehlAndCookHebbianCPUClean.out"
file_path4 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/diehlAndCookExperiments/cleanedOutFiles/diehlAndCookWDSTDPCPUClean.out"

#These have 100 and 50 neurons but they now have 25 rows out
file_path5 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/Results_25Rows/diehlAndCookStandard_nograd_25rowsClean.out"
file_path6 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/Results_25Rows/diehlAndCookReduced_nograd_25rowsClean.out"
file_path7 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/Results_25Rows/diehlAndCookHebbian_nograd_25rowsClean.out"
file_path8 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/Results_25Rows/diehlAndCookWDSTDP_nograd_25rowsClean.out"

#These have 100 neurons but adjusted some other params. 25 rows profiling out
file_path9 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/Results_25Rows/diehlAndCookStandardAdjInhib_nograd_25rowsClean.out"
file_path10 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/Results_25Rows/diehlAndCookStandardAdjThresh_nograd_25rowsClean.out"

#These have 1600 neurons, only 10 rows from profiling
file_path11 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/largeNetworkDiehlResults/diehlAndCookReduced_nogradClean.out"
file_path12 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/largeNetworkDiehlResults/diehlAndCookHebbian_nogradClean.out"
file_path13 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/largeNetworkDiehlResults/diehlAndCookWDSTDP_nogradClean.out"

#This is from profiling the chinese network. Partially done training, maybe it can still be used however? Need to analyze today
#file_path11 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/Results_25Rows/pyMemLongAnalysisClean.out"



#Holds 4 dicts one for each network, each dict has keys of operators, then each dict holds a dict containing columns which in turn holds value vectors
operator_column_vectors_collected = [] 
CPU_mem_add_total = []


x_axis = np.array(list(range(1, 10001)))

filenames = [file_path5, file_path6,file_path7, file_path8, file_path9, file_path10, file_path11,file_path12,file_path13]
#filenames = [file_path1, file_path2,file_path3, file_path4]

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
                    #print("parts: ", parts)
                    if len(parts) >= 5:
                        try:
                            total_self_cpu_time += float(parts[-1][:-2])  # Extracting the value before "ms"
                        except ValueError:
                            pass

        # Print total self CPU time and table count
        print(f'Total Self CPU Time: {total_self_cpu_time:.3f} ms')
        print(f'Total Tables: {table_count}')

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
                        if "ms" in splitList[i]:
                            splitList[i] = str(float(splitList[i][:-2])*1000)
                        if "us" in splitList[i]:
                            splitList[i] = splitList[i][:-2]

                    dataList.append(splitList)
                    ##Now I have a list of all the data with a name in the first entry. 
                # I want to store them into a new table now with all the column names and row names in a pandas df.
                # Maybe I just make a 2D array to feed to the pandas dataframe
                # They want a dict. I can make this

        print("Length ", len(dataList))
        print("length one", len(dataList[0]))
        for values in dataList:
            #print(values)
            #print(values[0])
            if values[0] not in accumulatedValues:
                accumulatedValues[values[0]] = {column: 0 for column in columns}
            for i in range(1,len(values)):
                accumulatedValues[values[0]][columns[i-1]] += float(values[i])

            if values[0] not in aten_operator_column_vectors:
                aten_operator_column_vectors[values[0]] = {column: [] for column in columns}
                #for column in aten_operator_column_vectors[values[0]]:
                #    aten_operator_column_vectors[values[0]][column] = 
                    
            #print(len(values))
            for i in range(1,len(values)):
                aten_operator_column_vectors[values[0]][columns[i-1]].append(float(values[i])) #Change from adding the numbers to appending to the list.
                #print("oooooooooooooo")
                #print(values[i])
                #print(aten_operator_column_vectors[values[0]][columns[i-1]])
                #print(aten_operator_column_vectors[values[0]])
                #print("key", values[0])
                #print("-----------")


            if values[0] == "aten::add":
                CPU_mem_add.append(float(values[columns.index("CPU Mem")+1]))
                #num_calls_add.append(float(values[columns.index('# of Calls')]))

        resultsPerColumn = {}

        for key in accumulatedValues:

            for subkey in accumulatedValues[key]:

                if subkey not in resultsPerColumn:
                    resultsPerColumn[subkey] = float(accumulatedValues[key][subkey])
                else:
                    resultsPerColumn[subkey] += float(accumulatedValues[key][subkey])
        CPU_mem_add_total.append(CPU_mem_add)            
        operator_column_vectors_collected.append(aten_operator_column_vectors)

        # sorting both the lists


        test_list1 = CPU_mem_add
        test_list2 = aten_operator_column_vectors["aten::add"]["CPU Mem"]
        #print(test_list1)
        #print(test_list2)
        test_list1.sort()
        test_list2.sort()

        # using == to check if
        # lists are equal
        if test_list1 == test_list2:
            print("The lists are identical")
        else:
            print("The lists are not identical")
        

        
accumulate()



def boxplotting(operationName, columnName, scaling=1000):
    lengthArr = len(operator_column_vectors_collected[0][operationName][columnName])

    # Dataset creation
    dfs = []
    configurations = ['Standard100', 'Standard50', 'Hebbian100', 'WDSTDP100', 'Adjusted_inhibition100',
                      'Adjusted_threshold100', 'Standard1600', 'Hebbian1600', 'WDSTDP1600']
    
    for idx, config in enumerate(configurations):
        value_arr = np.array(operator_column_vectors_collected[idx][operationName][columnName]) / scaling
        temp_df = pd.DataFrame({'Network configuration': np.repeat(config, lengthArr), 'value': value_arr})
        dfs.append(temp_df)

    df = pd.concat(dfs, ignore_index=True)
    df['log_value'] = np.emath.logn(1000000, df['value'])
    sns.boxplot(x='Network configuration', y='log_value', data=df)
    plt.ylabel('Log20(Value)')
    plt.show()

#boxplotting('aten::add', 'CPU Mem', scaling=1000)


def boxplottingAttempt2(operationName, columnName, scaling=1000):
    configurations = ['Standard100', 'Standard50', 'Hebbian100', 'WDSTDP100', 'Adjusted_inhibition100',
                      'Adjusted_threshold100', 'Standard1600', 'Hebbian1600', 'WDSTDP1600']
    data_dict = {config: np.array(operator_column_vectors_collected[idx][operationName][columnName])/scaling 
                 for idx, config in enumerate(configurations)}

    df = pd.DataFrame(data_dict)

    ax = sns.boxplot(data=df)
    plt.ylabel('Memory use in Kb')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.show()

#boxplottingAttempt2('aten::add', 'CPU Mem', scaling=1000)

def boxplottingAttempt3(operationName, columnName, scaling=1000):
    configurations = ['Standard100', 'Standard50', 'Hebbian100', 'WDSTDP100', 'Adjusted_inhibition100',
                      'Adjusted_threshold100', 'Standard1600', 'Hebbian1600', 'WDSTDP1600']
    
    data_dict = {config: np.log10(np.array(operator_column_vectors_collected[idx][operationName][columnName])/scaling)
                 for idx, config in enumerate(configurations)}

    df = pd.DataFrame(data_dict)

    ax = sns.boxplot(data=df)
    plt.ylabel('Log(Memory use in Kb)')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.show()

#boxplottingAttempt3('aten::add', 'CPU Mem', scaling=1000)


def boxplottingAttempt4(operationName, columnName, scaling=1000):
    configurations = ['Standard100', 'Hebbian100', 'WDSTDP100', 'Adjusted_inhibition100',
                      'Adjusted_threshold100']
    
    data_dict = {configurations[0]: np.log10(np.array(operator_column_vectors_collected[0][operationName][columnName])/scaling),
                 configurations[1]: np.log10(np.array(operator_column_vectors_collected[2][operationName][columnName])/scaling),
                 configurations[2]: np.log10(np.array(operator_column_vectors_collected[3][operationName][columnName])/scaling),
                 configurations[3]: np.log10(np.array(operator_column_vectors_collected[4][operationName][columnName])/scaling),
                 configurations[4]: np.log10(np.array(operator_column_vectors_collected[5][operationName][columnName])/scaling)}

    df = pd.DataFrame(data_dict)

    ax = sns.boxplot(data=df,palette='Greens')
    plt.ylabel('Log(Memory use in Kb)')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.show()

boxplottingAttempt4('aten::add', 'CPU Mem', scaling=1000)
