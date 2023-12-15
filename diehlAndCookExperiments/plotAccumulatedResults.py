import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
# Define the column names
columns = ['Self CPU %', 'Self CPU', 'CPU total %', 'CPU total', 'CPU time avg', 'CPU Mem',
           'Self CPU Mem', '# of Calls', 'Total KFLOPs']

file_path1 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/diehlAndCookExperiments/cleanedOutFiles/diehlAndCookStandardCPUClean.out"
file_path2 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/diehlAndCookExperiments/cleanedOutFiles/diehlAndCookReducedCPUClean.out"
file_path3 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/diehlAndCookExperiments/cleanedOutFiles/diehlAndCookHebbianCPUClean.out"
file_path4 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/diehlAndCookExperiments/cleanedOutFiles/diehlAndCookWDSTDPCPUClean.out"


#file_path5 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/Results_25Rows/diehlAndCookStandard_nograd_25rowsClean.out"
file_path6 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/Results_25Rows/diehlAndCookReduced_nograd_25rowsClean.out"
file_path7 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/Results_25Rows/diehlAndCookHebbian_nograd_25rowsClean.out"
file_path8 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/Results_25Rows/diehlAndCookWDSTDP_nograd_25rowsClean.out"

#file_path9 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/Results_25Rows/diehlAndCookStandardAdjInhib_nograd_25rowsClean.out"
#file_path10 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/Results_25Rows/diehlAndCookStandardAdjThresh_nograd_25rowsClean.out"


#file_path11 = "/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/Programming/Results_25Rows/pyMemLongAnalysisClean.out"



#Holds 4 dicts one for each network, each dict has keys of operators, then each dict holds a dict containing columns which in turn holds value vectors
operator_column_vectors_collected = [] 
CPU_mem_add_total = []


x_axis = np.array(list(range(1, 10001)))

#filenames = [file_path1, file_path6,file_path7, file_path8]
filenames = [file_path1, file_path2,file_path3, file_path4]

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

    




"""
They want min, max, median and average for all the params and then we can plot those into a histogram.

I already have arrays for all the different ones. Should be fairly straight forward to do.

"""

def medAvgMinMax(operationName, columnName):
    medArr = []
    minArr = []
    maxArr = []
    avgArr = []
    if columnName == "CPU Mem":
        if operationName == "aten::add":
            for arra in operator_column_vectors_collected:
                arr = arra["aten::add"]["CPU Mem"]
                medArr.append(np.median(np.array(arr)))
                minArr.append(min(arr))
                maxArr.append(max(arr))
                avgArr.append(np.average(np.array(arr)))
        elif operationName == "aten::mul":
            for arra in operator_column_vectors_collected:
                arr = arra["aten::mul"]["CPU Mem"]
                medArr.append(np.median(np.array(arr)))
                minArr.append(min(arr))
                maxArr.append(max(arr))
                avgArr.append(np.average(np.array(arr)))
        elif operationName == "aten::mm":
            for arra in operator_column_vectors_collected:
                arr = arra["aten::mm"]["CPU Mem"]
                medArr.append(np.median(np.array(arr)))
                minArr.append(min(arr))
                maxArr.append(max(arr))
                avgArr.append(np.average(np.array(arr)))
    return medArr, avgArr,minArr, maxArr


medianAdd, averageAdd, minimumAdd, maximumAdd = medAvgMinMax("aten::add", "CPU Mem")
print("medians: ", medianAdd, "\n Averages: ", averageAdd, " \nMinimums: ", minimumAdd, "\nMaximums: ", maximumAdd)

def barPlotting(medians, averages, minimums, maximums, operationName, columnName):
    titleOfPlotMedian = "Histogram of the median values for the " + columnName + " for the " + operationName + " operation for the 4 different network configurations"
    titleOfPlotAverage = "Histogram of the average values for the " + columnName + " for the " + operationName + " operation for the 4 different network configurations"
    titleOfPlotMinimum = "Histogram of the minimum values for the " + columnName + " for the " + operationName + " operation for the 4 different network configurations"
    titleOfPlotMaximum = "Histogram of the maximum values for the " + columnName + " for the " + operationName + " operation for the 4 different network configurations"
    categories = ["Standard", "Reduced", "Hebbian", "WDSTDP"]


    ax = sns.barplot(x=categories, y=medians)
    ax.bar_label(ax.containers[0])
    # Create custom legend
    legend_labels = categories
    legend_colors = [patch.get_facecolor() for patch in ax.patches]
    legend_patches = [Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
    # Display custom legend with specified labels and colors
    plt.legend(handles=legend_patches, title="Network topology", loc='upper right')
    plt.title(titleOfPlotMedian)
    plt.axis('off')
    plt.show()

    ax = sns.barplot(x=categories, y=averages)
    ax.bar_label(ax.containers[0])
    # Create custom legend
    legend_labels = categories
    legend_colors = [patch.get_facecolor() for patch in ax.patches]
    legend_patches = [Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
    # Display custom legend with specified labels and colors
    plt.legend(handles=legend_patches, title="Network topology", loc='upper right')
    plt.title(titleOfPlotAverage)
    plt.axis('off')
    plt.show()

    ax = sns.barplot(x=categories, y=minimums)
    ax.bar_label(ax.containers[0])
    # Create custom legend
    legend_labels = categories
    legend_colors = [patch.get_facecolor() for patch in ax.patches]
    legend_patches = [Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
    # Display custom legend with specified labels and colors
    plt.legend(handles=legend_patches, title="Network topology", loc='upper right')
    plt.title(titleOfPlotMinimum)
    plt.axis('off')
    plt.show()

    ax = sns.barplot(x=categories, y=maximums)
    ax.bar_label(ax.containers[0])
    # Create custom legend
    legend_labels = categories
    legend_colors = [patch.get_facecolor() for patch in ax.patches]
    legend_patches = [Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
    # Display custom legend with specified labels and colors
    plt.legend(handles=legend_patches, title="Network topology", loc='upper right')
    plt.title(titleOfPlotMaximum)
    plt.axis('off')
    plt.show()



#barPlotting(medianAdd, averageAdd, minimumAdd, maximumAdd, "aten::add", "CPU Mem")
#
#medianMul, averageMul, minimumMul, maximumMul = medAvgMinMax("aten::mul", "CPU Mem")
#print("medians: ", medianMul, "\n Averages: ", averageMul, " \nMinimums: ", minimumMul, "\nMaximums: ", maximumMul)
#barPlotting(medianMul, averageMul, minimumMul, maximumMul, "aten::mul", "CPU Mem")
#
#medianMm, averageMm, minimumMm, maximumMm = medAvgMinMax("aten::mm", "CPU Mem")
#print("medians: ", medianMm, "\n Averages: ", averageMm, " \nMinimums: ", minimumMm, "\nMaximums: ", maximumMm)
#barPlotting(medianMm, averageMm, minimumMm, maximumMm, "aten::mm", "CPU Mem")


"""
I want to plot how the different columns change over time. Lets start one at a time. :)
Lets look at add, mul and mm. Lets look at how much memory they use for example.
What do I need?
I need to make a list containing all the measurements for each single table and then plot like normal.

TODO: Make the histogram plots nice with legends and title and etc.

"""

def plotOperationResourceUse(operationName, networkName, CPU_mem_index, columnName): 
    y_axis = operator_column_vectors_collected[CPU_mem_index][operationName]["CPU Mem"]

    if networkName == "Reduced":
        sub = x_axis[:-1]
        print(y_axis[5003])
        y_axis.pop(5003)

    fig, ax = plt.subplots()
    ax.set_xlabel('Test image')
    ax.set_ylabel('Memory in Kb')
    title = 'Memory usage for ' + operationName + ' in ' +networkName+' network'
    ax.set_title(title)
    #ax.set_ylim([1.900, 5.713])
    #plt.yticks(1.900, 5.713,0.5)
    plt.yticks(np.arange(min(y_axis), max(y_axis)+1, (max(y_axis)+1)/10))
    if networkName == "Reduced":
        plt.scatter(sub, np.array(y_axis))
    else:
        plt.scatter(x_axis, np.array(y_axis))
    plt.show()

#plotOperationResourceUse("aten::add", "Standard", 0, "CPU Mem")
#plotOperationResourceUse("aten::add", "Reduced", 1, "CPU Mem")
#plotOperationResourceUse("aten::add", "Hebbian", 2, "CPU Mem")
#plotOperationResourceUse("aten::add", "WDSTDP", 3, "CPU Mem")

#plotOperationResourceUse("aten::mul", "Standard", 0, "CPU Mem")
#plotOperationResourceUse("aten::mul", "Reduced", 1, "CPU Mem")
#plotOperationResourceUse("aten::mul", "Hebbian", 2, "CPU Mem")
#plotOperationResourceUse("aten::mul", "WDSTDP", 3, "CPU Mem")

#plotOperationResourceUse("aten::mm", "Standard", 0, "CPU Mem")
#plotOperationResourceUse("aten::mm", "Reduced", 1, "CPU Mem")
#plotOperationResourceUse("aten::mm", "Hebbian", 2, "CPU Mem")
#plotOperationResourceUse("aten::mm", "WDSTDP", 3, "CPU Mem")


def plotOperationNumOfCallsPerImage(operationName, networkName, num_call_index): 
    y_axis = operator_column_vectors_collected[num_call_index][operationName]["# of Calls"]

    fig, ax = plt.subplots()
    ax.set_xlabel('Test image index')
    ax.set_ylabel('Number of calls')
    title = 'Number of calls for ' + operationName + ' in ' +networkName+' network'
    ax.set_title(title)
    #ax.set_ylim([1.900, 5.713])
    #plt.yticks(1.900, 5.713,0.5)
    plt.yticks(np.arange(min(y_axis), max(y_axis)+1, (max(y_axis)+1)/10))
    plt.scatter(x_axis, np.array(y_axis))
    plt.show()

#plotOperationNumOfCallsPerImage("aten::add", "Standard", 0)
#plotOperationNumOfCallsPerImage("aten::add", "Reduced", 1)
#plotOperationNumOfCallsPerImage("aten::add", "Hebbian", 2)
#plotOperationNumOfCallsPerImage("aten::add", "WDSTDP", 3)

#plotOperationNumOfCallsPerImage("aten::mul", "Standard", 0)
#plotOperationNumOfCallsPerImage("aten::mul", "Reduced", 1)
#plotOperationNumOfCallsPerImage("aten::mul", "Hebbian", 2)
#plotOperationNumOfCallsPerImage("aten::mul", "WDSTDP", 3)

#plotOperationNumOfCallsPerImage("aten::mm", "Standard", 0)
#plotOperationNumOfCallsPerImage("aten::mm", "Reduced", 1)
#plotOperationNumOfCallsPerImage("aten::mm", "Hebbian", 2)
#plotOperationNumOfCallsPerImage("aten::mm", "WDSTDP", 3)

#print(operator_column_vectors_collected[1]["aten::add"]["CPU Mem"])
def histogramPlotting(operationName, columnName,numBins = 20):
    titleOfPlot = 'Histogram of Memory use by the ' + operationName + 'operation for 4 different configurations of the Diehl & Cook SNN'
    colors = ['red', 'green', 'blue', 'orange']
    operator_column_vectors_collected[1]["aten::add"]["CPU Mem"][5003] = 0 #This is because there is a spike in reduced. This will be corrected once the run is finished in a few days with good data
    operator_column_vectors_collected[1]["aten::mm"]["CPU Mem"][5003] = 0 #This is because there is a spike in reduced. This will be corrected once the run is finished in a few days with good data
    plt.hist(operator_column_vectors_collected[0][operationName][columnName], bins=numBins, alpha=0.5, color=colors[0], label=f'Standard')
    plt.hist(operator_column_vectors_collected[1][operationName][columnName], bins=numBins, alpha=0.5, color=colors[1], label=f'Reduced')
    plt.hist(operator_column_vectors_collected[2][operationName][columnName], bins=numBins, alpha=0.5, color=colors[2], label=f'Hebbian')
    plt.hist(operator_column_vectors_collected[3][operationName][columnName], bins=numBins, alpha=0.5, color=colors[3], label=f'WDSTDP')
    
    # Set labels and legend
    plt.xlabel('Memory use in Kb')
    plt.ylabel('# of samples within the bins')
    plt.title(titleOfPlot)
    plt.legend()
    
    # Show the plot
    plt.show()

       
            
#histogramPlotting("aten::add", "CPU Mem")
#histogramPlotting("aten::mul", "CPU Mem")
#histogramPlotting("aten::mm", "CPU Mem")


"""
I need to present the data from the different experiments. Lets find all mins, maxs and avgs and medians for all of them. Lets just make a new script for that maybe?
Need some structure in the code.

"""