import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
# Define the column names
columns = ['Self CPU %', 'Self CPU', 'CPU total %', 'CPU total', 'CPU time avg', 'CPU Mem',
           'Self CPU Mem', '# of Calls', 'Total KFLOPs']

#These have 100 and 50 neurons but only 10 rows out for profiling
file_path1 = "/diehlAndCookExperiments/cleanedOutFiles/diehlAndCookStandardCPUClean.out"
file_path2 = "/diehlAndCookExperiments/cleanedOutFiles/diehlAndCookReducedCPUClean.out"
file_path3 = "/diehlAndCookExperiments/cleanedOutFiles/diehlAndCookHebbianCPUClean.out"
file_path4 = "/diehlAndCookExperiments/cleanedOutFiles/diehlAndCookWDSTDPCPUClean.out"

#These have 100 and 50 neurons but they now have 25 rows out
file_path5 = "/Results_25Rows/diehlAndCookStandard_nograd_25rowsClean.out"
file_path6 = "/Results_25Rows/diehlAndCookReduced_nograd_25rowsClean.out"
file_path7 = "/Results_25Rows/diehlAndCookHebbian_nograd_25rowsClean.out"
file_path8 = "/Results_25Rows/diehlAndCookWDSTDP_nograd_25rowsClean.out"

#These have 100 neurons but adjusted some other params. 25 rows profiling out
file_path9 = "/Results_25Rows/diehlAndCookStandardAdjInhib_nograd_25rowsClean.out"
file_path10 = "/Results_25Rows/diehlAndCookStandardAdjThresh_nograd_25rowsClean.out"

#These have 1600 neurons, only 10 rows from profiling
file_path11 = "/largeNetworkDiehlResults/diehlAndCookReduced_nogradClean.out"
file_path12 = "/largeNetworkDiehlResults/diehlAndCookHebbian_nogradClean.out"
file_path13 = "/largeNetworkDiehlResults/diehlAndCookWDSTDP_nogradClean.out"



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

#small parts of the functionality in this function was written with use of chatgpt.
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


        print("Length ", len(dataList))
        print("length one", len(dataList[0]))
        for values in dataList:

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




        test_list1 = CPU_mem_add
        test_list2 = aten_operator_column_vectors["aten::add"]["CPU Mem"]
        test_list1.sort()
        test_list2.sort()

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
    varArr = []
    stdArr = []
    if columnName == "CPU Mem":
        if operationName == "aten::add":
            for arra in operator_column_vectors_collected: #For every network dictionary with operator values in lists
                arr = arra["aten::add"]["CPU Mem"] #Find the network key addition operation and also the column for cpu memory. This should all memory measurements for additions for this network.
                #looks like this is the issue, that it is only a value not a list.
                
                medArr.append(np.median(np.array(arr)))
                minArr.append(min(arr))
                maxArr.append(max(arr))
                avgArr.append(np.average(np.array(arr)))
                varArr.append(np.var(np.array(arr)/1000))
                stdArr.append(np.std(np.array(arr)/1000))

                #print(np.array(arr)[0:30])
                #print(arr[0:30])
                #print(np.var(np.array(arr)))
        elif operationName == "aten::mul":
            for arra in operator_column_vectors_collected:
                arr = arra["aten::mul"]["CPU Mem"]
                medArr.append(np.median(np.array(arr)))
                minArr.append(min(arr))
                maxArr.append(max(arr))
                avgArr.append(np.average(np.array(arr)))
                varArr.append(np.var(np.array(arr)/1000))
                stdArr.append(np.std(np.array(arr)/1000))
        elif operationName == "aten::mm":
            for arra in operator_column_vectors_collected:
                arr = arra["aten::mm"]["CPU Mem"]
                medArr.append(np.median(np.array(arr)))
                minArr.append(min(arr))
                maxArr.append(max(arr))
                avgArr.append(np.average(np.array(arr)))
                varArr.append(np.var(np.array(arr)/1000))
                stdArr.append(np.std(np.array(arr)/1000))
    return medArr, avgArr,minArr, maxArr, varArr, stdArr


medianAdd, averageAdd, minimumAdd, maximumAdd, varianceAdd, stdDevAdd = medAvgMinMax("aten::add", "CPU Mem")
print("medians: ", medianAdd, "\n Averages: ", averageAdd, " \nMinimums: ", minimumAdd, "\nMaximums: ", maximumAdd, "\nVariances: ", varianceAdd, "\nStandard deviations: ", stdDevAdd)
print("medians in Kb: ", np.array(medianAdd) / 1000, "\n Averages in Kb: ", np.array(averageAdd) / 1000, " \nMinimums in Kb: ", np.array(minimumAdd) / 1000, "\nMaximums in Kb: ", np.array(maximumAdd) / 1000)

def barPlotting(medians, averages, minimums, maximums, operationName, columnName):
    titleName = ""
    if columnName == "CPU Mem":
        titleName = "CPU Memory"
    else:
        titleName = columnName
    titleOfPlotMedian = "Bar plot of the median values for the " + titleName + " for the " + operationName + " operation for several different network configurations"
    titleOfPlotAverage = "Bar plot of the average values for the " + titleName + " for the " + operationName + " operation for several different network configurations"
    titleOfPlotMinimum = "Bar plot of the minimum values for the " + titleName + " for the " + operationName + " operation for several different network configurations"
    titleOfPlotMaximum = "Bar plot of the maximum values for the " + titleName + " for the " + operationName + " operation for several different network configurations"
    categories = ["Standard100", "Standard50", "Hebbian100", "WDSTDP100", "Adjusted inhibition100", "Adjusted threshold100", "Standard1600", "Hebbian1600", "WDSTDP1600"]


    ax = sns.barplot(x=categories, y=medians)
    ax.bar_label(ax.containers[0])
    # Create custom legend
    legend_labels = categories
    legend_colors = [patch.get_facecolor() for patch in ax.patches]
    legend_patches = [Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
    # Display custom legend with specified labels and colors
    plt.legend(handles=legend_patches, title="Network topology", loc='upper left')
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
    plt.legend(handles=legend_patches, title="Network topology", loc='upper left')
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
    plt.legend(handles=legend_patches, title="Network topology", loc='upper left')
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
    plt.legend(handles=legend_patches, title="Network topology", loc='upper left')
    plt.title(titleOfPlotMaximum)
    plt.axis('off')
    plt.show()



#barPlotting([x / 1000 for x in medianAdd], [x / 1000 for x in averageAdd], [x / 1000 for x in minimumAdd], [x / 1000 for x in maximumAdd], "aten::add", "CPU Mem")

#
#medianMul, averageMul, minimumMul, maximumMul, varianceMul, stdDevMul = medAvgMinMax("aten::mul", "CPU Mem")
#print("medians: ", medianMul, "\n Averages: ", averageMul, " \nMinimums: ", minimumMul, "\nMaximums: ", maximumMul)
#barPlotting([x / 1000 for x in medianMul], [x / 1000 for x in averageMul], [x / 1000 for x in minimumMul], [x / 1000 for x in maximumMul], "aten::mul", "CPU Mem")
#
#medianMm, averageMm, minimumMm, maximumMm, varianceMM, stdDevMm = medAvgMinMax("aten::mm", "CPU Mem")
#print("medians: ", medianMm, "\n Averages: ", averageMm, " \nMinimums: ", minimumMm, "\nMaximums: ", maximumMm)
#barPlotting([x / 1000 for x in medianMm], [x / 1000 for x in averageMm], [x / 1000 for x in minimumMm], [x / 1000 for x in maximumMm], "aten::mm", "CPU Mem")




"""
Dette er sånn det skal funke:
Du tar ett nettverk av gangen
Så tar du og henter data fra fil korresponderende til det nettverket.
Så tar du vektoren med alle dataene som nå ligger i en liste og plasser det inn i en ny dictionary som har operasjonsnavnet som key og hver listeverdi inn under riktig kolonne
Da har man samlet alle operatorverdiene fra alle de ulike operatorene i en fin dictionary
Så lagrer vi den dictionaryen i en liste som holder dictionaryene for alle de ulike nettverkene.
"""

"""
Okay now I want to make a few plots that indicate number of calls for these so that I can show how many operations are needed in software
Since this is my most hard cold data to be used.
I can first plot the histograms then num of calls.
"""
#print(operator_column_vectors_collected[1]["aten::add"]["CPU Mem"])
def histogramPlotting(operationName, columnName, numBins = 20, scaling=1):
    titleOfPlot = 'Histogram of Memory use by the ' + operationName + ' operation for several different configurations of the Diehl & Cook SNN'
    colors = ['red', 'green', 'blue', 'orange','purple', 'brown', 'olive', 'cyan','pink']
    categories = [f"Standard100", f"Standard50", f"Hebbian100", f"WDSTDP100", f"Adjusted inhibition100", f"Adjusted threshold100", f"Standard1600", f"Hebbian1600", f"WDSTDP1600"]
    tmp = operator_column_vectors_collected[0][operationName][columnName]
    binArr = np.arange(min(tmp)/1000, max(tmp)/1000+0.001, step=0.001)
    for i in range(len(filenames)):
        plt.hist(np.array(operator_column_vectors_collected[i][operationName][columnName])/scaling, bins=binArr, alpha=0.5, color=colors[i], label=categories[i])
        #print("uten ",np.array(operator_column_vectors_collected[i][operationName][columnName]))
        #print("med", np.array(operator_column_vectors_collected[i][operationName][columnName])/1000)
        #print("med variabel", np.array(operator_column_vectors_collected[i][operationName][columnName])/scaling)
    
    # Set labels and legend
    plt.xlabel('Memory use in Kb')
    plt.ylabel('# of samples within the bins')
    plt.title(titleOfPlot)
    plt.legend()
    
    # Show the plot
    plt.show()

#histogramPlotting("aten::add", "CPU Mem",numBins= 1,scaling=1000)
#histogramPlotting("aten::mul", "CPU Mem",scaling=1000)
#histogramPlotting("aten::mm", "CPU Mem",scaling=1000)


"""
Okei her legges til litt number of calls
"""

def barPlottingNumCalls(operationName, columnName):
    titleName = ""
    if columnName == "# of Calls":
        titleName = "number of calls"
    else:
        titleName = columnName
    titleOfPlot = "Bar plot of the "  + titleName + " for the " + operationName + " operation for several different network configurations"
    categories = ["Standard100", "Standard50", "Hebbian100", "WDSTDP100", "Adjusted inhibition100", "Adjusted threshold100", "Standard1600", "Hebbian1600", "WDSTDP1600"]
    y_List = []
    for arra in operator_column_vectors_collected:
        networkCallList = arra[operationName][columnName]
        #print(type(networkCallList[0]))
        #print(networkCallList[0])
        y_List.append(np.sum(np.array(networkCallList)))
        print("---")
        print(np.sum(np.array(networkCallList)))
    ax = sns.barplot(x=categories, y=y_List)
    ax.bar_label(ax.containers[0])
    # Create custom legend
    legend_labels = categories
    legend_colors = [patch.get_facecolor() for patch in ax.patches]
    legend_patches = [Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
    # Display custom legend with specified labels and colors
    plt.legend(handles=legend_patches, title="Network topology", loc='upper left')
    plt.title(titleOfPlot)
    plt.axis('off')
    plt.show()

barPlottingNumCalls("aten::add", '# of Calls')
barPlottingNumCalls("aten::mul", '# of Calls')
barPlottingNumCalls("aten::mm", '# of Calls')



def barPlottingFLOPS(operationName, columnName):
    titleName = ""
    if columnName == "Total KFLOPs":
        titleName = "Total number of K-floating point operations"
    else:
        titleName = columnName
    titleOfPlot = "Bar plot of the "  + titleName + " for the " + operationName + " operation for several different network configurations"
    categories = ["Standard100", "Standard50", "Hebbian100", "WDSTDP100", "Adjusted inhibition100", "Adjusted threshold100", "Standard1600", "Hebbian1600", "WDSTDP1600"]
    y_List = []
    for arra in operator_column_vectors_collected:
        networkKFLOPList = arra[operationName][columnName]
        #print(type(networkCallList[0]))
        #print(networkCallList[0])
        y_List.append(np.sum(np.array(networkKFLOPList)))
        print("---")
        print(np.sum(np.array(networkKFLOPList)))
    ax = sns.barplot(x=categories, y=y_List)
    ax.bar_label(ax.containers[0])
    # Create custom legend
    legend_labels = categories
    legend_colors = [patch.get_facecolor() for patch in ax.patches]
    legend_patches = [Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
    # Display custom legend with specified labels and colors
    plt.legend(handles=legend_patches, title="Network topology", loc='upper left')
    plt.title(titleOfPlot)
    plt.axis('off')
    plt.show()

barPlottingNumCalls("aten::add", "Total KFLOPs")
barPlottingNumCalls("aten::mul", "Total KFLOPs")
barPlottingNumCalls("aten::mm", "Total KFLOPs")

