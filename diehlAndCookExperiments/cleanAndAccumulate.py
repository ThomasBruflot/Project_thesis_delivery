#This code was partially built using chatGPT
path1 = "Results_25Rows/pyMemLongAnalysis.out"
path2 = "Results_25Rows/pyMemLongAnalysisClean.out"
with open(path1, 'r') as input_file, open(path2, 'w') as output_file:
    for line in input_file:
        if not line.startswith('STAGE:'):
            output_file.write(line)