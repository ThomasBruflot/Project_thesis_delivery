# Project_thesis_delivery
GitHub repository holding code and support scripts related to the my project thesis December 2023. 

Repo structure:
There are several files in the repo used for either profiling, running simulations on HPC clusters or for doing analysis of the profiling data. 

The two files located under "Profiling_analysis_scripts" contains analysis code for extracting information from the profiling data. 

The "diehlAndCookExperiments" folder contains code for the Diehl & Cook networks with profiling from the pytorch profiler added. These were ran on IDUN. The folder also contains some support scripts for some analysis of the data, as well as slurm scripts configuring the simulations on IDUN. The Diehl & Cook networks are implemented in BindsNet and then altered by me with different configurations.

The "Misc_slurm_scripts" folder contains different versions of the slurm scripts used for running simulations. 

"FilesToSigma contains network code and profiling code for the Diehl & Cook network but with configurations that work only on the Sigma2 cluster. The Sigma2 and IDUN HPC clusters use different packages and modules to run code, and therefore libraries in python is installed in the command line through the python script.

"Previous_iterations_of_profiling_analysis_scripts" contains like the name suggests, previous versions of the analysis code. 

The MemEstUnsupSTDP.py file holds the Dong et al. network with profiling added. The source code here is from the BrainCog repo and is not made by me.

