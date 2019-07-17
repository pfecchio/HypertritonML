## Tutorial for running the two body analysis
If you want to run the analysis in one of the two centrality classes you have to:

1) Create a directory named HypertritonAnalysis in the same directory in which you have the HypertritonML package. The directory must have this structure:



![Structure](structure.png)


2) Put the HyperTritonTree.root files (both for data and Monte Carlo) in the subdirectory: Trees

3) Run the macros in /HypertritonML/2body/TreeGeneration: HyperTreeFatherData.root and HyperTreeFatherMC.root

4) Choose the centrality class for your analysis: 0-10% or 10-40%

5) Run the Uproot_conversion notebook

6) Run the Training_and_significance_scan notebook
