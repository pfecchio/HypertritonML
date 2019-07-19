## Tutorial for running the two body analysis
If you want to run the analysis in one of the two centrality classes you have to:

1) Put the HyperTritonTree.root files (both for data and Monte Carlo) in the same directory
2) Edit the file "config.sh" adding the path to your Trees
3) Type in your terminal : source config.sh

3) Run the macros in /HypertritonML/2body/TreeGeneration: HyperTreeFatherData.root and HyperTreeFatherMC.root.The first argument of the macros is the name of your trees.

4) Choose the centrality class for your analysis: 0-10% or 10-40%

6) Run the Training notebook
