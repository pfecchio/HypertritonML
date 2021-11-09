import os

# os.system("python3 run_analysis.py ../Config/2body_analysis.yaml -t -a -s")

os.system('python3 signal_extraction.py ../Config/2body_analysis.yaml -dbshape -s')
os.system('python3 signal_extraction.py ../Config/2body_analysis.yaml -s')

os.system('python3 compute_blambda.py ../Config/2body_analysis.yaml -dbshape -s -syst')
os.system('python3 compute_blambda.py ../Config/2body_analysis.yaml -s -syst')

os.system('python3 compute_lifetime.py ../Config/2body_analysis.yaml -s -syst')