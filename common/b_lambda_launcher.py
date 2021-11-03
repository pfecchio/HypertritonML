import os

os.system("python3 run_analysis.py ../Config/2body_ct.yaml -t -a -s")
os.system("python3 separation_energy_2body.py ../Config/2body_ct.yaml -s -syst")