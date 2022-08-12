import os

# os.system("python3 run_analysis.py ../Config/2body_analysis_upd.yaml -t -a -s")
# os.system("python3 run_analysis.py ../Config/2body_analysis_upd.yaml -t -a -s --antimatter")


# os.system('python3 signal_extraction.py ../Config/2body_analysis_upd.yaml -dbshape -s')
# os.system('python3 signal_extraction.py ../Config/2body_analysis_upd.yaml -s')

# os.system('python3 compute_blambda.py ../Config/2body_analysis_upd.yaml -dbshape -s -syst')
# os.system('python3 compute_blambda.py ../Config/2body_analysis_upd.yaml -s -syst')
os.system('python3 compute_lifetime.py ../Config/2body_analysis_upd.yaml -s ')


# os.system("python3 run_analysis.py ../Config/2body_analysis_new.yaml -t -a -s --antimatter")
# os.system("python3 run_analysis.py ../Config/2body_analysis_B.yaml -t -a -s --matter")



# os.system('python3 compute_lifetime.py ../Config/2body_analysis_B.yaml -dbshape -s -syst --antimatter')

# os.system('python3 signal_extraction.py ../Config/2body_analysis_B.yaml -s -dbshape --matter')
# os.system('python3 signal_extraction.py ../Config/2body_analysis_B.yaml -s -dbshape --antimatter')

# os.system('python3 compute_blambda.py ../Config/2body_analysis_B.yaml -s -dbshape --matter --syst')
# os.system('python3 compute_blambda.py ../Config/2body_analysis_B.yaml -s -dbshape --antimatter --syst')

# os.system('python3 compute_blambda.py ../Config/2body_analysis_B.yaml -s --matter --syst')
# os.system('python3 compute_blambda.py ../Config/2body_analysis_B.yaml -s --antimatter --syst')