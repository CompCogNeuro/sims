#!/usr/local/bin/python3

# this prints out files to be captured for simulation results

import glob

print("\n".join(glob.glob('*_epc.csv')))
print("\n".join(glob.glob('*_run.csv')))
    

