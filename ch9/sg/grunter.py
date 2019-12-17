#!/usr/local/bin/python3

# this is the grunt git-based run tool user-script: https://github.com/emer/grunt
# it must be checked into your local source repository and handles any user commands
# including mandatory submit and results commands.
#
# script is run in the jobs path:
# ~/grunt/wc/server/username/projname/jobs/active/jobid/projname
#
# this sample version includes slurm status and cancel commands

import sys
import os
import subprocess
from subprocess import Popen, PIPE
import glob
import datetime
import shutil
import re
import getpass

##############################################################
# key job parameters here, used in writing the job.sbatch

# max number of hours -- slurm will terminate if longer, so be generous
# 2d = 48, 3d = 72, 4d = 96, 5d = 120, 6d = 144, 7d = 168
hours = 24  

# memory per task
mem = "4G"

# number of mpi "tasks" (procs in MPI terminology)
tasks = 1

# number of cpu cores (threads) per task
cpus_per_task = 1

# how to allocate tasks within compute nodes
# cpus_per_task * tasks_per_node <= total cores per node
tasks_per_node = 1

# qos is the queue name
qos = "blanca-ccn"

##############################################################
# main vars

# grunt_jobpath is current full job path:
# ~/grunt/wc/server/username/projname/jobs/active/jobid/projname
grunt_jobpath = os.getcwd()

# grunt_user is user name (note: this is user *on the server*)
grunt_user = getpass.getuser()
# print("grunt_user: " + grunt_user)

# grunt_proj is the project name
grunt_proj = os.path.split(grunt_jobpath)[1]

##############################################################
# utility functions

def write_string(fnm, stval):
    with open(fnm,"w") as f:
        f.write(stval + "\n")

def read_string(fnm):
    # reads a single string from file and strips any newlines -- returns "" if no file
    if not os.path.isfile(fnm):
        return ""
    with open(fnm, "r") as f:
        val = str(f.readline()).rstrip()
    return val

def read_strings(fnm):
    # reads multiple strings from file, result is list and strings still have \n at end
    if not os.path.isfile(fnm):
        return []
    with open(fnm, "r") as f:
        val = f.readlines()
    return val

def read_strings_strip(fnm):
    # reads multiple strings from file, result is list of strings with no \n at end
    if not os.path.isfile(fnm):
        return []
    with open(fnm, "r") as f:
        val = f.readlines()
        for i, v in enumerate(val):
            val[i] = v.rstrip()
    return val

def timestamp():
    return str(datetime.datetime.now())

# write_sbatch writes the job submission script: job.sbatch
def write_sbatch():
    args = " ".join(read_strings_strip("job.args"))
    f = open('job.sbatch', 'w')
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --mem=" + mem + "\n") # amount of memory -- not always needed
    f.write("#SBATCH --time=" + str(hours) + ":00:00\n") 
    f.write("#SBATCH --ntasks=" + str(tasks) + "\n")
    f.write("#SBATCH --cpus-per-task=" + str(cpus_per_task) + "\n")
    f.write("#SBATCH --ntasks-per-node=" + str(tasks_per_node) + "\n")
    f.write("#SBATCH --qos=" + qos + "\n")
    f.write("#SBATCH --output=job.out\n")
    f.write("#SBATCH --mail-type=FAIL\n")
    f.write("#SBATCH --mail-user=" + grunt_user + "\n")
    f.write("#SBATCH --export=NONE\n")
    f.write("unset SLURM_EXPORT_ENV\n")
    f.write("\n\n")
    f.write("go build\n")  # add anything here needed to prepare code
    f.write("date > job.start\n")
    f.write("./" + grunt_proj + " --nogui + " + args + "\n")
    f.write("date > job.end\n")
    f.flush()
    f.close()
    
def submit():
    write_sbatch()
    try:
        result = subprocess.check_output(["sbatch","job.sbatch"])
    except subprocess.CalledProcessError:
        print("Failed to submit job.sbatch script")
        return
    prog = re.compile('.*Submitted batch job (\d+).*')
    result = prog.match(str(result))
    slurmid = result.group(1)
    write_string("job.slurmid", slurmid)
    print("submitted successfully -- slurm job id: " + slurmid)

def results():
    # important: update this to include any results you want to add to results repo
    print("\n".join(glob.glob('*_epc.csv')))
    print("\n".join(glob.glob('*_run.csv')))

def status():
    slid = read_string("job.slurmid")
    stat = "NOSLURMID"
    if slid == "" or slid == None:
        print("No slurm id found -- maybe didn't submit properly?")
    else:    
        print("slurm id to stat: ", slid)
        result = ""
        try:
            result = subprocess.check_output(["squeue","-j",slid,"-o","%T"], universal_newlines=True)
        except subprocess.CalledProcessError:
            print("Failed to stat job")
        res = result.splitlines()
        if len(res) == 2:
            stat = res[1].rstrip()
        else:
            stat = "NOTFOUND"
    print("status: " + stat)
    write_string("job.status", stat)
    
def cancel():
    write_string("job.canceled", timestamp())
    slid = read_string("job.slurmid")
    if slid == "" or slid == None:
        print("No slurm id found -- maybe didn't submit properly?")
        return
    print("canceling slurm id: ", slid)
    try:
        result = subprocess.check_output(["scancel",slid])
    except subprocess.CalledProcessError:
        print("Failed to cancel job")
        return

if len(sys.argv) < 2 or sys.argv[1] == "help":
    print("\ngrunter.py is the git-based run tool extended run script\n")
    print("supports the following commands:\n")
    print("submit\t submit job to slurm")
    print("results\t list job results")
    print("status\t get current slurm status")
    print("cancel\t tell slurm to cancel job")
    print()
    exit(0)
    
cmd = sys.argv[1]

if cmd == "submit":
    submit()
elif cmd == "results":
    results()
elif cmd == "status":
    status()
elif cmd == "cancel":
    cancel()
else:
    print("grunter.py: error: cmd not recognized: " + cmd)
    exit(1)
exit(0)

