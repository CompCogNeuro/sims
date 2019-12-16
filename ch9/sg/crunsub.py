#!/usr/local/bin/python3

# this generates a crun.sh for running this sim through slurm on cluster

import sys
import os

# project name is passed as first arg
crun_proj = sys.argv[1]

# jobid is passed as second arg
crun_jobid = sys.argv[2]

# crun_user is user name *ON THE SERVER* (cluster)
# may not be the same as local user!
crun_user = "oreillyr"

f = open('crun.sh', 'w+')
f.write("#!/bin/bash\n")
# f.write("#SBATCH --mem=4G\n") # amount of memory -- not always needed
f.write("#SBATCH --time=24:00:00\n")  # number of hours
f.write("#SBATCH -n 1\n")   # number of tasks total (mpi processors)
f.write("#SBATCH --ntasks-per-node=1\n")
f.write("#SBATCH --cpus-per-task=1\n")  # threads
f.write("#SBATCH --output=job.out\n") # expected output file
f.write("#SBATCH --qos=blanca-ccn\n")
f.write("#SBATCH --mail-type=FAIL\n")
f.write("#SBATCH --mail-user=" + crun_user + "\n")
f.write("#SBATCH --export=NONE\n")
f.write("unset SLURM_EXPORT_ENV\n")
f.write("\n\n")
f.write("go build\n")
f.write("date > job.start\n")
f.write("./" + crun_proj + " --nogui\n")
f.write("date > job.end\n")
f.flush()
f.close()


