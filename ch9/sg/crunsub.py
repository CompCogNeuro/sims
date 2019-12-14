#!/usr/local/bin/python3

# this generates a crun.sh for running this sim through slurm on cluster

import sys
import os

# project name is passed as first arg
crun_proj = sys.argv[1]

# jobid is passed as second arg
crun_jobid = sys.argv[2]

# crun_root is path to crun *ON THE SERVER* (cluster) starting at home dir
crun_root = "crun"

# crun_user is user name *ON THE SERVER* (cluster)
# may not be the same as local user!
crun_user = "oreillyr"

# crun_jobs is the jobs git working dir for project
crun_jobs = os.path.join(crun_root, "wd", crun_user, crun_proj, "jobs")

# new_job is the full path to new job
# turns out we are automatically placed there so we don't need this!
new_job = os.path.join(crun_jobs, "active", crun_jobid)

f = open('crun.sh', 'w+')
f.write("#!/bin/bash\n")
# f.write("#SBATCH --mem=4G\n") # amount of memory -- not always needed
f.write("#SBATCH --time=24:00:00\n")  # number of hours
f.write("#SBATCH -n 1\n")   # number of tasks total (mpi processors)
f.write("#SBATCH --ntasks-per-node=1\n")
f.write("#SBATCH --cpus-per-task=1\n")  # threads
f.write("#SBATCH --qos=blanca-ccn\n")
f.write("#SBATCH --mail-type=FAIL\n")
f.write("#SBATCH --mail-user=" + crun_user + "\n")
f.write("#SBATCH --export=NONE\n")
f.write("unset SLURM_EXPORT_ENV\n")
f.write("\n\n")
# f.write("cd " + new_job + "\n")
f.write("go build\n")
f.write("date > job.start\n")
f.write("./" + crun_proj + " --nogui\n")
f.write("date > job.end\n")
f.flush()
f.close()
    

