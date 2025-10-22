#!/bin/bash
#SBATCH --partition=swan
#SBATCH --qos=swan_default
#SBATCH --account=seismo
#SBATCH --mem=10G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=FAIL,INVALID_DEPEND,REQUEUE,STAGE_OUT,END
#SBATCH --mail-user=joshin@mps.mpg.de
#SBATCH --output=logs/%x_slurm%A_%a.log
#SBATCH --job-name=2017_360_030_diff_rot_5deg_gran_4k
#SBATCH --cpus-per-task=1
##SBATCH --exclude=swan[18,27,28]
#SBATCH --array=0-364

# ref: https://slurm.schedmd.com/job_array.html
# ref: https://slurm.schedmd.com/sbatch.html#lbAH
# %x: --job-name
# %A: Job array's master job allocation number.
# %a: Job array ID (index) number.
# %j and %J give the same job id? (%j=%A+%a i guess)
# suggest %A_%a

#module load Miniconda3

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/sw/eb/Miniconda3/23.9.0-0/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/sw/eb/Miniconda3/23.9.0-0/etc/profile.d/conda.sh" ]; then
        . "/sw/eb/Miniconda3/23.9.0-0/etc/profile.d/conda.sh"
    else
        export PATH="/sw/eb/Miniconda3/23.9.0-0/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate py311
source /usr/local/lmod/8.7.14/init/bash
module use /sw/eb/hmns/modules/all/Core
module purge
module load GCC/12.2.0
# export PMIX_MCA_psec=^munge
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TIME="\nET %E | CPU %P | Max %M KB max"

yr=${SLURM_JOB_NAME:0:4}
dspan=360
dstep=30
downsample=""
interpolate="--interp"

get_start_stop_from_index(){
    python3 <<END
from datetime import datetime, timedelta
index=$1
year=$2
start = datetime(year, 1, 1) + timedelta(days=index)
end = start + timedelta(days=1)
print(start.strftime("%Y.%m.%d_%H:%M:%S_TAI"))
print(end.strftime("%Y.%m.%d_%H:%M:%S_TAI"))
END
}

start_stop=( $(get_start_stop_from_index $SLURM_ARRAY_TASK_ID $yr) )
dstart="${start_stop[0]}"
dstop="${start_stop[1]}"

echo $dstart
echo $dstop

env time python -W ignore diff_rot_single_thread.py $dstart $dstop $dspan $dstep $downsample $interpolate -l debug
