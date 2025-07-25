#!/bin/bash
#SBATCH --partition=swan
#SBATCH --qos=swan_default
#SBATCH --account=seismo
#SBATCH --mem=200G
#SBATCH --time=2-00:40:00
#SBATCH --mail-type=FAIL,REQUEUE,STAGE_OUT,END
#SBATCH --mail-user=joshin@mps.mpg.de
#SBATCH --output=logs/%x_slurm_%a.log
#SBATCH --job-name=2019_06_2019_06_120_030_4k_sg_flow_010Res_2deg_ccfs_gran
#SBATCH --ntasks=500
##SBATCH --nodelist=helio[43-51]
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
##SBATCH --constraint=bigmem
#SBATCH --exclude=swan[18,27,28]
##SBATCH --array=1-12

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
module load GCC/12.2.0 OpenMPI/4.1.4
export PMIX_MCA_psec=^munge
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

yr_start=${SLURM_JOB_NAME:0:4}
month_start=${SLURM_JOB_NAME:5:2}
yr_end=${SLURM_JOB_NAME:8:4}
month_end=${SLURM_JOB_NAME:13:2}
dspan=${SLURM_JOB_NAME:16:3}
dstep=${SLURM_JOB_NAME:20:3}
downsample=1
interpolate=1
# month_start=$SLURM_ARRAY_TASK_ID
# echo $yr_start $month_start

# if [[ $month_start -eq 12 ]]
# then
#     yr_end=$(($yr_start+1))
#     month_end=1
#     echo "month_start is 12"
# else
#     yr_end=$yr_start
#     month_end=$(($month_start+1))
#     echo "month_start is not 12"
# fi

# yr_end= if [ $month_start -eq 12 ]; then echo $yr_start+1; else echo $yr_start; fi
# month_end= if [ $month_start -eq 12 ]; then echo 1; else echo $month_start+1; fi
# month_end=${SLURM_JOB_NAME:13:2}

echo $yr_start $month_start 
echo $yr_end $month_end

srun --mpi=pmix python -W ignore main_psf_interp_gran.py $yr_start $month_start $yr_end $month_end $dspan $dstep $downsample $interpolate -l debug