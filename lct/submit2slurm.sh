#!/bin/bash
#SBATCH --partition=swan
#SBATCH --qos=swan_default
#SBATCH --account=seismo
#SBATCH --mem=200G
#SBATCH --time=2-00:40:00
#SBATCH --mail-type=FAIL,REQUEUE,STAGE_OUT,END
#SBATCH --mail-user=joshin@mps.mpg.de
#SBATCH --output=logs/%x_slurm_%a.log
#SBATCH --job-name=2019_gran_lct
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

config_file="config_file.cfg"

srun --mpi=pmix python -W ignore main.py $config_file -l debug
