#!/bin/bash
#SBATCH --partition=helio
#SBATCH --qos=helio_default
#SBATCH --account=helio
#SBATCH --mem=140G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=FAIL,REQUEUE,STAGE_OUT
#SBATCH --mail-user=joshin@mps.mpg.de
#SBATCH --output=logs/%x_slurm%j.log
#SBATCH --job-name=ic45_21
##SBATCH --ntasks=140
##SBATCH --nodelist=helio[43-51]
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=1
#SBATCH --exclude=helio[10,20,52]

# ref: https://slurm.schedmd.com/job_array.html
# ref: https://slurm.schedmd.com/sbatch.html#lbAH
# %x: --job-name
# %A: Job array's master job allocation number.
# %a: Job array ID (index) number.
# %j and %J give the same job id? (%j=%A+%a i guess)
# suggest %A_%a

# set up environment variables
source /etc/profile.d/modules.sh
module purge > /dev/null 2>&1
module load anaconda > /dev/null 2>&1
source activate lct
unset PYTHONPATH
#export PYTHONPATH="/data/seismo/joshin/opt/gcc-9.3.0/mpi4py-3.1.3_openmpi-4.0.5/lib/python3.6/site-packages:/data/seismo/joshin/pypkg"
#export PATH="/opt/netdrms/default/bin/linux_x86_64:$PATH" # for show_info on helio[56-57]
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export LD_LIBRARY_PATH=/opt/pgsql/default/lib:/opt/cfitsio/default/lib:$LD_LIBRARY_PATH
#export PATH=/opt/pgsql/default/bin:/opt/netdrms/default/bin/linux_avx:/opt/netdrms/default/scripts:/opt/cfitsio/default/bin:$PATH

which python
which mpirun

#mpirun -x OMP_NUM_THREADS -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH --bind-to none -H helio1:1,helio2:1 python main.py test.cfg -l debug
#mpirun -x OMP_NUM_THREADS -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH --bind-to none -H helio1:1,helio2:1 python main.py remap.cfg -l debug
srun python main_10years.py 2021 2022 -l debug
