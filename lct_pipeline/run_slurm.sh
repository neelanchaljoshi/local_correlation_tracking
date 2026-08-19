#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# LCT Pipeline — SLURM submission script
#
# Usage:
#   sbatch --array=1-12 run_slurm.sh config/granulation.ini 2019
#   sbatch --array=5-12 run_slurm.sh config/magnetic.ini 2010
#
# Arguments:
#   $1  Path to the .ini config file
#   $2  Year to process
#
# The SLURM array index ($SLURM_ARRAY_TASK_ID) provides the month (1–12).
# ══════════════════════════════════════════════════════════════════════════════

#SBATCH --partition=swan
#SBATCH --qos=swan_default
#SBATCH --account=seismo
#SBATCH --mem=200G
#SBATCH --time=2-00:40:00
#SBATCH --mail-type=FAIL,REQUEUE,STAGE_OUT,END
#SBATCH --mail-user=joshin@mps.mpg.de
#SBATCH --output=logs/%x_slurm_%A_%a.log
#SBATCH --job-name=lct_pipeline
#SBATCH --ntasks=1
##SBATCH --nodelist=helio[43-51]
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
##SBATCH --constraint=bigmem
#SBATCH --exclude=swan[18,27,28]
##SBATCH --array=1-12

# ── Validate arguments ───────────────────────────────────────────────────────
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "ERROR: Usage: sbatch --array=1-12 run_slurm.sh <config_file> <year>"
    exit 1
fi

CONFIG_FILE="$1"
YEAR="$2"
MONTH="${SLURM_ARRAY_TASK_ID:-1}"   # default to 1 if not running as array

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# ── Create logs directory ────────────────────────────────────────────────────
mkdir -p logs

# ── Conda environment ────────────────────────────────────────────────────────
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

# ── Modules ──────────────────────────────────────────────────────────────────
source /usr/local/lmod/8.7.14/init/bash
module use /sw/eb/hmns/modules/all/Core
module purge
module load GCC/12.2.0 OpenMPI/4.1.4

# ── MPI / OpenMP settings ────────────────────────────────────────────────────
export PMIX_MCA_psec=^munge
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ── Log job info ─────────────────────────────────────────────────────────────
echo "========================================"
echo "Job:        $SLURM_JOB_NAME"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Config:     $CONFIG_FILE"
echo "Year:       $YEAR"
echo "Month:      $MONTH"
echo "Nodes:      $SLURM_JOB_NODELIST"
echo "Tasks:      $SLURM_NTASKS"
echo "Started:    $(date)"
echo "========================================"

# ── Run ──────────────────────────────────────────────────────────────────────
srun --mpi=pmix python -W ignore main.py "$CONFIG_FILE" "$YEAR" \
    --month "$MONTH" \
    --loglevel debug

EXIT_CODE=$?

echo "========================================"
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
