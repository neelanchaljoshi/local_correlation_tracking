#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# LCT Pipeline — embarrassingly-parallel SLURM submission (no MPI)
#
# Each array task processes exactly ONE time chunk (one dspan window —
# e.g. one day at dspan_hours=24, one hour at dspan_hours=1) for the
# given year/month, independently of every other task. No MPI: no
# module loads, no srun, no inter-task communication.
#
# Usage:
#   # 1. Find out how many chunks this month has:
#   python main_chunk.py config/granulation.ini 2019 6 --print-nchunks
#
#   # 2. Submit exactly that many array tasks:
#   sbatch --array=1-30 run_slurm_chunk.sh config/granulation.ini 2019 6
#
# Arguments:
#   $1  Path to the .ini config file
#   $2  Year to process
#   $3  Month to process (1-12)
#
# The SLURM array index ($SLURM_ARRAY_TASK_ID) provides the chunk number
# (1-indexed). A task whose chunk index is out of range for the month
# (e.g. day 31 of a 30-day month) exits cleanly with a warning, not a
# crash — see pipeline_chunk.resolve_chunk_bounds.
# ══════════════════════════════════════════════════════════════════════════════

#SBATCH --partition=swan
#SBATCH --qos=swan_default
#SBATCH --account=seismo
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --mail-type=FAIL,REQUEUE
#SBATCH --output=logs/%x_slurm_%A_%a.log
#SBATCH --job-name=lct_pipeline_chunk
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclude=swan[18,27,28]

# ── Validate arguments ───────────────────────────────────────────────────────
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "ERROR: Usage: sbatch --array=1-N run_slurm_chunk.sh <config_file> <year> <month>"
    exit 1
fi

CONFIG_FILE="$1"
YEAR="$2"
MONTH="$3"
CHUNK="${SLURM_ARRAY_TASK_ID:-1}"

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

# ── Log job info ─────────────────────────────────────────────────────────────
echo "========================================"
echo "Job:        $SLURM_JOB_NAME"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Config:     $CONFIG_FILE"
echo "Year:       $YEAR"
echo "Month:      $MONTH"
echo "Chunk:      $CHUNK"
echo "Started:    $(date)"
echo "========================================"

# ── Run (single process, no MPI) ─────────────────────────────────────────────
python -W ignore main_chunk.py "$CONFIG_FILE" "$YEAR" "$MONTH" \
    --chunk "$CHUNK" \
    --loglevel debug

EXIT_CODE=$?

echo "========================================"
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
