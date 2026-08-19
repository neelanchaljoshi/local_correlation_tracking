#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# LCT Pipeline — embarrassingly-parallel SLURM submission (no MPI)
#
# Each array task processes exactly ONE time chunk (one dspan window —
# e.g. one day at dspan_hours=24, one hour at dspan_hours=1),
# independently of every other task. No MPI: no module loads, no srun,
# no inter-task communication.
#
# Two modes — pass year/month for month mode, omit both for range mode
# (uses range_start/range_end set in the config's [job] section):
#
#   # Month mode: chunks span one calendar month
#   python main_chunk.py config/granulation.ini 2019 6 --print-nchunks
#   sbatch --array=1-30 run_slurm_chunk.sh config/granulation.ini 2019 6
#
#   # Range mode: chunks span range_start/range_end from the config —
#   # e.g. one day of hourly files as a clean --array=1-24, no
#   # day-offset arithmetic needed
#   python main_chunk.py config/one_day_hourly.ini --print-nchunks
#   sbatch --array=1-24 run_slurm_chunk.sh config/one_day_hourly.ini
#
# Arguments:
#   $1  Path to the .ini config file
#   $2  Year to process (month mode only — omit along with $3 for range mode)
#   $3  Month to process, 1-12 (month mode only)
#
# The SLURM array index ($SLURM_ARRAY_TASK_ID) provides the chunk number
# (1-indexed). A task whose chunk index is out of range (e.g. day 31 of
# a 30-day month, or beyond range_end) exits cleanly with a warning, not
# a crash — see pipeline_chunk.resolve_chunk_bounds/resolve_range_chunk_bounds.
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
if [ -z "$1" ]; then
    echo "ERROR: Usage: sbatch --array=1-N run_slurm_chunk.sh <config_file> [year month]"
    exit 1
fi
if [ -n "$2" ] && [ -z "$3" ]; then
    echo "ERROR: year given without month — pass both, or neither for range mode"
    exit 1
fi
if [ -z "$2" ] && [ -n "$3" ]; then
    echo "ERROR: month given without year — pass both, or neither for range mode"
    exit 1
fi

CONFIG_FILE="$1"
YEAR="${2:-}"
MONTH="${3:-}"
CHUNK="${SLURM_ARRAY_TASK_ID:-1}"

YEAR_MONTH_ARGS=()
if [ -n "$YEAR" ] && [ -n "$MONTH" ]; then
    YEAR_MONTH_ARGS=("$YEAR" "$MONTH")
fi

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
echo "Year:       ${YEAR:-<range mode>}"
echo "Month:      ${MONTH:-<range mode>}"
echo "Chunk:      $CHUNK"
echo "Started:    $(date)"
echo "========================================"

# ── Run (single process, no MPI) ─────────────────────────────────────────────
python -W ignore main_chunk.py "$CONFIG_FILE" "${YEAR_MONTH_ARGS[@]}" \
    --chunk "$CHUNK" \
    --loglevel debug

EXIT_CODE=$?

echo "========================================"
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
