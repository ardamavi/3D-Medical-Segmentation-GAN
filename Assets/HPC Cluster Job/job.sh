#!/bin/bash
## Arda Mavi
# For Help: http://login.kuacc.ku.edu.tr
# You should only work under the /scratch/users/<username> directory.
#
# TODO:
#   - Set name of the job below changing "Test" value.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter.
#   - Select the partition (queue) you want to run the job in:
#     - short : For jobs that have maximum run time of 120 mins. Has higher priority.
#     - long  : For jobs that have maximum run time of 7 days. Lower priority than short.
#     - longer: For testing purposes, queue has 31 days limit but only 3 nodes.
#   - Set the required time limit for the job with --time parameter.
#     - Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input and output file names below.
#   - If you do not want mail please remove the line that has --mail-type
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch examle_submit.sh

# -= Resources =-
#
#SBATCH --job-name=3D-Seg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=mid
#SBATCH --time=24:00:00
#SBATCH --mem=13GB
#SBATCH --gres gpu:1
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --partition=ai
#SBATCH --output=outputs.out
#SBATCH --mail-type=END
#SBATCH --mail-user=ardamavi2@gmail.com

## Load Python 3.6.6
echo "Activating Python 3.6.6..."
export PATH=/kuacc/users/lyo-amavi18/anaconda3/bin:$PATH

echo ""
echo "======================================================================================"

echo "Running Example Job...!"
echo "==============================================================================="
# Command 1 for matrix
echo "Running Python script..."
# Put Python script command below
python train.py
# Command 2 for matrix

# Command 3 for matrix
echo "Running compiled binary..."
# Put compiled binary command below
