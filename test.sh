#!/bin/bash
#$ -cwd
#$ -pe smp 16
#$ -j y
#$ -P gpu
#$ -l s_rt=12:00:00
#$ -o $JOB_ID.log


################################################################
DIR=$(pwd)
module load aocc/3.2.0
module load aocl/3.2.0-aocc
module load mpi/openmpi-x86_64

function Cleanup ()
{
    trap "" SIGUSR1 EXIT SIGTERM SIGKILL # Disable trap now in it
    # Clean up task
    rsync -rlt $TMPDIR/* $DIR/
    exit 0
}
ulimit -s unlimited
trap Cleanup SIGUSR1 EXIT SIGTERM SIGKILL
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NMPI=$(expr $NSLOTS / $OMP_NUM_THREADS )

# Upload files to TMPDIR
INFILE='*' # copy infiles to the cluster (command line arguments here, modify as appropriate)
OUTFILE='*'
rsync -rlt $DIR/* $TMPDIR --exclude='*.log'
cd $TMPDIR # Use temporary directories to avoid i/o wastage from cluster to disk
################################################################
# Run stuff in here
echo "Running Job..."
python test_script.py
# mpirun -in lammps_in.in
#...

################################################################
PID=$!
while kill -0 $PID 2> /dev/null; do
    rsync -rltq --exclude '*.sh' --exclude '*.in' --exclude '*.out' --exclude 'ompi*' --exclude 'log.lammps' $TMPDIR/ $DIR/
    #sleep 600
done
wait $PID
rsync -rltq --exclude '*.in' --exclude '*.out' --exclude 'ompi*' --exclude 'log.lammps' ./ $DIR
#sleep 200
cd $DIR
pwd
