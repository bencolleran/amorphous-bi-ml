#!/bin/bash
#$ -cwd
#$ -pe smp 128
#$ -l s_rt=24:00:00
#$ -j y
#$ -o $JOB_ID.log
#$ -N bismuth_MD
#$ -P cpu

 
# sets the working directory to the one submitted from
# pe smp = number of cores. Our 2 nodes, accessed via omega, have 48 cores each.
# s_rt = soft run-time limit
# -j y merge stderr and stdout
# -o choose $JOB_ID.log as the outfile
 
# set PATH, LD_LIBRARY_PATH etc. for the intel libraries used to compile lammps
module load aocc/3.2.0
module load aocl/3.2.0-aocc
#module load mpi/openmpi-x86_64
module load openmpi4.1.6

export DIR=$(pwd)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/
 
#run in lammps
 #../../../spet5633/lmp-files/lmp-exe/lmp-init-11_22
 #lmp_exe_versions.txt  lmp-init-11_22  lmp-keat-160323  lmp-kim-010823  lmp-mace-050225  lmp-mtp-71222
 #lmp_exec=$DIR/../../../spet5633/lmp-files/lmp-exe/lmp-mace-050225 # LAMMPS executable binary file
#---------------------- Bits you may need to change ---------------------------------
 
# all input files and the lammps executable should be in the project directory
 
restart=data # choose either data or continuation as starting structure for run
lmp_in=$DIR/in_nvt # lammps infile, for constant 'NVT' ensemble MD
lmp_exec=$DIR/../../../applications/lammps-installs/autoplex/lammps/build/lmp # LAMMPS executable binary file
 
# Starting structure - provided by qsub -v STRUCTURE=... in the run script
s=$DIR/Bi_diamond_64.data
 
# Name of the run directory that gets created - based on simulation parameters captured in unique_key
rundir="$(date +"%Y%m%d_%H%M%S")"           
#-----------------------------------------------------------------------------------
 
# this takes care of copying back your data from the compute node when the job finishes
function Cleanup ()
{
    trap "" SIGUSR1 EXIT # Disable trap now we're in it
    # Clean up task
    rsync -rltq $TMPDIR/ $DIR/
    exit 0
}
trap Cleanup SIGUSR1 EXIT # Enable trap
 
# some environment variables for parallelisation and memory usageage
# LAMMPS mainly uses MPI parallelisation (at least with QUIP), so we
# 'turn off' the OpenMP parallelisation by setting the number of threads
# to 1 (also for the intel Math Kernel Library (MKL) that handles matrix
# operations etc.)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NMPI=$(expr $NSLOTS / $OMP_NUM_THREADS )
export GFORTRAN_UNBUFFERED_ALL=y
ulimit -s unlimited
 
mkdir -p ${rundir}/NVT
mkdir -p ${rundir}/restart
cp "$s" "${rundir}/"
 
# some lammps settings (units, setting up the potential)
system=Bi
model=gap
units=metal
## the pairstyle below is currently re-defined in the in_nvt input, so these will do nothing
pot1=$DIR/library.meam # filename for the GAP potential that you use for the simulation
pot2=$DIR/Bi.meam
# make sure that the 'sparseX' file(s) are in the same directory as the .xml
pair_style=meam/c
pair_coeff="* * ${pot1} Bi ${pot2} Bi"
 
INFILE="${rundir}"
rsync -rltq $INFILE $TMPDIR/
cd $TMPDIR  # use the temporary directory local to the compute node. This avoids
# writing output over the network filesystem, which is slow for you and slows down
# the NFS for everyone else (especially as jobs get larger)
 
mpirun -np $NMPI $lmp_exec -in ${lmp_in} \
   -var system ${system} \
   -var units ${units} \
   -var pair_style "${pair_style}" \
   -var pair_coeff "${pair_coeff}" \
\
   -var model ${model} \
   -var rundir ${rundir} \
   -var restart_from ${restart} \
   -var data_file "$s" &
 
pid=$! # copy back job data every 10 seconds while we wait for it to finish
while kill -0 $pid 2> /dev/null; do
    sleep 300
    rsync -rltq $TMPDIR/ $DIR/
done
wait $pid
 
cd $DIR
mv $JOB_ID.log $DIR/$rundir/
 