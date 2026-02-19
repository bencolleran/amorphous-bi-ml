#!/bin/bash -p
#$ -cwd
#$ -pe smp 64
#$ -l s_rt=150:00:00
#$ -P highmem
#$ -j y
#$ -o $JOB_ID_$TASK_ID.log
 
 
module load mpi/openmpi-x86_64
 
conda activate autoplex
 
export rundir=/u/vld/sedm7085/test/fit
export train_file=/u/vld/sedm7085/test/fit/train.extxyz
 
echo ${rundir}
 
mkdir -p ${rundir}/1-train-DFT/${SGE_TASK_ID}
mkdir -p ${rundir}/2-pseudolabel
mkdir -p ${rundir}/3-train-SSL
 
cd ${rundir}/1-train-DFT/${SGE_TASK_ID}
 
gap_fit at_file=${train_file} gap={distance_Nb order=2 cutoff=5.0 n_sparse=15 covariance_type=ard_se delta=2.0 theta_uniform=0.5 sparse_method=uniform compact_clusters=T : distance_Nb order=3 cutoff=3.25 n_sparse=100 covariance_type=ard_se delta=2.0 theta_uniform=1.0 sparse_method=uniform compact_clusters=T : soap l_max=10 n_max=12 atom_sigma=0.5 cutoff=5.0 radial_scaling=0.0 cutoff_transition_width=1.0 central_weight=1.0 n_sparse=2000 delta=1.0 covariance_type=dot_product zeta=4 sparse_method=cur_points} default_sigma={0.0001 0.05 0.05 0} energy_parameter_name=REF_energy force_parameter_name=REF_forces sparse_jitter=1.0e-8 do_copy_at_file=F sparse_separate_file=T gp_file=GAP.xml e0={Bi:-695.12685957}