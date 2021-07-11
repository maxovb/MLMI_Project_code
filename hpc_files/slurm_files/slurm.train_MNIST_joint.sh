#!/bin/bash
#SBATCH -J MLMI_Project_code
#SBATCH -A MLMI-mov22-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --mail-type=FAIL
#SBATCH --array=0-7
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH -p pascal
#! ############################################################
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel7/default-gpu              # REQUIRED - loads the basic environment
module load cuda/10.2 intel/mkl/2017.4
source /home/mov22/rds/hpc-work/MLMI_Project_code/hpc_files/activate_environment/README.Thesis_code.activate
export OMP_NUM_THREADS=1

STEPARRAY=(100 100 100 100 100 100 100 100)
CLARRAY=(False True False True False True False True)
ETARRAY=(False False True True False False True True)
GNARRAY=(False False False False True True True True)
STEP=${STEPARRAY[$SLURM_ARRAY_TASK_ID]}
CL=${CLARRAY[$SLURM_ARRAY_TASK_ID]}
ET=${ETARRAY[$SLURM_ARRAY_TASK_ID]}
GN=${GNARRAY[$SLURM_ARRAY_TASK_ID]}

WDIR=/home/mov22/rds/hpc-work/MLMI_Project_code
TDIR=${WDIR}/hpc_files
LOG=${TDIR}/log.$JOBID
ERR=${TDIR}/stderr.$JOBID

##
mkdir -p $TDIR
JOBID=$SLURM_JOB_ID.${SLURM_ARRAY_TASK_ID}


echo -e "JobID: $JOBID\n======" > $LOG
echo "Time: `date`" >> $LOG
echo "Running on master node: `hostname`" >> $LOG
echo $CL > $LOG

python $WDIR/train_MNIST_joint.py $STEP --consitencyloss $CL --extratask $ET --gradnorm $GN


