#!/bin/bash

#SBATCH --ntasks=120
#SBATCH --time=120:00:00
#SBATCH --job-name=ecco
#SBATCH --tmp=5G               # 5G , 4000
#SBATCH --mem-per-cpu=5G       # 1G , 2000

decomposePar -force > log.decomposePar

for ((i = 0; i<1001; ++i)); 
do 
    LOGFILE1="log.pimpleFoamPar$i" 
    
    if grep -Fq "$Finalising parallel run" $LOGFILE1 
        then 
            :
        else
            mpirun -np 120 pimpleFoam -parallel > $LOGFILE1 
    fi
done


reconstructPar -newTimes

rm -rf processor*

