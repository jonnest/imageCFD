#!/bin/bash

#SBATCH --ntasks=18
#SBATCH --time=120:00:00
#SBATCH --job-name=v54
#SBATCH --tmp=5G               # 5G , 4000
#SBATCH --mem-per-cpu=5G       # 1G , 2000

decomposePar -force > log.decomposePar

mpirun -np 18 pimpleFoam -parallel > log.pimpleFoam

reconstructPar -newTimes

rm -rf processor*