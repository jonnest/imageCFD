#!/bin/bash  
cd /datadrive/h0002_OF
blockMesh > log.blockMesh  
decomposePar -force > log.decomposePar 
renumberLog="log.renumberMesh" 
mpirun.openmpi -np 44 renumberMesh -overwrite -parallel > $renumberLog 
mpirun.openmpi -np 44 potentialFoam -parallel 
for ((i = 0; i<1001; ++i)); 
do 
   t1=$( bc -l <<<"0.01*$i") 
   t2=$( bc -l <<<"0.01*$((i+1))") 
   echo "$t1 , $t2 , $((i%25))" 
   python UpdateDispData.py $t2 $t1 $((i%100)) 44 
   LOGFILE1="log.pimpleFoamPar$i" 
   LOGFILE2="log.pimpleFoamPar${i}_${i}" 
    
   mpirun.openmpi -np 44 pimpleFoam -parallel > $LOGFILE1 
 
   if grep -Fq "$Finalising parallel run" $LOGFILE1 
    then  
    :  
    else  
    mpirun.openmpi -np 44 pimpleFoam -parallel > $LOGFILE2 
    fi 
   if ! ((($i +1) % 100)); then 
    python vtkclean.py  
    echo "Running cleaning procedure..." 
   fi  
   if ! ((($i+1) % 100)); then 
    python UpdateInflow.py 1.0  
    echo "Running cleaning procedure..." 
   fi 
done 
python VTK_merger.py 
python generatePhaseAverageIntegrated.py 
