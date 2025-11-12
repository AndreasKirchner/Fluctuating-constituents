#!/bin/bash
#SBATCH --output=trento2d.out
#SBATCH --error=trento2d.err

#SBATCH --mem=20G

module load trento
module load HDF5

cd /hpc/group/qcd/abk66/work/Fluctuatingm
trento Pb Pb 1000000 -p 1.0 -w 1.0 -k 1.0 -x 6.4 -m 3 > trento_p1_w1_k1_m3.dat
trento Pb Pb 1000000 -p 1.0 -w 1.0 -k 1.0 -x 6.4 -m 4 > trento_p1_w1_k1_m4.dat
trento Pb Pb 1000000 -p 1.0 -w 1.0 -k 1.0 -x 6.4 -m 5 > trento_p1_w1_k1_m5.dat
trento Pb Pb 1000000 -p 1.0 -w 1.0 -k 1.0 -x 6.4 -m 10 > trento_p1_w1_k1_m10.dat
trento Pb Pb 1000000 -p 1.0 -w 1.0 -k 1.0 -x 6.4 -m 15 > trento_p1_w1_k1_m15.dat

