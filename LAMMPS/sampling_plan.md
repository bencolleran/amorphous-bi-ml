I would like to sample the most representative data based on how dissimilar they are from the other data points in the trajectory
From my PCA tests, the initial density sits firmly inside the training set so i should definitely increase and decrease the density
I have already done, 0.512 (0.8), 0.729 (0.9) and 3.375 (1.5)
I should instead do 0.8,0.9,1.0,1.1,1.2
I will do FPs with my training set and the combined set of all my structures so 4000 new structuers
I will choose maybe 500 structures?

new
do castep calculations on 3 random strudtres with 1x1x1 2x2x2 3x3x3
use all 128 cores cpu