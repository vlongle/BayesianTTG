## To compile
 g++-7 *.cpp -I /usr/local/include/eigen3 -std=c++17 -O3 -fopenmp 




## Case study

ntrial = 20
SEED = 3

__exploitTrustSignalSoftmax__

trustLevel = 20 // very reliant on the trustLevel

min, max reward: 1 5.9
trials_mean: 147.50000000000003
reading beliefs from file  ./caseStudyData/cpp_softmax_exploit_beliefs_small.txt
agent 0 belief about 0 is [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
agent 0 belief about 1 is [2.06115e-009 4.24835e-018 8.75651e-027 1.80485e-035 2.40719e-042
 1.63509e-049 6.82598e-057 4.03281e-067 3.57235e-077 4.36710e-087
 6.93982e-097 4.37152e-101 8.65035e-100 4.45648e-099 6.93678e-099
 3.69842e-099 7.51474e-100 6.37675e-101 3.15774e-103 8.27362e-106]
agent 0 belief about 2 is [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
agent 1 belief about 0 is [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
agent 1 belief about 1 is [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
agent 1 belief about 2 is [2.06115e-009 4.24835e-018 8.75651e-027 1.80485e-035 3.58492e-040
 1.54358e-045 2.22999e-051 3.13644e-054 9.91618e-058 9.49889e-062
 3.42538e-066 1.81686e-070 3.98218e-075 4.50090e-080 2.86555e-085
 1.10409e-090 2.73115e-096 4.55639e-102 1.83542e-109 6.38990e-117]
agent 2 belief about 0 is [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
agent 2 belief about 1 is [2.06115e-09 4.24835e-18 8.75651e-27 1.80485e-35 8.54474e-39 6.18538e-43
 1.12416e-47 7.20769e-49 8.71667e-51 2.66818e-53 2.58270e-56 3.95099e-59
 2.10635e-62 5.02688e-66 5.91972e-70 3.72894e-74 1.34436e-78 2.93723e-83
 4.45838e-90 5.51965e-97]
agent 2 belief about 2 is [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
belief_strength_mean: [0.66666667 0.66666667 0.66666667 0.66666667 0.66666667 0.66666667
 0.66666667 0.66666667 0.66666667 0.66666667 0.66666667 0.66666667
 0.66666667 0.66666667 0.66666667 0.66666667 0.66666667 0.66666667
 0.66666667 0.66666667]
final payoff: 147.50000000000003
payoffs:
 [7.0, 7.0, 7.0, 7.0, 8.899999999999999, 7.0, 7.0, 8.899999999999999, 7.0, 7.0, 7.0, 8.900000000000006, 8.900000000000006, 7.0, 7.0, 7.0, 7.0, 7.000000000000014, 6.900000000000006, 7.0]
proposers:
 [1. 1. 1. 2. 2. 0. 1. 0. 0. 2. 1. 0. 2. 1. 1. 0. 0. 1. 2. 2.]
wealth:
 [[ 1.    3.    3.  ]
 [ 2.    6.    6.  ]
 [ 3.    9.    9.  ]
 [ 4.   12.   12.  ]
 [ 5.18 15.   16.72]
 [ 6.18 18.   19.72]
 [ 7.18 21.   22.72]
 [ 7.77 24.   28.03]
 [ 8.77 27.   31.03]
 [ 9.77 30.   34.03]
 [10.77 33.   37.03]
 [13.13 36.   40.57]
 [16.08 39.   43.52]
 [17.08 42.   46.52]
 [18.08 45.   49.52]
 [19.08 48.   52.52]
 [20.08 51.   55.52]
 [21.08 54.   58.52]
 [22.08 58.13 60.29]
 [23.08 61.13 63.29]]
 
 
 __Softmax with limiting Belief__
 
 trials_mean: 142.6
reading beliefs from file  ./caseStudyData/cpp_softmax_beliefs_small.txt
agent 0 belief about 0 is [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
agent 0 belief about 1 is [0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5
 0.5 0.5]
agent 0 belief about 2 is [0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5
 0.5 0.5]
agent 1 belief about 0 is [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
agent 1 belief about 1 is [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
agent 1 belief about 2 is [0.333333 0.333333 0.333333 0.333333 0.333333 0.333333 0.333333 0.333333
 0.333333 0.333333 0.333333 0.333333 0.333333 0.333333 0.333333 0.333333
 0.333333 0.333333 0.333333 0.333333]
agent 2 belief about 0 is [0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25
 0.25 0.25 0.25 0.25 0.25 0.25]
agent 2 belief about 1 is [0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25
 0.25 0.25 0.25 0.25 0.25 0.25]
agent 2 belief about 2 is [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
belief_strength_mean: [0.64814811 0.64814811 0.64814811 0.64814811 0.64814811 0.64814811
 0.64814811 0.64814811 0.64814811 0.64814811 0.64814811 0.64814811
 0.64814811 0.64814811 0.64814811 0.64814811 0.64814811 0.64814811
 0.64814811 0.64814811]
final payoff: 142.6
payoffs:
 [7.0, 7.0, 7.0, 7.0, 8.899999999999999, 7.0, 7.0, 8.899999999999999, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 5.900000000000006, 7.0, 7.0, 6.999999999999986, 6.900000000000006, 7.0]
proposers:
 [1. 1. 1. 2. 2. 0. 1. 0. 0. 2. 1. 0. 2. 1. 1. 0. 0. 1. 2. 2.]



## NOTE

Also, at the moment, the Gaussian pdf trust-signal updating is cumulative. We might not want to do that. Maybe, we only want to impose one layer of signal before action selection.




## CRITICAL!! DEBUG THIS

For beliefInjectionBulkExperiment(0.1)
there's appear to be inconsistency between trial = 1 and trial = 30
The number of trials shouldn't affect anything!
We are still using SEEDED random number generator


__RESOLVED__: This is an extremely pesky bug. I was passing the variable _tasks_ to _Game_ object by reference before. This is totally fine
for sequential program. But for parallel program, this is really bad due to race condition! Different threads would attempt to access _tasks_ to execute
the function _evaluateCoalition_ which can yields wrong result!!!
