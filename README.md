# FAMD
code of the paper "FAMD: A Few-Shot Android Family Detection Framework"
## Dataset
[drebin](https://www.sec.cs.tu-bs.de/~danarp/drebin/download.html) [CIC-InvesAndMal2019](https://www.unb.ca/cic/datasets/invesandmal2019.html)    
The folder dataset contains the processed CSV file in an unobfuscated state.   
The obfuscation tool is AvPass:https://github.com/sslab-gatech/avpass.
## Train phase
Based on contrastive learning, triplet. siamese.py
## Test phase
Based on our customized prototype detection method, combined with ensemble learning. ensemble_learning.py.
## Other
caculate.py: the implementation of the algorithm mentioned in the paper.
