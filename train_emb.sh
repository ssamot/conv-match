#!/bin/bash

if [ "$HOSTNAME" = "ApolloII" ]; then
    printf '%s\n' "This is the laptop"
	export OMP_NUM_THREADS=2
	export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,compute_test_value='ignore'
else
	printf '%s\n' "This is the the big server machine"
	export OMP_NUM_THREADS=5
	export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,compute_test_value='ignore',allow_gc=True,lib.cnmem=0.9
fi


#python experiment.py --memory --rnn --hops 1
#python experiment.py --memory --rnn --hops 2
#python experiment.py --memory --rnn --hops 3
python experiment.py --memory --hops 3  --runid 10000 
#python experiment.py --memory --hops 2
#python experiment.py --memory --hops 3

