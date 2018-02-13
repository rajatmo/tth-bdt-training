#!/bin/bash

###############################################
#
################################################

#for i in 0 1 2 3 4 5 6 7 8 9 10; do 
for i in 11 12 13 14 15 16 17 18 19 20; do
	python make_CSVprion_oneShot.py --process 'ttHToNonbb' --dofiles ${i} > ttHToNonbb_${i}.log & 
	python make_CSVprion_oneShot.py --process 'TTToSemilepton' --dofiles ${i} > TTToSemilepton_${i}.log &
	python make_CSVprion_oneShot.py --process 'TTWJetsToLNu' --dofiles ${i} > TTWJetsToLNu_${i}.log &
	python make_CSVprion_oneShot.py --process 'TTZToLLNuNu' --dofiles ${i} > TTZToLLNuNu_${i}.log & 


done
