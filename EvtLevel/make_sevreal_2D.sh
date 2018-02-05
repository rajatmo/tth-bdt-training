#!/bin/bash

# arrays on parameters
#declare -a Nstart=(270 300 350 400 450 500 550 600 650 700 750 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400 2500 2600 2700 2800 2900 3000)

Nstart=8 # 15 , 20 , 25
for (( i = 14 ; i <40 ; i++ )); do
  python map_2D_evtLevel_ttH.py --channel '1l_2tau' --variables "HTT" --nbins-start 20 --nbins-target $i &
done
