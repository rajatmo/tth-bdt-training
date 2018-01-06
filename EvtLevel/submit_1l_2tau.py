#!/usr/bin/env python
import os, re
import commands
import math, time
import sys

print 
print 'START'
print 
########   YOU ONLY NEED TO FILL THE AREA BELOW   #########
########   customization  area #########
queue = "1nh" # give bsub queue -- 8nm (8 minutes), 1nh (1 hour), 8nh, 1nd (1day), 2nd, 1nw (1 week), 2nw 
###

########   customization end   #########

path = os.getcwd()
path2 ="/home/acaan/CMSSW_9_4_0_pre1/src/"
print
os.system("mkdir tmp")
print
channel="1l_2tau"
bdtTypes=("evtLevelTTV_TTH", "evtLevelTT_TTH")
vars =("allVar", "notForbidenVar","noHadTopTaggerVar")

##### loop for creating and sending jobs #####
for bdtType in bdtTypes:
  for var in vars: 
   ##### creates directory and file list for job #######
   os.system("mkdir tmp/training_"+str(bdtType)+"_model"+str(var))
   os.chdir("tmp/training_"+str(bdtType)+"_model"+str(var))
   ##### creates jobs #######
   command="python sklearn_Xgboost_csv_evtLevel_ttH.py --channel "+str(channel)+" --variables "+str(var)+" --bdtType "+str(bdtType)+" --doXML \n"
   print (command)
   with open('job.sh', 'w') as fout:
      fout.write("#!/bin/sh\n")
      fout.write("echo\n")
      fout.write("echo\n")
      fout.write("echo 'START---------------'\n")
      fout.write("echo 'WORKDIR ' ${PWD}\n")
      #fout.write("source /afs/cern.ch/cms/cmsset_default.sh\n")
      fout.write("cd "+str(path)+"\n")
      fout.write("cmsenv\n")
      fout.write("export PYTHONUSERBASE=/home/acaan/python_local/\n")
      fout.write("export PATH=/home/acaan/python_local/bin:$PATH\n")
      fout.write("export PYTHONPATH=/home/acaan/python_local/lib/python2.7/site-packages:$PYTHONPATH\n")
      fout.write(command)
      fout.write("echo 'STOP---------------'\n")
      fout.write("echo\n")
      fout.write("echo\n")
   os.system("chmod 755 job.sh")
   
   ###### sends bjobs ######
   # https://twiki.cern.ch/twiki/pub/Main/BatchJobs/submitJobs.py.txt
   # https://twiki.cern.ch/twiki/bin/view/Main/BatchJobs#MultiJobSub
   # https://wiki.med.harvard.edu/Orchestra/IntroductionToLSF#Default_memory_requirements
   os.system("bsub  -q "+queue+" -R \"rusage[mem=30096]\" -o logs job.sh -f") # -f datacard_SMTraining_kl_-18_kt_1.0.txt
   #os.system("bsub -q "+queue+" -o logs -f \"*Training"+str(training)+"_kl_"+str(sam_list[j])+"_kt_1.0.txt <\" job.sh")
   print "job for exp = "+str(bdtType)+" model "+str(var) +" submitted"
   
   os.chdir("../..")
   
print
print "your jobs:"
os.system("bjobs")
print
print 'END'
print

