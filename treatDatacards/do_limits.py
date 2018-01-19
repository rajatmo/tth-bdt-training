#!/usr/bin/env python
import os, subprocess, sys
from array import array
from ROOT import *
from math import sqrt, sin, cos, tan, exp
import numpy as np
workingDir = os.getcwd()

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from optparse import OptionParser
parser = OptionParser()
parser.add_option("--channel ", type="string", dest="channel", help="The ones whose variables implemented now are:\n   - 1l_2tau\n   - 2lss_1tau\n It will create a local folder and store the report*/xml", default="2lss_1tau")
parser.add_option("--variables", type="string", dest="variables", help="  Set of variables to use -- it shall be put by hand in the code", default=1000)
parser.add_option("--BDTtype", type="string", dest="BDTtype", help="Variable set", default="1B")
parser.add_option("--BINtype", type="string", dest="BINtype", help="regular / ranged / quantiles", default="regular")
(options, args) = parser.parse_args()

user="acaan"
channel='2lss_1tau'
label='2018Jan13_VHbb_addMEM'
year="2016"
source=workingDir+"/"+options.channel+"_"+label+"/"+options.variables+"/prepareDatacards_"+channel+"_sumOS_"
local=workingDir+"/"+options.channel+"_"+label+"/"+options.variables+"/"
bdtTypes=["tt","ttV","1B","2MEM","2HTT"]
sources=[]
bdtTypesToDo=[]

print ("to run this script your CMSSW_base should be the one that CombineHavester installed")
import shutil,subprocess
proc=subprocess.Popen(['cd /home/acaan/VHbbNtuples_8_0_x/CMSSW_7_4_7/src/ ; cmsenv ; cd -'],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()

def run_cmd(command):
  print "executing command = '%s'" % command
  p = subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
  stdout, stderr = p.communicate()
  print stderr
  return stdout

nbinRegular=np.arange(1,20)
nbinQuant= np.arange(1,20)
counter=0
if options.variables=="MEM" :
    #source= source+"memOutput_LR"+".root"
    my_file = source+'memOutput_LR.root'
    if os.path.exists(my_file) :
        sources = sources + [source+'memOutput_LR']
        bdtTypesToDo = bdtTypesToDo +['1D']
        print ("rebinning ",sources[counter])
    else : print ("does not exist ",source+"memOutput_LR.root")
else :
    for bdtType in bdtTypes :
        my_file = source+'mvaOutput_2lss_'+options.variables+'_'+bdtType+'.root'
        if os.path.exists(my_file) :
            sources = sources + [source+'mvaOutput_2lss_'+options.variables+'_'+bdtType]
            bdtTypesToDo = bdtTypesToDo +[bdtType]
            print (sources[counter],"rebinning ")
            counter=counter+1
        else : print (source+"mvaOutput_2lss_"+options.variables+"_"+bdtType+".root","does not exist ")

if options.BINtype == "regular" or options.BINtype == "ranged" : binstoDo=nbinRegular
if options.BINtype == "quantiles" : binstoDo=nbinQuant

folder=options.channel+"_"+label+"/"+options.variables+"/"
file = open("execute_plots"+options.channel+"_"+options.variables+".sh","w")
file.write("#!/bin/bash\n")
for ns,source in enumerate(sources) :
    for nn,nbins in enumerate(nbinQuant) :
        if options.BINtype=="regular" :
            name=source+'_'+str(nbins)+'bins.root'
            nameout=source+'_'+str(nbins)+'bins_dat.root'
        if options.BINtype=="ranged" :
            name=source+'_'+str(nbins)+'bins_ranged.root'
            nameout=source+'_'+str(nbins)+'bins_ranged_dat.root'
        if options.BINtype=="quantiles" :
            name=source+'_'+str(nbins+1)+'bins_quantiles.root'
            nameout=source+'_'+str(nbins+1)+'bins_quantiles_dat.root'
        print ("doing", name)
        shapeVariable=options.variables+'_'+bdtTypesToDo[ns]+'_nbin_'+str(nbins)
        if options.BINtype=="ranged" : shapeVariable=shapeVariable+"_ranged"
        if options.BINtype=="quantiles" : shapeVariable=shapeVariable+"_quantiles"
        datacardFile_output = os.path.join(workingDir, folder, "ttH_%s.root" % shapeVariable)
        run_cmd('%s --input_file=%s --output_file=%s --add_shape_sys=false' % ('WriteDatacards_'+channel, name, datacardFile_output))
        txtFile = datacardFile_output.replace(".root", ".txt")
        logFile = datacardFile_output.replace(".root", ".log")
        run_cmd('combine -M Asymptotic -m %s -t -1 %s &> %s' % (str(125), txtFile, logFile))
        run_cmd('rm higgsCombineTest.Asymptotic.mH125.root')
        rootFile = os.path.join(workingDir, folder, "ttH_%s_shapes.root" % (shapeVariable))
        run_cmd('PostFitShapes -d %s -o %s -m 125 ' % (txtFile, rootFile))
        ##### 2lss_1taumvaOutput_2lss_1tau_ttV_nbin_6

        makeplots='root -l -b -n -q /home/acaan/VHbbNtuples_8_0_x/CMSSW_7_4_7/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\\"'+shapeVariable+'\\",\\"'+folder+'\\",\\"'+options.channel+'\\",\\"'+local+'\\")'
        file.write(makeplots+ "\n")
file.close()
