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

# cd /home/acaan/VHbbNtuples_8_0_x/CMSSW_7_4_7/src/ ; cmsenv ; cd -
# python do_limits.py --channel "2lss_1tau" --variables "oldVarA"  --BINtype "quantiles"
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--channel ", type="string", dest="channel", help="The ones whose variables implemented now are:\n   - 1l_2tau\n   - 2lss_1tau\n It will create a local folder and store the report*/xml", default="2lss_1tau")
parser.add_option("--variables", type="string", dest="variables", help="  Set of variables to use -- it shall be put by hand in the code", default=1000)
parser.add_option("--BDTtype", type="string", dest="BDTtype", help="Variable set", default="1B")
parser.add_option("--BINtype", type="string", dest="BINtype", help="regular / ranged / quantiles", default="regular")
(options, args) = parser.parse_args()

user="acaan"
year="2016"
channel=options.channel

if channel == "2lss_1tau" :
    label='2lss_1tau_2018Feb28_VHbb_TLepMTau_shape'
    bdtTypes=["tt","ttV","SUM_T","SUM_M","1B_T","1B_M"] #,"1B"] #,"2MEM","2HTT"]
if channel == "1l_2tau" :
    label= "1l_2tau_2018Mar02_VHbb_TLepTTau_shape" #"1l_2tau_2018Feb08_VHbb_TightTau" # "1l_2tau_2018Feb02_VHbb_VTightTau" #  "1l_2tau_2018Jan30_VHbb_VVTightTau" # "1l_2tau_2018Jan30_VHbb" # "1l_2tau_2018Jan30_VHbb_VTightTau" #
    bdtTypes=["ttbar","ttV","SUM_T","SUM_VT","1B_T","1B_VT"] #"1B"] #
if channel == "2l_2tau" :
    label= "2l_2tau_2018Feb20_VHbb_TLepMTau" #
    bdtTypes= ["tt","ttV","SUM_M","SUM_T","SUM_VT","1B_M","1B_T","1B_VT"] #
if channel == "3l_1tau" :
    label= "3l_1tau_2018Mar12_VHbb_TLepMTau_shape" #
    bdtTypes= ["tt","ttV","SUM_M","SUM_T","SUM_VT","1B_M","1B_T","1B_VT"] #[] #,,
if channel == "2017" :
    label="datacards_ICHEP"
    #bdtTypes= ["HTT_SUM_VT", "plainKin_SUM_M", "plainKin_SUM_VT",  "plainKin_SUM_VT"]
    #channelsTypes= ["1l_2tau", "3l_1tau", "1l_2tau", "2l_2tau"]
    bdtTypes= ["plainKin_SUM_VT"] #, "HTT_SUM_M", "plainKin_SUM_M"]
    channelsTypes= [ "2l_2tau"] #, "2lss_1tau", "2lss_1tau"]
    #
    #bdtTypes= ["plainKin_SUM_VT", "HTT_SUM_VT"]
    #channelsTypes= ["1l_2tau", "1l_2tau"]
    #
    #bdtTypes= ["plainKin_SUM_M"]
    #channelsTypes= ["3l_1tau"]
    # prepareDatacards_2lss_1tau_sumOS_mvaOutput_2lss_1tau_HTT_SUM_M_noRebin.root

sources=[]
bdtTypesToDo=[]
bdtTypesToDoFile=[]
channelToDo=[]
local=workingDir+"/"+options.channel+"_"+label+"/"+options.variables+"/"

print ("to run this script your CMSSW_base should be the one that CombineHavester installed")

def run_cmd(command):
  print "executing command = '%s'" % command
  p = subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
  stdout, stderr = p.communicate()
  print stderr
  return stdout

nbinRegular=np.arange(3,10) #list(divisorGenerator(originalBinning))
nbinQuant= np.arange(1,13)

counter=0
if channel == "2lss_1tau" :
    source=local+"/prepareDatacards_"+channel+"_sumOS_"
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
    bdtTypesToDoFile=bdtTypesToDo
if channel == "1l_2tau" or channel == "2l_2tau" or channel == "3l_1tau" :
    source=local+"/prepareDatacards_"+channel+"_"
    if options.variables=="oldTrain" :
        oldVar=["1l_2tau_ttbar_Old", "1l_2tau_ttbar_OldVar","ttbar_OldVar"] #["1l_2tau_ttbar"] #"1l_2tau_ttbar_Old", "1l_2tau_ttbar_OldVar", "ttbar_OldVar"]
        for ii,nn in enumerate(oldVar) :
            my_file = source+"mvaOutput_"+nn+'.root'
            print my_file
            if os.path.exists(my_file) :
                sources = sources + [source+"mvaOutput_"+nn]
                bdtTypesToDo = bdtTypesToDo +['1D']
                bdtTypesToDoFile=bdtTypesToDoFile+[oldVar[ii]]
                print ("rebinning ",sources[counter])
            else : print ("does not exist ",source+"mvaOutput_"+nn+".root")
    elif "HTT" in options.variables :
        for bdtType in bdtTypes :
            #if channel == "2l_2tau" :
            fileName=options.variables+"_"+bdtType
            #elif channel == "1l_2tau" : fileName=bdtType+"_"+options.variables
            my_file =  source+"mvaOutput_"+fileName+".root"
            if os.path.exists(my_file) :
                sources = sources + [source+"mvaOutput_"+fileName]
                bdtTypesToDo = bdtTypesToDo +[bdtType]
                bdtTypesToDoFile=bdtTypesToDoFile+[fileName]
                print (sources[counter],"rebinning ")
                counter=counter+1
            else : print (source+fileName+".root","does not exist ")
    elif "mTauTauVis" in options.variables :
        my_file = source+options.variables+".root"
        if os.path.exists(my_file) :
            proc=subprocess.Popen(["cp "+my_file+" " +local],shell=True,stdout=subprocess.PIPE)
            out = proc.stdout.read()
            sources = sources + [source+options.variables]
            bdtTypesToDo = bdtTypesToDo +[options.variables]
            bdtTypesToDoFile=bdtTypesToDoFile+[options.variables]
            print (sources[counter],"rebinning ")
            counter=counter+1
        else : print (source+options.variables+".root","does not exist ")
    else : print ("options",channel,options.variables,"are not compatible")
if channel=="2017" :
    for ii, bdtType in enumerate(bdtTypes) :
        source=local+"/prepareDatacards_"+channelsTypes[ii]+"_mvaOutput_"+bdtTypes[ii]+"_noRebin"
        my_file = source+".root"
        if os.path.exists(my_file) :
            #proc=subprocess.Popen(["cp "+source+options.variables+".root " +local],shell=True,stdout=subprocess.PIPE)
            #out = proc.stdout.read()
            sources = sources + [source]
            bdtTypesToDo = bdtTypesToDo +[channelsTypes[ii]+"_mvaOutput_"+bdtTypes[ii]]
            bdtTypesToDoFile=bdtTypesToDoFile+[channelsTypes[ii]+"_mvaOutput_"+bdtTypes[ii]]
            channelToDo=channelToDo+[channelsTypes[ii]]
            print (sources[counter],"rebinning ")
            counter=counter+1
        else : print (source+options.variables+".root","does not exist ")

if options.BINtype == "regular" or options.BINtype == "ranged" : binstoDo=nbinRegular
if options.BINtype == "quantiles" : binstoDo=nbinQuant

print ("I will rebin",bdtTypesToDoFile,"(",len(sources),") BDT options")

"""
combineTool.py  -M Asymptotic -m 125 -t -1 %s
 (do ws)
#

text2workspace.py
mkdir GoF; cd GoF
combineTool.py  -M T2W -i ../ttH_noHTT_noHTT_SUM_M_nbin_5_quantiles.txt
combine -M MaxLikelihoodFit -d ../ttH_noHTT_noHTT_SUM_M_nbin_5_quantiles.root -t -1
python $CMSSW_BASE/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py -a mlfit.root -g plots.root
combineTool.py -M GoodnessOfFit --algorithm saturated -d ../ttH_noHTT_noHTT_SUM_M_nbin_5_quantiles.root -n .saturated
combineTool.py -M GoodnessOfFit --algorithm saturated -d ../ttH_noHTT_noHTT_SUM_M_nbin_5_quantiles.root -n .saturated.toys -t 200 -s 0:4:1 --parallel 5
combineTool.py -M CollectGoodnessOfFit --input higgsCombine.saturated.GoodnessOfFit.mH120.root higgsCombine.saturated.toys.GoodnessOfFit.mH120.*.root -o GoF_saturated.json
$CMSSW_BASE/src/CombineHarvester/CombineTools/scripts/plotGof.py --statistic saturated --mass 120.0 GoF_saturated.json -o GoF_saturated

combineTool.py  -M T2W -i /home/acaan/ttHAnalysis/2016/1l_2tau_2018Mar02_VHbb_TLepTTau_shape/datacards/1l_2tau/ttH_mvaOutput_1l_2tau_ttbar_Old_5bins.txt


combineTool.py  -M T2W -i ../ttH_noHTT_noHTT_SUM_M_12bins_quantiles_fullsyst.txt
# noHTT_SUM_M_5bins_quantiles_fullsyst

mkdir impacts; cd impacts
combineTool.py -M Impacts -m 125 -d ../ttH_noHTT_noHTT_SUM_M_nbin_5_quantiles.root --expectSignal 1 --allPars --parallel 8 -t -1 --doInitialFit
combineTool.py -M Impacts -m 125 -d ../ttH_noHTT_noHTT_SUM_M_nbin_5_quantiles.root --expectSignal 1 --allPars --parallel 8 -t -1 --robustFit 1 --doFits
combineTool.py -M Impacts -m 125 -d ../ttH_noHTT_noHTT_SUM_M_nbin_5_quantiles.root -o impacts.json
plotImpacts.py -i impacts.json -o  impacts
"""

file = open("execute_plots"+options.channel+"_"+options.variables+".sh","w")
file.write("#!/bin/bash\n")
for ns,source in enumerate(sources) :
    for nn,nbins in enumerate(binstoDo) :
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
        shapeVariable=options.variables+'_'+bdtTypesToDoFile[ns]+'_nbin_'+str(nbins)
        if options.BINtype=="ranged" : shapeVariable=shapeVariable+"_ranged"
        if options.BINtype=="quantiles" : shapeVariable=shapeVariable+"_quantiles"
        datacardFile_output = os.path.join(workingDir, local, "ttH_%s.root" % shapeVariable)
        #run_cmd('%s --input_file=%s --output_file=%s --add_shape_sys=false' % ('WriteDatacards_'+channel, name, datacardFile_output))
        run_cmd('%s --input_file=%s --output_file=%s --add_shape_sys=false' % ('WriteDatacards_'+channelToDo[ns], name, datacardFile_output))

        txtFile = datacardFile_output.replace(".root", ".txt")
        logFile = datacardFile_output.replace(".root", ".log")
        run_cmd('combine -M Asymptotic -m %s -t -1 %s &> %s' % (str(125), txtFile, logFile))
        run_cmd('rm higgsCombineTest.Asymptotic.mH125.root')
        rootFile = os.path.join(workingDir, local, "ttH_%s_shapes.root" % (shapeVariable))
        run_cmd('PostFitShapes -d %s -o %s -m 125 ' % (txtFile, rootFile))
        ##### 2lss_1taumvaOutput_2lss_1tau_ttV_nbin_6

        makeplots='root -l -b -n -q /home/acaan/VHbbNtuples_8_0_x/CMSSW_7_4_7/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\\"'+shapeVariable+'\\",\\"'+local+'\\",\\"'+options.channel+'\\",\\"'+local+'\\")'
        file.write(makeplots+ "\n")
file.close()
