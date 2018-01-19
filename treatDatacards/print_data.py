#!/usr/bin/env python
import os, subprocess, sys
workingDir = os.getcwd()
from array import array
from ROOT import *
from math import sqrt, sin, cos, tan, exp
import itertools as it
import glob

#
#procP1=glob.glob("/hdfs/local/acaan/ttHAnalysis/2016/2018Jan17_VHbb_addMEM/histograms/2lss_1tau/Tight_SS_OS/*Run2016*/*.root")
#procP1=glob.glob("/hdfs/local/acaan/ttHAnalysis/2016/2018Jan18_VHbb_addMEM/histograms/2lss_1tau/histograms_harvested_stage1_2lss_1tau_*_Run2016*_Tight_lepSS_sumOS.root")


procP1=glob.glob("/hdfs/local/acaan/ttHAnalysis/2016/2018Jan18_VHbb_addMEM/histograms/2lss_1tau/Tight_SS_OS/SingleMuon_Run2016H_v2_promptReco/SingleMuon_Run2016H_v2_promptReco_Tight_lepSS_sumOS_central_*.root")
alldata=0
alldata1=0
for ii in procP1 :
    #print ii
    file = TFile(ii,"READ");
    file.cd()
    #file.cd("2lss_1tau_lepSS_sumOS_Tight/sel/evt/data_obs")
    #file.cd("2lss_1tau_lepSS_sumOS_Tight/sel/evt/data_obs")
    histo=file.Get("2lss_1tau_lepSS_sumOS_Tight/sel/evt/data_obs/mvaOutput_2lss_oldVarA_tt")
    histo1=file.Get("2lss_1tau_lepSS_sumOS_Tight/sel/evt/data_obs/oldVarA_from20_to_10")
    #for keyO in file.GetListOfKeys() :
    #   obj =  keyO.ReadObj()
    #   if type(obj) is not TH1F : continue
    #   #if "_to_" in keyO.GetName() :
    if histo.Integral() != histo1.Integral() : print ii
    print (histo.GetName(),histo.Integral(),histo1.GetName(),histo1.Integral())
    alldata=alldata+histo.Integral()
    alldata1=alldata1+histo1.Integral()
print ("sum of data",alldata,alldata1)

# /hdfs/local/acaan/ttHAnalysis/2016/2018Jan17_VHbb_addMEM/histograms/2lss_1tau/histograms_harvested_stage1_2lss_1tau_SingleMuon_Run2016H_v2_promptReco_Tight_lepSS_sumOS.root
#mvaOutput_2lss_oldVarA_tt->Integral()
#(double) 2.000000
#root [8] oldVarA_from20_to_10->Integral()
#(double) 0.000000
