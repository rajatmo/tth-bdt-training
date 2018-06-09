#!/usr/bin/env python
import os, subprocess, sys
from array import array
import CombineHarvester.CombineTools.ch as ch
from ROOT import *
from math import sqrt, sin, cos, tan, exp
import numpy as np
workingDir = os.getcwd()
#from pathlib2 import Path
execfile("../python/data_manager.py")

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# cd /home/acaan/VHbbNtuples_8_0_x/CMSSW_7_4_7/src/ ; cmsenv ; cd -
# python do_limits.py --channel "2lss_1tau" --uni "Tallinn"
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--channel ", type="string", dest="channel", help="The ones whose variables implemented now are:\n   - 1l_2tau\n   - 2lss_1tau\n It will create a local folder and store the report*/xml", default="2lss_1tau")
parser.add_option("--uni", type="string", dest="uni", help="  Set of variables to use -- it shall be put by hand in the code", default="Tallinn")
(options, args) = parser.parse_args()

doLimits = False
doImpacts = False
doYields = True
doGOF = False
doPlots = False

blinded=True
#prepareDatacards_1l_2tau_mvaOutput_final.root  prepareDatacards_2lss_1tau_sumOS_mvaOutput_final.root
#prepareDatacards_2l_2tau_mvaOutput_final.root  prepareDatacards_3l_1tau_mvaOutput_final.root
channel = options.channel
university = options.uni
if university == "Tallinn":
    mom = "/home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun09/"
    local = "Tallinn/"
    card_prefix = "prepareDatacards_"
    cards = [
    "1l_2tau_mvaOutput_final_x",
    "2lss_1tau_sumOS_mvaOutput_final_x",
    "2l_2tau_mvaOutput_final_x",
    "3l_1tau_mvaOutput_final_x"
    ]
    folders = [
    "ttH_1l_2tau/",
    "",
    "ttH_2l_2tau/",
    "ttH_3l_1tau/"
    ]
elif university == "Cornell":
    mom = "/home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun05/"
    local = "Cornell/ch/"
    card_prefix = "datacards_"
    cards = [
    "1l2tau_41p53invfb_Binned_2018jun04",
    "2lss1tau_41p53invfb_Binned_2018jun04",
    "2l2tau_41p53invfb_Binned_2018jun04",
    "3l1tau_41p53invfb_Binned_2018jun04"
    ]

channels = [
"1l_2tau",
"2lss_1tau",
"2l_2tau",
"3l_1tau"
]

wdata = [
"",
"_FRjt_syst",
"",
"_FRjt_syst"
]

dolog = [
"true",
"false",
"false",
"false"
]

min = [
0.1,
0.0,
0.0,
0.0
]

max = [
2000,
15.0,
15.0,
2.0
]

hasConversions = [
"true",
"true",
"false",
"false"
]

hasFlips = [
"false",
"true",
"false",
"true"
]

print ("to run this script your CMSSW_base should be the one that CombineHavester installed")

#####################################################################
datacardToRun=[]
for nn, card in enumerate(cards) :
    if not channel == channels[nn] : continue
    #if nn == 0 : continue
    my_file = mom+local+card_prefix+card+'.root'
    if os.path.exists(my_file) :
        print ("testing ", my_file)
        if channel=="3l_1tau" :
            print ("testing ", my_file)
            file = TFile(my_file,"READ");
            my_file = mom+local+card_prefix+card+'_noNeg.root'
            file2 = TFile(my_file,"RECREATE");
            file2.cd()
            h2 = TH1F()
            for keyO in file.GetListOfKeys() :
               obj =  keyO.ReadObj()
               if type(obj) is not TH1F : continue
               h2=obj.Clone()
               for bin in range (0, h2.GetXaxis().GetNbins()) :
                   if h2.GetBinContent(bin) < 0 :
                       print keyO
                       print h2.GetBinContent(bin)
                       h2.AddBinContent(bin, abs(h2.GetBinContent(bin))+0.0001)
               h2.Write()
            file2.Close()

        # make txt datacard
        datacard_file=my_file
        datacardFile_output = mom+local+"ttH_"+card+".root"
        run_cmd('%s --input_file=%s --output_file=%s --add_shape_sys=true' % ('WriteDatacards_'+channels[nn]+wdata[nn], my_file, datacardFile_output))

        #
        txtFile = datacardFile_output.replace(".root", ".txt")
        logFile = datacardFile_output.replace(".root", ".log")
        logFileNoS = datacardFile_output.replace(".root", "_noSyst.log")
        if doLimits :
            run_cmd('combine -M AsymptoticLimits -m %s -t -1 --run blind -S 0 %s &> %s' % (str(125), txtFile, logFileNoS))
            run_cmd('combine -M AsymptoticLimits -m %s -t -1 --run blind %s &> %s' % (str(125), txtFile, logFile))
            run_cmd('rm higgsCombineTest.AsymptoticLimits.mH125.root')

        if doPlots :
            filesh = open(mom+local+"execute_plots"+channels[nn]+"_"+university+".sh","w")
            filesh.write("#!/bin/bash\n")
            rootFile = mom+local+"ttH_"+card+"_shapes.root"
            run_cmd('PostFitShapes -d %s -o %s -m 125 ' % (txtFile, rootFile))
            makeplots=('root -l -b -n -q /home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\\"'
            +str(card)+'\\",\\"'+str(local)+'\\",\\"'+str(channels[nn])+'\\",\\"'+str(mom)+'\\",'+str(dolog[nn])+','+str(hasFlips[nn])+','+hasConversions[nn]+',\\"BDT\\",\\"Events\\",'+str(min[nn])+','+str(max[nn])+')')
            #root -l -b -n -q /home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"2lss_1tau_sumOS_mvaOutput_2lss_1tau_HTT_SUM_M_11bins_quantiles\",\"2018jun02/\",\"2lss_1tau\",\"/home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/\",false,false,\"BDT\",\"\",0.0,10.0)
            filesh.write(makeplots+ "\n")
            print ("to have the plots take the makePlots command from: ",mom+local+"execute_plots"+channels[nn]+"_"+university+".sh")

        if doImpacts :
            run_cmd('combineTool.py  -M T2W -i %s' % (txtFile))
            run_cmd('combineTool.py -M Impacts -m 125 -d %s  --expectSignal 1 --allPars --parallel 8 -t -1 --doInitialFit' % (datacardFile_output))
            run_cmd('combineTool.py -M Impacts -m 125 -d %s  --expectSignal 1 --allPars --parallel 8 -t -1 --robustFit 1 --doFits' % (datacardFile_output))
            run_cmd('combineTool.py -M Impacts -m 125 -d %s  -o impacts.json' % (datacardFile_output))
            run_cmd('rm higgsCombineTest*.root')
            run_cmd('rm higgsCombine*root')
            run_cmd('plotImpacts.py -i impacts.json -o  impacts')
            run_cmd('mv impacts.pdf '+mom+local+'impacts_'+channels[nn]+"_"+university+'.pdf')

        if doGOF :
            run_cmd('%s --input_file=%s --output_file=%s --add_shape_sys=true' % ('WriteDatacards_'+channels[nn]+wdata[nn], my_file, datacardFile_output))
            run_cmd('combine -M GoodnessOfFit --algo=saturated --fixedSignalStrength=1 %s' % (txtFile))
            run_cmd('combine -M GoodnessOfFit --algo=saturated --fixedSignalStrength=1 -t 1000 -s 12345  %s --saveToys --toysFreq' % (txtFile))
            # the bellow work on CMSSW7X
            #run_cmd('combineTool.py -M GoodnessOfFit --algorithm saturated -d %s -n .saturated' % (datacardFile_output))
            #run_cmd('combineTool.py -M GoodnessOfFit --algorithm saturated -d %s -n .saturated  -n .saturated.toys -t 200 -s 0:4:1 --parallel 5' % (datacardFile_output))
            run_cmd('combineTool.py -M CollectGoodnessOfFit --input higgsCombineTest.GoodnessOfFit.mH120.root higgsCombineTest.GoodnessOfFit.mH120.12345.root -o GoF_saturated.json')
            run_cmd('$CMSSW_BASE/src/CombineHarvester/CombineTools/scripts/plotGof.py --statistic saturated --mass 120.0 GoF_saturated.json -o GoF_saturated')
            run_cmd('mv GoF_saturated.pdf '+mom+local+'GoF_saturated_'+channels[nn]+"_"+university+'.pdf')
            run_cmd('mv GoF_saturated.png '+mom+local+'GoF_saturated_'+channels[nn]+"_"+university+'.png')
            run_cmd('rm higgsCombine*root')

        if doYields :
            # fitDiagnostics.root
            run_cmd('%s --input_file=%s --output_file=%s --add_shape_sys=true' % ('WriteDatacards_'+channels[nn]+wdata[nn], my_file, datacardFile_output))
            run_cmd('combine -M FitDiagnostics -d %s  -t -1' % (txtFile))
            run_cmd('python $CMSSW_BASE/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py -a fitDiagnostics.root -g plots.root')
            run_cmd('combineTool.py  -M T2W -i %s' % (txtFile))
            ROOT.PyConfig.IgnoreCommandLineOptions = True
            gROOT.SetBatch(ROOT.kTRUE)
            gSystem.Load('libHiggsAnalysisCombinedLimit')
            print ("Retrieving yields from: ",datacardFile_output)
            fin = TFile(datacardFile_output)
            wsp = fin.Get('w')
            cmb = ch.CombineHarvester()
            cmb.SetFlag("workspaces-use-clone", True)
            ch.ParseCombineWorkspace(cmb, wsp, 'ModelConfig', 'data_obs', True)
            mlf = TFile('fitDiagnostics.root')
            print "arrived here"
            rfr = mlf.Get('fit_s')
            print "arrived here3"
            print 'Pre-fit tables:'
            filey = open(mom+local+"yields_"+channels[nn]+"_"+university+".tex","w")
            PrintTables(cmb, tuple(), 'ttH_'+channels[nn], filey, university, channels[nn], blinded)
            #cmb.UpdateParameters(rfr) 'ttH_2l_2tau'
            #print 'Post-fit tables:\n\n'
            #PrintTables(cmb, (rfr, 500))
            print ("the yields are on this file: ", mom+local+"yields_"+channels[nn]+"_"+university+".tex")

    else : print (my_file,"does not exist ")
    if doPlots : run_cmd("bash "+mom+local+"execute_plots"+channels[nn]+"_"+university+".sh")
################################################################
