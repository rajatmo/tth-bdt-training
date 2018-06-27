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

doLimits = True
doImpacts = False
doYields = False
doGOF = False
doPlots = False
readLimits = False

blinded=True

channel = options.channel
university = options.uni

if university == "Tallinn_alternative":
    typeFit = "postfit"
    takeRebinedFolder=False
    add_x_prefix=False
    doRebin = True
    divideByBinWidth = "false"
    doKeepBlinded = "true"
    autoMCstats = "true"
    useSyst = "true" # use shape syst
    mom = "/home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun09/"
    local = "Tallinn_alternative/"
    card_prefix = "prepareDatacards_"
    cards = [
    "1l_2tau_mTauTauVis_x",
    "2lss_1tau_sumOS_mTauTauVis_x",
    "2l_2tau_mTauTauVis_x",
    "3l_1tau_mTauTauVis_x",
    #
    "1l_2tau_numJets_x",
    "2lss_1tau_sumOS_numJets_x",
    "2l_2tau_numJets_x",
    "3l_1tau_numJets_x"
    ]

    channels = [
    "1l_2tau",
    "2lss_1tau",
    "2l_2tau",
    "3l_1tau",
    #
    "1l_2tau",
    "2lss_1tau",
    "2l_2tau",
    "3l_1tau"
    ]

if university == "Tallinn_CR":
    typeFit = "postfit"
    takeRebinedFolder=False
    add_x_prefix=False
    doRebin = False
    divideByBinWidth = "false"
    doKeepBlinded = "false"
    autoMCstats = "true"
    useSyst = "true" # use shape syst
    mom = "/home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun09/"
    local = "Tallinn_CR/"
    card_prefix = "prepareDatacards_"
    cards = [
    "ttWctrl_mvaDiscr_2lss",
    "ttWctrl_EventCounter",
    "ttWctrl_numJets",
    "ttZctrl_mvaDiscr_3l",
    "ttZctrl_EventCounter",
    "ttZctrl_mT",
    "ttZctrl_numJets",
    "ttZctrl_mLL"
    ]

    channels = [
    "ttWctrl",
    "ttWctrl",
    "ttWctrl",
    "ttZctrl",
    "ttZctrl",
    "ttZctrl",
    "ttZctrl",
    "ttZctrl",
    ]

if university == "Tallinn_HH_autoMCstats":
    typeFit = "postfit"
    takeRebinedFolder=False
    add_x_prefix=False
    doRebin = False
    doKeepBlinded = "true"
    autoMCstats = "true"
    useSyst = "true" # use shape syst
    mom = "/home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun26/"
    local = "Tallinn/"
    card_prefix = "addSystFakeRates_"
    cards = [
    "1l_2tau_OS_mvaOutput_final_x",
    "2lss_1tau_sumOS_mvaOutput_final_x",
    "2l_2tau_sumOS_mvaOutput_final_x",
    "3l_1tau_OS_mvaOutput_final_x",
    "ttWctrl_mvaDiscr_2lss_x",
    "ttZctrl_mvaDiscr_3l_x"
    ]

    channels = [
    "1l_2tau",
    "2lss_1tau",
    "2l_2tau",
    "3l_1tau",
    "ttWctrl",
    "ttZctrl"
    ]

    folders = [
    "ttH_1l_2tau/",
    "",
    "ttH_2l_2tau/",
    "ttH_3l_1tau/",
    "ttH_2lss_1tau/",
    "ttH_1l_2tau/",
    "ttH_1l_2tau/"
    ]

if university == "Tallinn":
    typeFit = "postfit"
    takeRebinedFolder=False
    add_x_prefix=False
    doRebin = False
    doKeepBlinded = "true"
    autoMCstats = "true"
    useSyst = "true" # use shape syst
    mom = "/home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun09/"
    local = "Tallinn/"
    card_prefix = "prepareDatacards_"
    cards = [
    "1l_2tau_mvaOutput_final_x",
    "2lss_1tau_sumOS_mvaOutput_final_x",
    "2l_2tau_mvaOutput_final_regularBin", #"2l_2tau_mvaOutput_final_x",
    "3l_1tau_mvaOutput_final_x_noNeg",
    "2lss_1tau_sumOS_mvaOutput_2lss_1tau_HTT_SUM_M",
    "1l_2tau_mvaOutput_HTT_SUM_VT",
    #
    "1l_2tau_SS_mTauTauVis_x",
    "1l_2tau_SS_mvaOutput_final_x",
    "1l_2tau_SS_EventCounter_x",
    "1l_2tau_SS_EventCounter"
    ]

    channels = [
    "1l_2tau",
    "2lss_1tau",
    "2l_2tau",
    "3l_1tau",
    "2lss_1tau",
    "1l_2tau",
    #
    "1l_2tau",
    "1l_2tau",
    "1l_2tau",
    "1l_2tau"
    ]

    folders = [
    "ttH_1l_2tau/",
    "",
    "ttH_2l_2tau/",
    "ttH_3l_1tau/",
    "ttH_2lss_1tau/",
    "ttH_1l_2tau/",
    #
    "ttH_1l_2tau/",
    "ttH_1l_2tau/",
    "ttH_1l_2tau/",
    ]

elif university == "Cornell":
    typeFit = "postfit"
    takeRebinedFolder=False
    add_x_prefix=False
    doRebin = False
    doKeepBlinded = "true"
    autoMCstats = "true"
    useSyst = "true" # use shape syst
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


print ("to run this script your CMSSW_base should be the one that CombineHavester installed")

if not readLimits :
    for nn, card in enumerate(cards) :
        if not nn < 4 : continue
        #if not channels[nn] == "2l_2tau" : continue #
        #####################################################################
        wdata = "" # to append to WriteDatacard_$channel
        hasConversions = "true"
        if channels[nn] == "1l_2tau" :
            wdata = ""
            hasFlips = "false"
            isSplit = "false"
            max = 2000
            minimim = 0.1
            dolog = "true"
            divideByBinWidth = "false"
        if channels[nn] == "2l_2tau" :
            wdata = ""
            hasFlips = "false"
            isSplit = "false"
            max = 15.0
            minimim = 0.0
            dolog = "false"
            divideByBinWidth = "false"
        if channels[nn] == "3l_1tau" :
            #if "CR" not in university : wdata = "_FRjt_syst"
            #else :
            wdata = ""
            isSplit = "true"
            hasFlips = "false"
            max = 2.5
            minimim = 0.0
            dolog = "false"
            divideByBinWidth = "true"
        if channels[nn] == "2lss_1tau" :
            #if "CR" not in university : wdata = "_FRjt_syst"
            #else :
            wdata = ""
            isSplit = "true"
            hasFlips = "true"
            max = 15.0
            minimim = 0.0
            dolog = "false"
            divideByBinWidth = "true"
        if "ctrl" in channels[nn] :
            wdata = ""
            hasFlips = "true"
            isSplit = "false"
            max = 500000
            minimim = 0.1
            dolog = "true"
            divideByBinWidth = "false"
        #####################################################################
        my_file = mom+local+card_prefix+card+'.root'
        file = TFile(my_file,"READ");
        if os.path.exists(my_file) :
            print ("testing ", my_file)
            my_file = mom+local+card_prefix+card+'_noNeg.root' # remove the negatives
            file2 = TFile(my_file,"RECREATE");
            file2.cd()
            h2 = TH1F()
            for keyO in file.GetListOfKeys() :
               obj =  keyO.ReadObj()
               if type(obj) is not TH1F : continue
               if not takeRebinedFolder :  h2=obj.Clone()
               else : h2 = file.Get(folders[nn]+"rebinned/"+str(keyO.GetName())+"_rebinned")
               if doRebin :
                   if channel[nn] == "2l_2tau" : h2.Rebin(5)
                   if channel[nn] == "3l_1tau" : h2.Rebin(4)
               for bin in range (0, h2.GetXaxis().GetNbins()) :
                   if h2.GetBinContent(bin) < 0 :
                       h2.AddBinContent(bin, abs(h2.GetBinContent(bin))+0.01)
                   h2.SetBinError(bin, min(h2.GetBinContent(bin), h2.GetBinError(bin)) ) # crop all uncertainties to 100% to avoid negative variations
                   if add_x_prefix : h2.SetName("x_"+str(keyO.GetName()))
               h2.Write()
            file2.Close()

            # make txt datacard
            datacard_file=my_file
            datacardFile_output = mom+local+"ttH_"+card
            run_cmd('%s --input_file=%s --output_file=%s --add_shape_sys=%s --use_autoMCstats=%s' % ('WriteDatacards_'+channels[nn]+wdata, my_file, datacardFile_output, useSyst, autoMCstats))
            txtFile = datacardFile_output + ".txt"
            logFile = datacardFile_output + ".log"
            logFileNoS = datacardFile_output +  "_noSyst.log"
            if doLimits :
                run_cmd('cd '+mom+local+' ; combine -M AsymptoticLimits -m %s -t -1 --run blind -S 0 %s &> %s' % (str(125), txtFile, logFileNoS))
                run_cmd('cd '+mom+local+' ; combine -M AsymptoticLimits -m %s -t -1 --run blind %s &> %s' % (str(125), txtFile, logFile))
                run_cmd('cd '+mom+local+' ;  rm higgsCombineTest.AsymptoticLimits.mH125.root')

            if doPlots :
                filesh = open(mom+local+"execute_plots"+channels[nn]+"_"+university+".sh","w")
                filesh.write("#!/bin/bash\n")
                rootFile = mom+local+"ttH_"+card+"_shapes.root"
                run_cmd('%s --input_file=%s --output_file=%s --add_shape_sys=%s --use_autoMCstats=%s' % ('WriteDatacards_'+channels[nn]+wdata, my_file, datacardFile_output, useSyst, autoMCstats))
                run_cmd('cd '+mom+local+' combine -M FitDiagnostics -d %s  -t -1  --expectSignal 1' % (txtFile))
                run_cmd('cd '+mom+local+' PostFitShapes -d %s -o %s -m 125 -f fitDiagnostics.root:fit_s --postfit --sampling --print' % (txtFile, rootFile)) # --postfit
                makeplots=('root -l -b -n -q /home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\\"'
                +str(card)+'\\",\\"'+str(local)+'\\",\\"'+str(channels[nn])+'\\",\\"'+str(mom)+'\\",'+str(dolog)+','+str(hasFlips)+','+hasConversions+',\\"BDT\\",\\"\\",'+str(minimim)+','+str(max)+','+isSplit+',\\"'+typeFit+'\\",'+divideByBinWidth+','+doKeepBlinded+')')
                #root -l -b -n -q /home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"2lss_1tau_sumOS_mvaOutput_2lss_1tau_HTT_SUM_M_11bins_quantiles\",\"2018jun02/\",\"2lss_1tau\",\"/home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/\",false,false,\"BDT\",\"\",0.0,10.0)
                filesh.write(makeplots+ "\n")
                print ("to have the plots take the makePlots command from: ",mom+local+"execute_plots"+channels[nn]+"_"+cards[nn]+"_"+university+".sh")

            if doImpacts :
                run_cmd('cd '+mom+local+' ; combineTool.py  -M T2W -i %s' % (txtFile))
                run_cmd('cd '+mom+local+' ; combineTool.py -M Impacts -m 125 -d %s.root  --expectSignal 1 --allPars --parallel 8 -t -1 --doInitialFit' % (datacardFile_output))
                run_cmd('cd '+mom+local+' ; combineTool.py -M Impacts -m 125 -d %s.root --expectSignal 1 --allPars --parallel 8 -t -1 --robustFit 1 --doFits' % (datacardFile_output))
                run_cmd('cd '+mom+local+' ; combineTool.py -M Impacts -m 125 -d %s.root -o impacts.json' % (datacardFile_output))
                run_cmd('cd '+mom+local+' ; rm higgsCombineTest*.root')
                run_cmd('cd '+mom+local+' ; rm higgsCombine*root')
                run_cmd('cd '+mom+local+' ; plotImpacts.py -i impacts.json -o  impacts')
                run_cmd('cd '+mom+local+' ; mv impacts.pdf '+mom+local+'impacts_'+channels[nn]+"_"+cards[nn]+'_'+university+'.pdf')

            if doGOF :
                run_cmd('%s --input_file=%s --output_file=%s --add_shape_sys=%s --use_autoMCstats=%s' % ('WriteDatacards_'+channels[nn]+wdata, my_file, datacardFile_output, useSyst, autoMCstats))
                run_cmd('cd '+mom+local+' ;  combine -M GoodnessOfFit --algo=saturated --fixedSignalStrength=1 %s' % (txtFile))
                run_cmd('combine -M GoodnessOfFit --algo=saturated --fixedSignalStrength=1 -t 1000 -s 12345  %s --saveToys --toysFreq' % (txtFile))
                run_cmd('cd '+mom+local+'combineTool.py -M CollectGoodnessOfFit --input higgsCombineTest.GoodnessOfFit.mH120.root higgsCombineTest.GoodnessOfFit.mH120.12345.root -o GoF_saturated.json')
                run_cmd('cd '+mom+local+' ; $CMSSW_BASE/src/CombineHarvester/CombineTools/scripts/plotGof.py --statistic saturated --mass 120.0 GoF_saturated.json -o GoF_saturated')
                run_cmd('cd '+mom+local+' ; mv GoF_saturated.pdf '+mom+local+'GoF_saturated_'+channels[nn]+"_"+cards[nn]+"_"+university+'.pdf')
                run_cmd('cd '+mom+local+' ; mv GoF_saturated.png '+mom+local+'GoF_saturated_'+channels[nn]+"_"+cards[nn]+"_"+university+'.png')
                run_cmd('cd '+mom+local+' ; rm higgsCombine*root')
                run_cmd('cd '+mom+local+' ; rm GoF_saturated.json')

            if doYields :
                run_cmd('%s --input_file=%s --output_file=%s --add_shape_sys=%s --use_autoMCstats=%s' % ('WriteDatacards_'+channels[nn]+wdata, my_file, datacardFile_output, useSyst, autoMCstats))
                run_cmd('combine -M FitDiagnostics -d %s  -t -1 --expectSignal 1' % (txtFile))
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
                ch.ParseCombineWorkspace(cmb, wsp, 'ModelConfig', 'data_obs', False)
                mlf = TFile('fitDiagnostics.root')
                rfr = mlf.Get('fit_s')
                print 'Pre-fit tables:'
                filey = open(mom+local+"yields_"+channels[nn]+"_"+university+"_prefit.tex","w")
                PrintTables(cmb, tuple(), 'ttH_'+channels[nn], filey, university, channels[nn], blinded)
                cmb.UpdateParameters(rfr)
                print 'Post-fit tables:'
                filey = open(mom+local+"yields_"+channels[nn]+"_"+university+"_postfit.tex","w")
                PrintTables(cmb, (rfr, 500), 'ttH_'+channels[nn], filey, university, channels[nn], blinded)
                print ("the yields are on this file: ", mom+local+"yields_"+channels[nn]+"_"+university+"_*.tex")

        else : print (my_file,"does not exist ")
        if doPlots : run_cmd("bash "+mom+local+"execute_plots"+channels[nn]+"_"+university+".sh")
################################################################

## in the future this will also do the plots of limits
if readLimits :
    colorsToDo = np.arange(1,4)
    binstoDo=np.arange(1,4)
    file = open(mom+local+"limits.csv","w")
    for ii in [0] :
        for nn,channel in enumerate(channels) :
            if not nn < 4 : continue
            #options.variables+'_'+bdtTypesToDoFile[ns]+'_nbin_'+str(nbins)
            if ii == 0 : limits=ReadLimits( cards[nn], [1],"" ,channel,mom+local,-1,-1)
            if ii == 1 : limits=ReadLimits( cards[nn], [1],"_noSyst" ,channel,mom+local,-1,-1)
            print (channel, limits)
            for jj in limits[0] : file.write(str(jj)+', ')
            file.write('\n')
            #plt.plot(binstoDo,limits[0], color=colorsToDo[nn],linestyle='-',marker='o',label="bdtTypesToDoLabel[nn]")
            #plt.plot(binstoDo,limits[1], color=colorsToDo[nn],linestyle='-')
            #plt.plot(binstoDo,limits[3], color=colorsToDo[nn],linestyle='-')
        #ax.legend(loc='best', fancybox=False, shadow=False , ncol=1)
        #ax.set_xlabel('nbins')
        #ax.set_ylabel('limits')
        #maxsum=0
        #plt.axis((min(binstoDo),max(binstoDo),0.5,2.5))
        #ax.legend(loc='best', fancybox=False, shadow=False , ncol=1)
        #ax.set_xlabel('nbins')
        #ax.set_ylabel('limits')
        #maxsum=0
        #plt.axis((min(binstoDo),max(binstoDo),0.5,2.5))
