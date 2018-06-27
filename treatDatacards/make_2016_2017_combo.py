#!/usr/bin/env python
import os, subprocess, sys
from array import array
from ROOT import *
from math import sqrt, sin, cos, tan, exp
import numpy as np
import glob
#from pathlib2 import Path
import CombineHarvester.CombineTools.ch as ch
execfile("../python/data_manager.py")

def run_cmd(command):
  print "executing command = '%s'" % command
  p = subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
  stdout, stderr = p.communicate()
  print stderr
  return stdout

# Download the bellow folders to mom and untar them on the same location than this script
# https://svnweb.cern.ch/cern/wsvn/cmshcg/trunk/cadi/HIG-18-019/
# https://svnweb.cern.ch/cern/wsvn/cmshcg/trunk/cadi/HIG-17-018/
mom_2017 = "HIG-18-019.r7705/2018jun26/"
mom_2016 = "HIG-17-018.r7707/2016_for_hig18019/"

workingDir = os.getcwd()
workingDir = workingDir+"/"
print "Working directory is: "+workingDir
procP1=glob.glob(workingDir+mom_2017+"*.txt")
procP2=glob.glob(workingDir+mom_2016+"*.txt")
everybody = procP1 + procP2

combine_cards = False
float_signal = False
# put false bellow if you already ran once to copy the cards
# to run for first time on a mode it should be true
copy_cards =  False
btag_correlated = True
if not btag_correlated : mom_result = "comblo_2016_2017_uncorrJES/"
else : mom_result = "combo_2016_2017/"

doYields = True
blinded = True
#mom_result = "combo_plain/"

def decorrelate_btag(p) :
    cb.cp().process([p.process()]).RenameSystematic(cb, "CMS_ttHl16_btag_"+s , "CMS_ttHl17_btag_"+s);

def correlate_tauES(p) :
    cb.cp().process([p.process()]).RenameSystematic(cb, "CMS_ttHl_tauES", "CMS_scale_t");

def correlate_tauID(p) :
    cb.cp().process([p.process()]).RenameSystematic(cb, "CMS_ttHl17_tauID", "CMS_eff_t");

def decorrelate_JES(p) :
    cb.cp().process([p.process()]).RenameSystematic(cb, "CMS_scale_j" , "CMS_ttHl17_scale_j");

if copy_cards :
    run_cmd('mkdir '+workingDir+mom_result)
    # rename only on the 2017
    for nn, process in enumerate(everybody) :
        #if nn > 0 : continue
        #print process
        cb = ch.CombineHarvester()
        tokens = process.split("/")
        if not "ttH" in process :
            print tokens[8]+" "+tokens[9]+" ignoring card "+process
            continue
        proc_name = "Name"+str(nn+1)
        for part in tokens :
            if "ttH" in part :
                for name in part.split(".") :
                    if "ttH" in name :
                        print " adding process "+name
                        proc_name = name

        if "HIG-18-019" in process :
            complement = "_2017"
        if "HIG-17-018" in process :
            complement = "_2016"

        cb.ParseDatacard(process, analysis = proc_name+complement, mass = "")

        #if not btag_correlated and "HIG-18-019" in process:
        #  print "start decorrelating btag"
        #  for s in  ["HF", "LF", "cErr1", "cErr2"] :
        #    print "renaming for "+s
        #    cb.ForEachProc(decorrelate_btag)
        if not btag_correlated and "HIG-18-019" in process:
            cb.ForEachProc(decorrelate_JES)

        writer = ch.CardWriter(workingDir+mom_result+proc_name+complement+'.txt',
                   workingDir+mom_result+proc_name+complement+'.root')
        writer.WriteCards('LIMITS/cmb', cb)


everybody = glob.glob(workingDir+mom_result+"*.txt")

if btag_correlated :
    cardToWrite = "card_combo_2016_2017_btag_correlated"
    cardToWrite_2017 = "card_combo_2017_btag_correlated"
else :
    cardToWrite = "card_combo_2016_2017_JES_Notcorrelated"
    cardToWrite_2017 = "card_combo_2017_JES_Notcorrelated"

if combine_cards :

    # combineCards.py Name1=card1.txt Name2=card2.txt .... > card.txt
    string_combine = "combineCards.py "
    string_combine_2017 = "combineCards.py "
    for nn, process in enumerate(everybody) :
        tokens = process.split("/")

        # collect the cards
        if not "ttH" in process :
            print "ignoring card "+process
            continue
        proc_name = "Name"+str(nn+1)
        file_name = "Name"
        for part in tokens :
            if "ttH" in part :
                file_name = part
                for name in part.split(".") :
                    if "ttH" in name :
                        print " adding process "+name
                        proc_name = name

        if "Name" in proc_name and "ttH" in process :
            print "There is a problem ..... ..... .... not ignoring card "+process
            break

        # collect the cards to run full combo
        string_combine = string_combine + proc_name+"="+file_name+" "
        if "2016" not in proc_name :
            string_combine_2017 = string_combine_2017 + proc_name+"="+file_name+" "

    string_combine = string_combine+" > "+cardToWrite+".txt"
    run_cmd("cd "+workingDir+mom_result+" ; "+string_combine+" ; cd %s"  % (workingDir))

    string_combine_2017 = string_combine_2017+" > "+cardToWrite_2017+".txt"
    run_cmd("cd "+workingDir+mom_result+" ; "+string_combine_2017+" ; cd %s"  % (workingDir))

if float_signal :

    ### for combo results 2017 and 2017 + 2016
    for card in [ cardToWrite,  cardToWrite_2017] :
        WS_output = card+"_3poi"

        run_cmd("cd "+workingDir+mom_result+" ; text2workspace.py %s.txt -o %s.root -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose  --PO 'map=.*/ttH.*:r_ttH[1,-2,5]'  --PO 'map=.*/TTW:r_ttW[1,-2,5]' --PO 'map=.*/TTW_.*:r_ttW[1,-2,5]' --PO 'map=.*/TTWW.*:r_ttW[1,-2,5]'  --PO 'map=.*/TTZ.*:r_ttZ[1,-2,5]'   ; cd %s"  % (card, WS_output, workingDir))

        run_cmd("cd "+workingDir+mom_result+" ; combine -M Significance --signif %s.root -t -1 --setParameters r_ttH=1,r_ttW=1,r_ttZ=1 --redefineSignalPOI r_ttH > %s.log  ; cd %s"  % (WS_output, WS_output, workingDir))

        run_cmd("cd "+workingDir+mom_result+" ; combine -M MultiDimFit %s.root -t -1 --setParameters r_ttH=1,r_ttW=1,r_ttZ=1 --algo singles -P r_ttH --floatOtherPOI=1 > %s_rate_ttH.log ; cd %s"  % (WS_output, WS_output, workingDir))

        run_cmd("cd "+workingDir+mom_result+" ; combine -M MultiDimFit %s.root -t -1 --setParameters r_ttH=1,r_ttW=1,r_ttZ=1 --algo singles > %s_rate_3D.log  ; cd %s"  % (WS_output, WS_output, workingDir))

        if card == cardToWrite : # hessian impacts

            run_cmd("cd "+workingDir+mom_result+" ; combineTool.py -M Impacts -d %s.root --rMin -2 --rMax 5 --redefineSignalPOIs r_ttH -t -1 --setParameters r_ttH=1,r_ttW=1,r_ttZ=1 -m 125 --doFits --approx hesse ; cd %s "  % (WS_output, workingDir))

            run_cmd("cd "+workingDir+mom_result+" ; combineTool.py -M Impacts -d %s.root -m 125 -o impacts.json --approx hesse --rMin -2 --rMax 5 --redefineSignalPOIs r_ttH -t -1; plotImpacts.py -i impacts.json -o  impacts ; mv impacts.pdf impacts_hesse_JESCorr%s ; cd %s" % (WS_output, str(btag_correlated), workingDir))

        if card == cardToWrite_2017 :
            ### for category by category - 2017 only
            WS_output_byCat = cardToWrite_2017+"_Catpoi"

            floating_by_cat = \
            "--PO 'map=.*2lss_e.*/ttH.*:r_ttH_2lss_0tau[1,-5,10]'\
            --PO 'map=.*2lss_m.*/ttH.*:r_ttH_2lss_0tau[1,-5,10]'\
            --PO 'map=.*3l_b.*/ttH.*:r_ttH_3l_0tau[1,-5,10]'\
            --PO 'map=.*4l_2.*/ttH.*:r_ttH_4l[1,-5,10]'\
            --PO 'map=.*2lss_1tau.*/ttH.*:r_ttH_2lss_1tau[1,-5,10]'\
            --PO 'map=.*3l_1tau.*/ttH.*:r_ttH_3l_1tau[1,-5,10]'\
            --PO 'map=.*2l_2tau.*/ttH.*:r_ttH_2l_2tau[1,-5,10]'\
            --PO 'map=.*1l_2tau.*/ttH.*:r_ttH_1l_2tau[1,-5,10]'\
            "

            run_cmd("cd "+workingDir+mom_result+" ; text2workspace.py %s -o %s.root -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose %s --PO 'map=.*/TTZ.*:r_ttZ[1,-2,5]'  --PO 'map=.*/TTW:r_ttW[1,-2,5]' --PO 'map=.*/TTW_.*:r_ttW[1,-2,5]' --PO 'map=.*/TTWW.*:r_ttW[1,-2,5]'  ; cd %s"  % (card+".txt", WS_output_byCat, floating_by_cat, workingDir))

            parameters = "r_ttW=1,r_ttZ=1"
            for rate in ["r_ttH_2lss_0tau", "r_ttH_3l_0tau", "r_ttH_4l.", "_ttH_2lss_1tau", "r_ttH_3l_1tau", "r_ttH_2l_2tau", "r_ttH_1l_2tau"] :
                parameters = parameters+","+rate+"=1"
            print "Will fit the parameters "+parameters

            for rate in ["r_ttH_2lss_0tau", "r_ttH_3l_0tau", "r_ttH_4l", "_ttH_2lss_1tau", "r_ttH_3l_1tau", "r_ttH_2l_2tau", "r_ttH_1l_2tau"] :

                run_cmd("cd "+workingDir+mom_result+" ; combine -M MultiDimFit %s.root -t -1 --setParameters %s --algo singles -P %s --floatOtherPOI=1 > %s_rate_%s.log ; cd %s"  % (WS_output_byCat, parameters, rate, WS_output_byCat, rate, workingDir))

        if card == cardToWrite and 0 > 1 :
            ### For impacts 2017 + 2016 only

            run_cmd("cd "+workingDir+mom_result+" ; combineTool.py -M Impacts -m 125 -d %s.root --setParameters r_ttH=1,r_ttW=1,r_ttZ=1 --redefineSignalPOI r_ttH --parallel 8 -t -1 --doInitialFit  ; cd %s "  % (WS_output, workingDir))

            run_cmd("cd "+workingDir+mom_result+" ; combineTool.py -M Impacts -m 125 -d %s.root   --setParameters r_ttH=1,r_ttW=1,r_ttZ=1 --redefineSignalPOI r_ttH --parallel 8 -t -1 --robustFit 1 --doFits  ; cd %s "  % (WS_output, workingDir))

            run_cmd("cd "+workingDir+mom_result+" ; combineTool.py -M Impacts -m 125 -d %s.root  -o impacts.json ; plotImpacts.py -i impacts.json -o  impacts ; mv impacts.pdf impacts_btagCorr%s ; cd %s" % (WS_output, str(btag_correlated), workingDir))

if doYields :
    channels = ["3l_1tau_OS_mvaOutput_final_x_2017", "2l_2tau_sumOS_mvaOutput_final_x_2017", "1l_2tau_OS_mvaOutput_final_x_2017", "2lss_1tau_sumOS_mvaOutput_final_x_2017"]
    run_cmd("cd "+workingDir+mom_result+' ; combine -M FitDiagnostics -d %s_3poi.root  -t -1  --setParameters r_ttH=1,r_ttW=1,r_ttZ=1 --redefineSignalPOI r_ttH  ; cd %s' % (cardToWrite_2017, workingDir))
    run_cmd("cd "+workingDir+mom_result+' ; python $CMSSW_BASE/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py -a fitDiagnostics.Test.root -g plots.root  -p r_ttH  ; cd '+workingDir)
    #ROOT.PyConfig.IgnoreCommandLineOptions = True
    #gROOT.SetBatch(ROOT.kTRUE)
    gSystem.Load('libHiggsAnalysisCombinedLimit')
    print ("Retrieving yields from: ",cardToWrite_2017)
    fin = TFile(workingDir+mom_result+cardToWrite_2017+"_3poi.root")
    wsp = fin.Get('w')
    cmb = ch.CombineHarvester()
    cmb.SetFlag("workspaces-use-clone", True)
    ch.ParseCombineWorkspace(cmb, wsp, 'ModelConfig', 'data_obs', False)
    print "datacard parsed"
    mlf = TFile(workingDir+mom_result+'fitDiagnostics.Test.root')
    rfr = mlf.Get('fit_s')

    print 'Pre-fit tables:'

    #labels = [ "1l_2tau_OS_mvaOutput_final_x_2017", "2l_2tau_sumOS_mvaOutput_final_x_2017", "3l_1tau_OS_mvaOutput_final_x_2017",  "2lss_1tau_sumOS_mvaOutput_final_x_2017"]
    #type = 'tau'
    colapseCat = False
    #filey = open(workingDir+mom_result+"yields_"+type+"_from_combo_prefit.tex","w")
    #PrintTables_Tau(cmb, tuple(), filey, blinded, labels, type)

    labels = ["ttH_2lss_ee_neg_2017",
    "ttH_2lss_ee_pos_2017",
    "ttH_2lss_em_bl_neg_2017",
    "ttH_2lss_em_bl_pos_2017",
    "ttH_2lss_em_bt_neg_2017",
    "ttH_2lss_em_bt_pos_2017",
    "ttH_2lss_mm_bl_neg_2017",
    "ttH_2lss_mm_bl_pos_2017",
    "ttH_2lss_mm_bt_neg_2017",
    "ttH_2lss_mm_bt_pos_2017",
    "ttH_3l_bl_neg_2017",
    "ttH_3l_bl_pos_2017",
    "ttH_3l_bt_neg_2017",
    "ttH_3l_bt_pos_2017",
    "ttH_4l_2017"]
    type = 'multilep'
    colapseCat = True
    filey = open(workingDir+mom_result+"yields_"+type+"_from_combo_prefit.tex","w")
    PrintTables_Tau(cmb, tuple(), filey, blinded, labels, type)


    cmb.UpdateParameters(rfr)
    workingDir = os.getcwd()
    workingDir = workingDir+"/"
    #print 'Post-fit tables:'
    #filey = open(workingDir+mom_result+"yields_tau_from_combo_postfit.tex","w")
    #PrintTables_Tau(cmb, (rfr, 500), filey, blinded)
    print ("the yields are on this file: ", workingDir+mom_result+"yields_tau_"+"from_combo"+"_*.tex")






"""
 ttH_3l_bl_neg_zpeak_2017
 ttH_3l_bl_pos_zpeak_2017
 ttH_3l_bt_neg_zpeak_2017
 ttH_3l_bt_pos_zpeak_2017

 ttH_2lss_ee_pos_3j_2017
 ttH_2lss_ee_neg_3j_2017

 ttH_2lss_em_bl_neg_3j_2017
 ttH_2lss_em_bl_pos_3j_2017
 ttH_2lss_em_bt_neg_3j_2017
 ttH_2lss_em_bt_pos_3j_2017
 ttH_2lss_mm_bl_neg_3j_2017
 ttH_2lss_mm_bl_pos_3j_2017
 ttH_2lss_mm_bt_neg_3j_2017
 ttH_2lss_mm_bt_pos_3j_2017

 ttH_3l_crwz_2017

 ttH_4l_crzz_2017
"""
