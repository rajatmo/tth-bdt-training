#!/usr/bin/env python
import os, subprocess, sys
from array import array
from ROOT import *
from math import sqrt, sin, cos, tan, exp
import numpy as np
import glob
#from pathlib2 import Path
execfile("../python/data_manager.py")
from random import randint

#####################################################################
## From where to take the cards:
# The bellow is going to construct the combo card from scratch given the path of the cards by channel (see mom_2017/mom_2016)
copy_cards =  False
combine_cards = False
# If you already have the combined card put the bellow true (see mom_result)
takeCombo = True
#####################################################################

#####################################################################
## What to do:
doFits = False ## To do any of the rest you must have ran with this being True once
## but, once it is done one there is no need to loose time repeating it

#######################
## Each of the steps bellow take time, I suggest to do on one separated runs
# (like this they can be also done in parallel)
# 1)
doCategories = True

# 2)
# for combo is almost mandatory send to grid,
# if you run on lxplus combineTool is already set to send to Condor (see bellow 'How')
doImpactCombo = False
doImpact2017 = False
# If you will run in Condor this must be done in two steps:
#  -  the first with impactsSubmit= True ---> wait for the jobs to be ready
#      - To see that is all done: check that you have the same number of root files than the number of nuissances to be considered
#  - after the jobs are done run again with impactsSubmit = False to wrap up the plot
impactsSubmit = False

# 3) -- optional
# fast version of the impacts -- Hessian approximation for quick checks
doHessImpactCombo = False
doHessImpact2017 = False

# 4)
# the bellow take lots of time, better to do on one run
# It is almost mandatory send to grid,
# if you run on lxplus combineTool is already set to send to Condor (see bellow 'How')
doGOFCombo = False
doGOF2017 = False
# If you will run in Condor this must be done in two steps:
#  -  the first with GOF_submit = True ---> wait for the jobs to be ready
#     -- each job will contain 2 toys
#        (if needed run the submission bash script created again up to you be happy with the number of toys)
#  - after the jobs are done run again with GOF_submit = False  to wrap up the plot
GOF_submit = False

# 5)
# OR you do from Combine or from from Havester
preparePostFitCombine = False
preparePostFitHavester = False

doYieldsAndPlots = False # will do the prefit and postfit table of yields.
# If any of the bellow are true it also do prefit and postfit plots
# You must have ran the respective preparePostFit... before (or put as true as well)
doPostFitCombine = False # doYieldsAndPlots must be true
doPostFitHavester = False # doYieldsAndPlots must be true
#####################################################################

#####################################################################
## How to do?
btag_correlated = True # if true it does not manipulate the cards
JES_correlated = True # if true it does not manipulate the cards
blinded = False
blindedOutput = False ## do not draw the rate on the impacts plot
sendToCondor = False ### Impacts and GOF for combo need it
#####################################################################

# Download the bellow folders to mom and untar them on the same location than this script
# https://svnweb.cern.ch/cern/wsvn/cmshcg/trunk/cadi/HIG-18-019/
# https://svnweb.cern.ch/cern/wsvn/cmshcg/trunk/cadi/HIG-17-018/
mom_2017 = "HIG-18-019.r7705/2018jun28/" ## these are updated from svn version !!!!!!!!!!!
mom_2017_multilep = "HIG-18-019.r7705/2018jun28/multilep/"
mom_2017_tau = "HIG-18-019.r7705/2018jun28/tau/"
mom_2016 = "HIG-17-018.r7707/2016_for_hig18019/"

print "Working directory is: "+os.getcwd()+"/"
procP1=glob.glob(os.getcwd()+"/"+mom_2017+"/*/*.txt")
procP2=glob.glob(os.getcwd()+"/"+mom_2016+"*.txt")
everybody = procP1 + procP2

## The results are going to be saved on the local folder mom_result
if not takeCombo :
    if not btag_correlated : mom_result = "combo_2016_2017_uncorrJES/"
    else :
        if  blinded : mom_result = "combo_2016_2017/"
        else : mom_result = "combo_2016_2017_unblided/"
else :
    # you will need to put the combined card on the folder hardcoded bellow
    mom_result = "gpetrucc_2017/"
    #mom_result = "gpetrucc_2017_2016/"

ToCondor = " "
if sendToCondor : ToCondor = " --job-mode condor --sub-opt '+MaxRuntime = 18000'"

if blinded : blindStatement = ' -t -1 '
else : blindStatement = ' '

def decorrelate_btag(p) :
    cb.cp().process([p.process()]).RenameSystematic(cb, "CMS_ttHl16_btag_"+s , "CMS_ttHl17_btag_"+s);

def correlate_tauES(p) :
    cb.cp().process([p.process()]).RenameSystematic(cb, "CMS_ttHl_tauES", "CMS_scale_t");

def correlate_tauID(p) :
    cb.cp().process([p.process()]).RenameSystematic(cb, "CMS_ttHl17_tauID", "CMS_eff_t");

def decorrelate_JES(p) :
    cb.cp().process([p.process()]).RenameSystematic(cb, "CMS_scale_j" , "CMS_ttHl17_scale_j");

if copy_cards :
    import CombineHarvester.CombineTools.ch as ch
    run_cmd('mkdir '+os.getcwd()+"/"+mom_result)
    # rename only on the 2017
    for nn, process in enumerate(everybody) :
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
        if not btag_correlated and "HIG-18-019" in process:
          print "start decorrelating btag"
          for s in  ["HF", "LF", "cErr1", "cErr2"] :
            print "renaming for "+s
            cb.ForEachProc(decorrelate_btag)
        if not JES_correlated and "HIG-18-019" in process:
            cb.ForEachProc(decorrelate_JES)
        writer = ch.CardWriter(os.getcwd()+"/"+mom_result+proc_name+complement+'.txt',
                   os.getcwd()+"/"+mom_result+proc_name+complement+'.root')
        writer.WriteCards('LIMITS/cmb', cb)

everybody = glob.glob(os.getcwd()+"/"+mom_result+"*.txt")

if btag_correlated :
    cardToWrite = "card_combo_2016_2017_btag_correlated"
    cardToWrite_2017 = "card_combo_2017_btag_correlated"
else :
    cardToWrite = "card_combo_2016_2017_JES_Notcorrelated"
    cardToWrite_2017 = "card_combo_2017_JES_Notcorrelated"

if combine_cards :
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
    run_cmd("cd "+os.getcwd()+"/"+mom_result+" ; "+string_combine+" ; cd %s"  % (os.getcwd()+"/"))
    string_combine_2017 = string_combine_2017+" > "+cardToWrite_2017+".txt"
    run_cmd("cd "+os.getcwd()+"/"+mom_result+" ; "+string_combine_2017+" ; cd %s"  % (os.getcwd()+"/"))


WS_output = card+"_3poi"
redefineToTTH = " --setParameters r_ttH=1,r_ttW=1,r_ttZ=1 --redefineSignalPOI r_ttH "

floating_ttV = \
"--PO 'map=.*/TTZ.*:r_ttZ[1,0,6]'\
--PO 'map=.*/TTW:r_ttW[1,-2,6]'\
--PO 'map=.*/TTW_.*:r_ttW[1,-2,6]'\
--PO 'map=.*/TTWW.*:r_ttW[1,-2,6]' "

### loop for combo results 2017 and 2017 + 2016
### hardcode the paths if you want to consider already combined cards
if takeCombo :
    cardToWrite_2017 = "comb_2017v2_withCR_sanity"
    cardToWrite = "../gpetrucc_2017_2016/comb_1617v2_withCR_sanity"

for card in [ cardToWrite_2017 , cardToWrite  ] : # ,

    if doFits :
        run_cmd("cd "+os.getcwd()+"/"+mom_result+" ; text2workspace.py %s.txt -o %s.root -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose  --PO 'map=.*/ttH.*:r_ttH[1,-5,10]' %s ; cd -"  % (card, WS_output, floating_ttV))

        run_cmd("cd "+os.getcwd()+"/"+mom_result+" ; combine -M Significance --signif %s.root %s %s > %s.log  ; cd -"  % (WS_output, blindStatement, redefineToTTH, WS_output))

        run_cmd("cd "+os.getcwd()+"/"+mom_result+" ; combine -M MultiDimFit %s.root %s --setParameters r_ttH=1,r_ttW=1,r_ttZ=1 --algo singles -P r_ttH --floatOtherPOI=1 > %s_rate_ttH.log  --cminDefaultMinimizerType Minuit --keepFailures ; cd -"  % (WS_output, blindStatement, WS_output))

        run_cmd("cd "+os.getcwd()+"/"+mom_result+" ; combine -M MultiDimFit %s.root %s --setParameters r_ttH=1,r_ttW=1,r_ttZ=1 --algo singles  --cminDefaultMinimizerType Minuit --keepFailures > %s_rate_3D.log  ; cd -"  % (WS_output, blindStatement, WS_output))

    if (doHessImpactCombo and card == cardToWrite) or (doHessImpactCombo2017 and card == cardToWrite2017) :
        # hessian impacts

        run_cmd("cd "+os.getcwd()+"/"+mom_result+" ; combineTool.py -M Impacts -d %s.root --rMin -2 --rMax 5  %s   -m 125 --doFits --approx hesse ; cd - "  % (WS_output, blindStatement))

        run_cmd("cd "+os.getcwd()+"/"+mom_result+" ; combineTool.py -M Impacts -d %s.root -m 125 -o impacts.json --approx hesse --rMin -2 --rMax 5  %s; plotImpacts.py -i impacts.json -o  impacts ; mv impacts.pdf impacts_hesse_JESCorr%s ; cd -" % (WS_output, blindStatement, str(btag_correlated)))

    if doCategories  and card == cardToWrite_2017 :
        ### for category by category - 2017 only
        run_cmd("mkdir categories_"+card)
        enterHere = os.getcwd()+"/"+mom_result+"/categories_"+card
        WS_output_byCat = card+"_Catpoi_final"

        if not takeCombo :
            floating_by_cat = \
            "--PO 'map=.*2lss_e.*/ttH.*:r_ttH_2lss_0tau[1,-5,10]'\
PO 'map=.*2lss_m.*/ttH.*:r_ttH_2lss_0tau[1,-5,10]'\
PO 'map=.*3l_b.*/ttH.*:r_ttH_3l_0tau[1,-5,10]'\
PO 'map=.*4l_2.*/ttH.*:r_ttH_4l[1,-5,10]'\
PO 'map=.*2lss_1tau.*/ttH.*:r_ttH_2lss_1tau[1,-5,10]'\
PO 'map=.*3l_1tau.*/ttH.*:r_ttH_3l_1tau[1,-5,10]'\
PO 'map=.*2l_2tau.*/ttH.*:r_ttH_2l_2tau[1,-5,10]'\
PO 'map=.*1l_2tau.*/ttH.*:r_ttH_1l_2tau[1,-5,10]'\
"
        else :
            floating_by_cat = \
            "--PO 'map=.*2lss_e.*/ttH.*:r_ttH_2lss_0tau[1,-5,10]'\
--PO 'map=.*2lss_m.*/ttH.*:r_ttH_2lss_0tau[1,-5,10]'\
--PO 'map=.*3l_b.*/ttH.*:r_ttH_3l_0tau[1,-5,10]'\
PO 'map=.*3l_cr.*/ttH.*:r_ttH_3l_0tau[1,-5,10]'\
PO 'map=.*4l_.*/ttH.*:r_ttH_4l[1,-10,20]'\
PO 'map=.*1l_2tau.*/ttH.*:r_ttH_1l_2tau[1,-5,10]'\
PO 'map=.*2lss_1tau_.*/ttH.*:r_ttH_2lss_1tau[1,-5,10]'\
PO 'map=.*3l_1tau_.*/ttH.*:r_ttH_3l_1tau[1,-5,10]'\
PO 'map=.*2l_2tau_.*/ttH.*:r_ttH_2l_2tau[1,-5,10]'\
"

        run_cmd("cd "+enterHere+" ; text2workspace.py ../%s.txt -o %s.root -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose %s %s; cd -"  % (card, WS_output_byCat, floating_ttV, floating_by_cat))

        parameters = "r_ttW=1,r_ttZ=1"
        for rate in ["r_ttH_2lss_0tau", "r_ttH_3l_0tau", "r_ttH_4l", "r_ttH_2lss_1tau", "r_ttH_3l_1tau", "r_ttH_2l_2tau", "r_ttH_1l_2tau", "r_ttW", "r_ttZ"] :
            parameters = parameters+","+rate+"=1"
        print "Will fit the parameters "+parameters

        for rate in ["r_ttH_2lss_0tau", "r_ttH_3l_0tau", "r_ttH_4l", "r_ttH_2lss_1tau", "r_ttH_3l_1tau", "r_ttH_2l_2tau", "r_ttH_1l_2tau", "r_ttW", "r_ttZ"  ] :

            run_cmd("cd "+enterHere+" ; combine -M MultiDimFit %s.root %s --setParameters %s --algo singles -P %s --floatOtherPOI=1  --cminDefaultMinimizerType Minuit --keepFailures > %s_rate_%s.log ; cd -"  % (WS_output_byCat, blindStatement, parameters, rate, WS_output_byCat, rate)) #

    if (card == cardToWrite and doImpactCombo) or (card == cardToWrite_2017 and doImpact2017) :
        ### For impacts 2017 + 2016 only
        ## it creates many files !!!!
        if not sendToCondor or impactsSubmit :
            run_cmd("mkdir impacts_"+card)
            enterHere = os.getcwd()+"/"+mom_result+"/impacts_"+card
            run_cmd("cd "+enterHere+" ; combineTool.py -M Impacts -m 125 -d %s.root  --setParameters r_ttH=1,r_ttW=1,r_ttZ=1 --redefineSignalPOI r_ttH  --parallel 8 %s --doInitialFit  --keepFailures ; cd - "  % (WS_output, blindStatement))
            run_cmd("cd "+enterHere+" ; combineTool.py -M Impacts -m 125 -d %s.root  --setParameters r_ttH=1,r_ttW=1,r_ttZ=1 --redefineSignalPOI r_ttH  --parallel 8 %s --robustFit 1 --doFits %s ; cd - "  % (WS_output, blindStatement, sendToCondor))
        if not sendToCondor or not impactsSubmit :
            blindedOutputOpt = ' '
            if blindedOutput : blindedOutputOpt =  ' --blind'
            run_cmd("cd "+enterHere+" ; combineTool.py -M Impacts -m 125 -d %s.root  -o impacts.json    %s ; plotImpacts.py -i impacts.json %s -o impacts_btagCorr%s_blinded%s  ; cd -" % (WS_output, redefineToTTH, str(blindedOutputOpt), str(btag_correlated), str(blinded)))

    if (card == cardToWrite and doGOFCombo) or (card == cardToWrite_2017 and doGOF2017) :
        ## it creates many files !!!!
        run_cmd("mkdir gof_"+card)
        enterHere = os.getcwd()+"/"+mom_result+"/gof_"+card
        if sendToCondor :
            ### if you are submitting to condor you need to do in 2 steps, the second step collect the toys
            if GOF_submit :
                run_cmd("cd "+enterHere+' ;  combineTool.py -M GoodnessOfFit --algo=saturated  %s %s.root ; cd -' % (redefineToTTH, WS_output))
                filesh = open(enterHere+"/submit_gof.sh","w")
                filesh.write(
                    "#!/bin/bash\n"+\
                    "for ii in {1..500}\n" # this makes 1000 toys
                    "do\n"
                    "  r=$(( $RANDOM % 10000 ))\n"
                    "  #echo $r \n"
                    "  combineTool.py -M GoodnessOfFit --algo=saturated  "+redefineToTTH+"  -t 2 -s $r -n .toys$ii "+enterHere+"/"+WS_output+".root  --saveToys --toysFreq "+sendToCondor+" \n"
                    "done\n"
                    )
                run_cmd(os.getcwd()+"/"+mom_result+"/GOF"+' ; bash submit_gof.sh ; cd -' )
            else : # CollectGoodnessOfFit
                run_cmd("combineTool.py -M CollectGoodnessOfFit --input higgsCombine.Test.GoodnessOfFit.mH120.root higgsCombine*.GoodnessOfFit.mH120.*.root -o gof.json")
                run_cmd("cd "+os.getcwd()+"/"+mom_result+" ;  $CMSSW_BASE/src/CombineHarvester/CombineTools/scripts/plotGof.py --statistic saturated --mass 120.0 gof.json -o GoF_saturated_"+WS_output+'_btagCorr'+str(btag_correlated)+'_blinded'+str(blinded)+" ; cd -")
        else : # do all toys in series
            run_cmd("cd "+enterHere+' ;  combineTool.py -M GoodnessOfFit --algo=saturated  %s  %s.root ; cd -' % (redefineToTTH, WS_output))
            run_cmd("cd "+enterHere+' ; combineTool.py -M GoodnessOfFit --algo=saturated  %s  -t 1000 -s 12345  %s.root --saveToys --toysFreq ; cd -' % (redefineToTTH, WS_output))
            run_cmd("cd "+enterHere+' ; combineTool.py -M CollectGoodnessOfFit --input higgsCombineTest.GoodnessOfFit.mH120.root higgsCombineTest.GoodnessOfFit.mH120.12345.root -o gof.json ; cd -')
            run_cmd("cd "+enterHere+" ;  $CMSSW_BASE/src/CombineHarvester/CombineTools/scripts/plotGof.py --statistic saturated --mass 120.0 gof.json -o GoF_saturated_"+WS_output+'_btagCorr'+str(btag_correlated)+'_blinded'+str(blinded)+" ; cd -")

    savePostfitCombine  = "PostFitCombine_"+card
    savePostfitHavester = "PostFitHavester_"+card
    if preparePostFitCombine  and card == cardToWrite_2017 :
        ### for category by category - 2017 only
        run_cmd("mkdir "+savePostfitCombine)
        enterHere = os.getcwd()+"/"+mom_result+"/"+savePostfitCombine
        run_cmd("cd "+enterHere+' ; combineTool.py -M FitDiagnostics ../%s.root %s --saveNormalization --saveShapes --saveWIthUncertainties %s ; cd -' % (WS_output, redefineToTTH, sendToCondor))
        print ("the output with the shapes is going to be fitDiagnostics.Test.root or fitDiagnostics.root depending on your version of combine")

    if preparePostFitHavester  and card == cardToWrite_2017 :
        print ("[WARNING:] combine does not deal well with autoMCstats option for bin by bin stat uncertainty")
        run_cmd("mkdir "+savePostfitHavester)
        enterHere = os.getcwd()+"/"+mom_result+"/"+savePostfitHavester
        run_cmd("cd "+enterHere+' ; combineTool.py -M FitDiagnostics ../%s.root %s ; cd -' % (WS_output, redefineToTTH))
        print ("the diagnosis that input Havester is going to be on fitDiagnostics.Test.root or fitDiagnostics.root depending on your version of combine -- check if you have a crash!")
        doPostfit = " -f fitDiagnostics.root:fit_s --postfit "
        run_cmd("cd "+enterHere+' ; PostFitShapesFromWorkspace --workspace ../%s.root -d ../%s.txt -o %s_shapes.root -m 125 --sampling --print %s ; cd -' % (WS_output, card, WS_output, doPostfit)) # --skip-prefit
        print ("the output with the shapes is "+WS_output+"_shapes.root")

    if doYieldsAndPlots :
        import CombineHarvester.CombineTools.ch as ch
        #takeYields = card + "_3poi_ttVFromZero"
        takeYields = WS_output
        doPostfit = "none"
        if blinded : blindStatementPlot = '  '
        else : blindStatementPlot = ' --unblind '

        enterHere = os.getcwd()+"/"+mom_result
        if doPostfitCombine :
            doPostfit = savePostfitCombine
            fileShapes = "fitDiagnostics.Test.root"
            appendHavester = " "
        elif doPostfitHavester :
            doPostfit = savePostfitHavester
            fileShapes = WS_output+"_shapes.root"
            appendHavester = " --fromHavester "
        if doPostfit == "none" :
            run_cmd("cd "+enterHere+' ; combineTool.py -M FitDiagnostics ../%s.root %s ; cd -' % (WS_output, redefineToTTH))
        else : enterHere = enterHere+"/"+doPostfit

        run_cmd("cd "+enterHere+' ; python $CMSSW_BASE/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py -a fitDiagnostics.root -g plots.root  -p r_ttH  ; cd -')
        gSystem.Load('libHiggsAnalysisCombinedLimit')
        print ("Retrieving yields from workspace: ", os.getcwd()+"/"++takeYields)
        fin = TFile(os.getcwd()+"/"+mom_result+takeYields+".root")
        wsp = fin.Get('w')
        cmb = ch.CombineHarvester()
        cmb.SetFlag("workspaces-use-clone", True)
        ch.ParseCombineWorkspace(cmb, wsp, 'ModelConfig', 'data_obs', False)
        print "datacard parsed"
        import os
        print ("taking uncertainties from: "enterHere+'/fitDiagnostics.root')
        print ("the diagnosis that input Havester is going to be on fitDiagnostics.Test.root or fitDiagnostics.root depending on your version of combine -- check if you have a crash!")
        mlf = TFile(enterHere+'/fitDiagnostics.root')
        rfr = mlf.Get('fit_s')
        typeFit = " "
        for fit in ["prefit", "postfit"] :
            print fit+' tables:'
            if fit == "postfit" :
                cmb.UpdateParameters(rfr)
                print ' Parameters updated '
                typeFit = " --doPostFit "
            if not takeCombo :
                labels = [
                "1l_2tau_OS_mvaOutput_final_x_2017",
                "2l_2tau_sumOS_mvaOutput_final_x_2017",
                "3l_1tau_OS_mvaOutput_final_x_2017",
                "2lss_1tau_sumOS_mvaOutput_final_x_2017"
                ]
            else :
                labels=[
                "1l_2tau_OS",
                "2l_2tau_sumOS",
                "3l_1tau_OS",
                "2lss_1tau_sumOS"
                ]
            type = 'tau'
            colapseCat = False
            filey = open(os.getcwd()+"/"+mom_result+"yields_"+type+"_from_combo_"+fit+".tex","w")
            if fit == "prefit" : PrintTables_Tau(cmb, tuple(), filey, blindedOutput, labels, type)
            if fit == "postfit" : PrintTables_Tau(cmb, (rfr, 500), filey, blindedOutput, labels, type)
            print ("the yields are on this file: ", os.getcwd()+"/"+mom_result+"yields_"+type+"_from_combo_"+fit+".tex")
            if not doPostfit == "none" :
                optionsToPlot = [
                    ' --minY 0.07 --maxY 5000. --useLogPlot --notFlips ',
                    ' --minY -0.35 --maxY 14 --notFlips --notConversions ',
                    ' --minY 0.7 --maxY 500 --useLogPlot --MC_IsSplit --notFlips --divideByBinWidth ',
                    ' --minY 0.7 --maxY 5000 --useLogPlot --MC_IsSplit --divideByBinWidth '
                ]
                for ll, label in enumerate(labels) :
                    run_cmd('python makePostFitPlots_FromCombine.py --channel  %s  --input %s %s %s %s > ' % (label, enterHere+"/"+fileShapes, appendHavester, typeFit, optionsToPlot[ll], blindStatementPlot, enterHere+"/"+fileShapes+"_"+label+".log"))
            ######################################
            labels = [
                "2lss_ee_neg",
                "2lss_ee_pos",
                "2lss_em_bl_neg",
                "2lss_em_bl_pos",
                "2lss_mm_bl_neg",
                "2lss_mm_bl_pos",
                "2lss_em_bt_neg",
                "2lss_em_bt_pos",
                "2lss_mm_bt_neg",
                "2lss_mm_bt_pos",
                ]
            if takeCombo :
                for label in labels : label = label+"_2017"
            type = 'multilep2lss'
            colapseCat = True
            filey = open(os.getcwd()+"/"+mom_result+"yields_"+type+"_from_combo_"+fit+".tex","w")
            if fit == "prefit" : PrintTables_Tau(cmb, tuple(), filey, blindedOutput, labels, type)
            if fit == "postfit" : PrintTables_Tau(cmb, (rfr, 500), filey, blindedOutput, labels, type)
            print ("the yields are on this file: ", os.getcwd()+"/"+mom_result+"yields_"+type+"_from_combo_"+fit+".tex")
            #################################
            labels = [
            "3l_bl_neg",
            "3l_bl_pos",
            "3l_bt_neg",
            "3l_bt_pos",
            "4l"
            ]
            if takeCombo :
                for label in labels : label = label+"_2017"
            type = 'multilep3l4l'
            colapseCat = True
            filey = open(os.getcwd()+"/"+mom_result+"yields_"+type+"_from_combo_"+fit+".tex","w")
            if fit == "prefit" : PrintTables_Tau(cmb, tuple(), filey, blindedOutput, labels, type)
            if fit == "postfit" : PrintTables_Tau(cmb, (rfr, 500), filey, blindedOutput, labels, type)
            print ("the yields are on this file: ", os.getcwd()+"/"+mom_result+"yields_"+type+"_from_combo_"+fit+".tex")
            #################################
            labels = [
             "2lss_ee_neg_3j",
             "2lss_ee_pos_3j",
             "2lss_em_bl_neg_3j",
             "2lss_em_bl_pos_3j",
             "2lss_mm_bl_neg_3j",
             "2lss_mm_bl_pos_3j",
             "2lss_em_bt_neg_3j",
             "2lss_em_bt_pos_3j",
             "2lss_mm_bt_neg_3j",
             "2lss_mm_bt_pos_3j"
            ]
            if takeCombo :
                for label in labels : label = label+"_2017"
            type = 'multilepCR2lss'
            colapseCat = True
            filey = open(os.getcwd()+"/"+mom_result+"yields_"+type+"_from_combo_"+fit+".tex","w")
            if fit == "prefit" : PrintTables_Tau(cmb, tuple(), filey, blindedOutput, labels, type)
            if fit == "postfit" : PrintTables_Tau(cmb, (rfr, 500), filey, blindedOutput, labels, type)
            print ("the yields are on this file: ", os.getcwd()+"/"+mom_result+"yields_"+type+"_from_combo_"+fit+".tex")
            #################################
            labels = [
            "3l_bl_neg_zpeak",
            "3l_bl_pos_zpeak",
            "3l_bt_neg_zpeak",
            "3l_bt_pos_zpeak",
            "3l_crwz",
            "4l_crzz"
            ]
            if takeCombo :
                for label in labels : label = label+"_2017"
            type = 'multilepCR3l4l'
            colapseCat = True
            filey = open(os.getcwd()+"/"+mom_result+"yields_"+type+"_from_combo_"+fit+".tex","w")
            if fit == "prefit" : PrintTables_Tau(cmb, tuple(), filey, blindedOutput, labels, type)
            if fit == "postfit" : PrintTables_Tau(cmb, (rfr, 500), filey, blindedOutput, labels, type)
            print ("the yields are on this file: ", os.getcwd()+"/"+mom_result+"yields_"+type+"_from_combo_"+fit+".tex")

# python makePostFitPlots_FromCombine.py --minY -0.35 --maxY 14 --channel  "ttH_2l_2tau_sumOS"  --notConversions --notFlips --input /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/gpetrucc_2017/posfit_3poi_ttVFromZero/card_combo_2017_btag_correlated_3poi_ttVFromZero_fromWS_shapes.root --fromHavester

# python makePostFitPlots_FromCombine.py --minY 0.07 --maxY 5000. --channel  "ttH_1l_2tau_OS" --useLogPlot --notFlips --input /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/gpetrucc_2017/posfit_3poi_ttVFromZero/card_combo_2017_btag_correlated_3poi_ttVFromZero_fromWS_shapes.root --fromHavester

# ttH_3l_1tau_OS
# python makePostFitPlots_FromCombine.py --minY 0.7 --maxY 5000 --channel  "ttH_2lss_1tau_sumOS"  --useLogPlot --MC_IsSplit --divideByBinWidth --input /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/gpetrucc_2017/posfit_3poi_ttVFromZero/card_combo_2017_btag_correlated_3poi_ttVFromZero_fromWS_shapes.root --fromHavester

# python makePostFitPlots_FromCombine.py --minY 0.7 --maxY 500 --channel  "ttH_3l_1tau_OS"  --useLogPlot --MC_IsSplit --notFlips --divideByBinWidth --input /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/gpetrucc_2017/posfit_3poi_ttVFromZero/card_combo_2017_btag_correlated_3poi_ttVFromZero_fromWS_shapes.root --original /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/gpetrucc_2017/comb_2017v2_withCR_sanity.root --fromHavester --unblind

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"2l_2tau_sumOS\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,false,\"BDT\",\"\",0.0,12.0,false,\"prefit\",false,false,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"3l_1tau_OS\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",true,false,true,\" BDT\",\"\",0.1,500.0,true,\"prefit\",true,false,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"card_combo_2017_btag_correlated_3poi_ttVFromZero_fromWS\",\"gpetrucc_2017/posfit_3poi_ttVFromZero/\",\"2lss_1tau_sumOS\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",true,true,true,\" BDT\",\"\",0.5,10000.0,true,\"postfit\",true,false,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"1l_2tau_OS\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",true,false,true,\" BDT\",\"\",0.1,5000.0,false,\"prefit\",false,false,false)

##############################################

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"3l_bl_neg\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,true,\" BDT (ttH,tt/ttV)\",\"\",0.0,50.0,false,\"postfit\",false,true,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"3l_bl_pos\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,true,\" BDT (ttH,tt/ttV)\",\"\",0.0,50.0,false,\"postfit\",false,true,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"3l_bt_neg\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,true,\" BDT (ttH,tt/ttV)\",\"\",0.0,50.0,false,\"postfit\",false,true,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"3l_bt_pos\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,true,\" BDT (ttH,tt/ttV)\",\"\",0.0,50.0,false,\"postfit\",false,true,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"3l_crwz\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,true,\" BDT (ttH,tt/ttV)\",\"\",0.0,1500.0,false,\"postfit\",false,true,false)

#############################################
# 3l CR

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"3l_bl_neg_zpeak\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,true,\" BDT (ttH,tt/ttV)\",\"\",0.0,70.0,false,\"postfit\",false,true,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"3l_bl_pos_zpeak\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,true,\" BDT (ttH,tt/ttV)\",\"\",0.0,70.0,false,\"postfit\",false,true,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"3l_bt_neg_zpeak\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,true,\" BDT (ttH,tt/ttV)\",\"\",0.0,70.0,false,\"postfit\",false,true,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"3l_bt_pos_zpeak\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,true,\" BDT (ttH,tt/ttV)\",\"\",0.0,70.0,false,\"postfit\",false,true,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"3l_crwz\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,true,\" BDT (ttH,tt/ttV)\",\"\",0.0,1500.0,false,\"postfit\",false,true,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"4l_crzz\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,true,\" Selected events\",\"\",0.0,50.0,false,\"postfit\",false,true,false)
## ===> do after
#############

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"2lss_ee_pos\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,true,\" BDT (ttH,tt/ttV)\",\"\",0.0,20.0,false,\"prefit\",false,true,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"2lss_ee_neg\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,true,\" BDT (ttH,tt/ttV)\",\"\",0.0,20.0,false,\"postfit\",false,true,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"2lss_em_bl_pos\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,true,\" BDT (ttH,tt/ttV)\",\"\",0.0,20.0,false,\"postfit\",false,true,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"2lss_em_bl_neg\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,true,\" BDT (ttH,tt/ttV)\",\"\",0.0,20.0,false,\"postfit\",false,true,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"2lss_em_bt_pos\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,true,\" BDT (ttH,tt/ttV)\",\"\",0.0,20.0,false,\"postfit\",false,true,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"2lss_em_bt_neg\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,true,\" BDT (ttH,tt/ttV)\",\"\",0.0,20.0,false,\"postfit\",false,true,false)

#root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"2lss_mm_bl_pos\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,false,\" BDT (ttH,tt/ttV)\",\"\",0.0,20.0,false,\"postfit\",false,true,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"2lss_mm_bl_neg\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,false,\" BDT (ttH,tt/ttV)\",\"\",0.0,20.0,false,\"postfit\",false,true,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"2lss_mm_bt_pos\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,false,\" BDT (ttH,tt/ttV)\",\"\",0.0,20.0,false,\"postfit\",false,true,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"2lss_mm_bt_neg\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,false,\" BDT (ttH,tt/ttV)\",\"\",0.0,20.0,false,\"postfit\",false,true,false)

####################################

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"2lss_ee_neg_3j\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,true,\" BDT (ttH,tt/ttV)\",\"\",0.0,30.0,false,\"postfit\",false,true,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"2lss_ee_pos_3j\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,true,\" BDT (ttH,tt/ttV)\",\"\",0.0,30.0,false,\"postfit\",false,true,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"2lss_em_bl_neg_3j\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,true,\" BDT (ttH,tt/ttV)\",\"\",0.0,30.0,false,\"postfit\",false,true,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"2lss_em_bl_pos_3j\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,true,\" BDT (ttH,tt/ttV)\",\"\",0.0,30.0,false,\"postfit\",false,true,false)

#root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"2lss_em_bt_neg_3j\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,true,\" BDT (ttH,tt/ttV)\",\"\",0.0,30.0,false,\"postfit\",false,true,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"2lss_em_bt_pos_3j\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,true,\" BDT (ttH,tt/ttV)\",\"\",0.0,30.0,false,\"postfit\",false,true,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"2lss_mm_bl_neg_3j\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,false,\" BDT (ttH,tt/ttV)\",\"\",0.0,30.0,false,\"postfit\",false,true,false) &

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"2lss_mm_bl_pos_3j\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,false,\" BDT (ttH,tt/ttV)\",\"\",0.0,30.0,false,\"postfit\",false,true,false) &

#root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"2lss_mm_bt_neg_3j\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,false,\" BDT (ttH,tt/ttV)\",\"\",0.0,30.0,false,\"postfit\",false,true,false) &

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"2lss_mm_bt_pos_3j\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,true,\" BDT (ttH,tt/ttV)\",\"\",0.0,30.0,false,\"postfit\",false,true,false) &

    #os.getcwd()+"/" = os.getcwd()
    #os.getcwd()+"/" = os.getcwd()+"/"+"/"
    #print 'Post-fit tables:'
    #filey = open(os.getcwd()+"/"+mom_result+"yields_tau_from_combo_postfit.tex","w")
    #PrintTables_Tau(cmb, (rfr, 500), filey, blinded)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_ws_postfit_txt\",\"gpetrucc_2017/\",\"2l_2tau_sumOS\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,false,\"BDT\",\"\",0.0,12.0,false,\"postfit\",false,false,false)

# root -l -b -n -q /afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"comb_2017v2_withCR_sanity_3poi\",\"gpetrucc_2017/posfit_3poi_ttVFromZero/\",\"2l_2tau_sumOS\",\"/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/src/tth-bdt-training/treatDatacards/\",false,false,false,\"BDT\",\"\",0.0,12.0,false,\"postfit\",false,false,false)
