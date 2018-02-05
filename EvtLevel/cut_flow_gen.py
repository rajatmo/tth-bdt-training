#import sklearn_to_tmva
import sklearn
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
import pandas
import math
import sklearn_to_tmva
import xgboost2tmva
import skTMVA
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import numpy as np

from rep.estimators import TMVAClassifier

import pickle

from sklearn.externals import joblib
import root_numpy
from root_numpy import root2array, rec2array, array2root, tree2array

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import ROOT
from tqdm import trange
import glob
import math , array

from keras.models import Sequential, model_from_json
import json
from collections import OrderedDict
execfile("../python/data_manager.py")

inputPathTight='/hdfs/local/acaan/ttHAnalysis/2016/1l_2tau_2018Jan23_forBDT_tightLtightT/histograms/1l_2tau/forBDTtraining_OS/'
criteria=[]
channelInTreeTight='1l_2tau_OS_Tight'
testtruth="bWj1Wj2_isGenMatchedWithKinFit"
genVars=["genPtTop",  "genPtTopB",  "genPtTopW",  "genPtTopWj1",  "genPtTopWj2",
"genEtaTop",  "genEtaTopB",  "genEtaTopW",  "genEtaTopWj1",  "genEtaTopWj2",
"genPhiTopB",  "genPhiTopWj1",  "genPhiTopWj2",
"genMTopB",  "genMTopWj1",  "genMTopWj2",
"genPtAntiTop",  "genPtAntiTopB",  "genPtAntiTopW",  "genPtAntiTopWj1",  "genPtAntiTopWj2",
"genEtaAntiTop",  "genEtaAntiTopB",  "genEtaAntiTopW",  "genEtaAntiTopWj1",  "genEtaAntiTopWj2",
"genPhiAntiTopB",  "genPhiAntiTopWj1",  "genPhiAntiTopWj2",
"genMAntiTopB",  "genMAntiTopWj1",  "genMAntiTopWj2","gencount","passtrigger"]
channel="1l_2tau"
keys=['ttHToNonbb','TTZToLLNuNu','TTWJetsToLNu','TTTo2L2Nu','TTToSemilepton']
key='ttHToNonbb'

tfile = ROOT.TFile('/hdfs/cms/store/user/atiko/VHBBHeppyV25tthtautau/MC/ttHJetToNonbb_M125_13TeV_amcatnloFXFX_madspin_pythia8_mWCutfix/VHBB_HEPPY_V25tthtautau_ttHJetToNonbb_M125_13TeV_amcatnloFXFX_madspin_Py8_mWCutfix__RunIISummer16MAv2-PUMoriond17_80r2as_2016_TrancheIV_v6_ext1-v1/170207_122849/0000/tree_1.root')

branches=["GenVbosons_charge",
"GenTop_pdgId", "GenTop_pt", "GenTop_eta", "GenTop_phi" , "GenTop_mass",
"GenWZQuark_pdgId", "GenWZQuark_pt", "GenWZQuark_eta", "GenWZQuark_phi" ,
"GenWZQuark_mass" , "GenWZQuark_charge", "GenWZQuark_status", "GenWZQuark_isPromptHard"
"GenBQuarkFromTop_pdgId", "GenBQuarkFromTop_pt", "GenBQuarkFromTop_eta", "GenBQuarkFromTop_phi", "GenBQuarkFromTop_mass",
"GenBQuarkFromTop_charge", "GenBQuarkFromTop_status", "GenBQuarkFromTop_isPromptHard" ]

tree = tfile.Get("tree")

pass1=0 #1)       2 genTopQuarks = 728 (weighted = 728)
pass2=0 #2)       2 genBJets = 728 (weighted = 728)
pass3=0 #3)       >= 2 genWBosons = 728 (weighted = 728)

pass4=0 #4)       >= 2 genWJets = 586 (weighted = 586)
pass5=0 #5)       genTopQuark && genAntiTopQuark = 586 (weighted = 586)
pass6=0 #6)       genBJetFromTop && genBJetFromAntiTop = 586 (weighted = 586)
pass7=0 #7)       genWBosonFromTop && genWBosonFromAntiTop = 586 (weighted = 586)  == not sure how you do that

pass8=0 #8)       genTopQuark mass = 575 (weighted = 575)
pass9=0 #9)       genAntiTopQuark mass = 569 (weighted = 569)
                  # I put 20 GeV window on the top that matches the WJets

pass10=0 #10)      2 genWJetsFromTop || 2 genWJetsFromAntiTop = 569 (weighted = 569)
                   # I read in arrays, this is useless

pass11=0 #11)      genWBosonFromTop mass = 528 (weighted = 528)
pass12=0 #12)      genWBosonFromAntiTop mass = 527 (weighted = 527)
                   # I skip that

pass13=0 #13)      genJet triplet = 527 (weighted = 527)
pass14=0 #14)      genJet triplet passes pT > 20 GeV = 512 (weighted = 512)
                   # I will consider that cut only one is enought

pass15=0 #15)      genJet triplet passes abs(eta) < 5.0 = 512 (weighted = 512)
pass16=0 #16)      genBJet passes abs(eta) < 2.4 = 511 (weighted = 511)

pass17=0 #17)      genJet triplet passes pT > 25 GeV = 501 (weighted = 501)
pass18=0 #18)      genJet triplet passes abs(eta) < 2.4 = 494 (weighted = 494)

pass19=0 #19)      dR(jet1,jet2) > 0.4 for any pair of genJets in triplet = 483 (weighted = 483)
pass20=0 #20)      selJet triplet = 202 (weighted = 202)
pass21=0 #21)      dR(jet1,jet2) > 0.3 for any pair of selJets in triplet = 200 (weighted = 200)
pass22=0 #22)      >= 1 selBJet passes loose b-tagging working-point = 197 (weighted = 197)

def sign(x): return 1 if x >= 0 else -1
#if ( mode == kGenTop ) {
#    pddIdTop    =  +6;
#    pddIdWBoson = +24;
#    pdgIdBJet   =  +5;
#  } else if ( mode == kGenAntiTop ) {
#    pddIdTop    =  -6;
#    pddIdWBoson = -24;
#    pdgIdBJet   =  -5;

Nhad=[]
for entry  in xrange(tree.GetEntries()) :
    tree.GetEntry(entry)
    Nhad=Nhad+[len(list(tree.GenWZQuark_charge))]
    if len(list(tree.GenTop_pdgId)) == 2 :
        pass1=pass1+1
    else : continue
    if len(list(tree.GenBQuarkFromTop_pdgId)) == 2 :
        pass2=pass2+1
    else : continue
    if len(list(tree.GenVbosons_charge)) > 1 :
        pass3=pass3+1
    else : continue
    if len(list(tree.GenWZQuark_charge)) >1 :
        pass4=pass4+1
    else : continue
    if sign(list(tree.GenTop_pdgId)[0]) == -sign(list(tree.GenTop_pdgId)[1]) :
        pass5=pass5+1
    else : continue
    if sign(list(tree.GenBQuarkFromTop_pdgId)[0]) == -sign(list(tree.GenBQuarkFromTop_pdgId)[1]) :
        pass6=pass6+1
    else : continue
    HadToppos=[]
    HadTopneg=[]
    bHadToppos=[]
    bHadTopneg=[]
    Wjpos=[]
    Wjneg=[]
    passmass=0
    for nn, top in enumerate(list(tree.GenTop_pdgId)) :
        if float(tree.GenTop_mass[nn]) > 165 and float(tree.GenTop_mass[nn] < 185 ) : passmass=passmass+1
        #if sign(top) > 0 : HadToppos = HadToppos+[nn]
        #if sign(top) < 0 : HadTopneg = HadTopneg+[nn]
    if passmass == 2 : pass8=pass8+1
    else : continue
    for nn,wjet in enumerate(list(tree.GenWZQuark_pdgId)) :
        Wj1dumb=ROOT.TLorentzVector()
        Wj1dumb.SetPtEtaPhiM(
            list(tree.GenWZQuark_pt)[nn] ,
            list(tree.GenWZQuark_eta)[nn] ,
            list(tree.GenWZQuark_phi)[nn] ,
            list(tree.GenWZQuark_mass)[nn]
            )
        if sign(wjet) > 0 : Wjpos = Wjpos+[Wj1dumb]
        if sign(wjet) < 0 : Wjneg = Wjneg+[Wj1dumb]
    for nn, bquark in enumerate(list(tree.GenBQuarkFromTop_pdgId)) :
        Bdumb=ROOT.TLorentzVector()
        Bdumb.SetPtEtaPhiM(
            list(tree.GenWZQuark_pt)[nn] ,
            list(tree.GenWZQuark_eta)[nn] ,
            list(tree.GenWZQuark_phi)[nn] ,
            list(tree.GenWZQuark_mass)[nn]
            )
        if sign(bquark) > 0 : bHadToppos = bHadToppos+[Bdumb]
        if sign(bquark) < 0 : bHadTopneg = bHadTopneg+[Bdumb]
    #if (len(Wjpos) > 0 and len(Wjneg) > 0 and len(HadToppos)==1 and len(bHadToppos)==1) or (len(Wjpos) > 0 and len(Wjneg) > 0 and len(HadTopneg)==1 and len(bHadTopneg)==1) :
        #pass13=pass13+1
    #print ("counting was ok",len(HadToppos)  , len(Wjpos) , len(bHadToppos), " o ", len(HadTopneg)  , len(Wjneg) , len(bHadTopneg))
    #else : continue
    if len(bHadToppos)>1 | len(bHadTopneg) > 1 : print "too many b-quarks"
    triplet_pos=ROOT.TLorentzVector()
    triplet_neg=ROOT.TLorentzVector()
    foundtriplet=0
    passpt=0
    passeta=0
    passptHard=0
    passetaHard=0
    passptB=0
    passetaB=0
    positive = 0
    for Wj1 in Wjpos :
        for Wj2 in Wjneg :
            if len(bHadToppos)==1 :
                triplet_pos = Wj1+Wj2+bHadToppos[0]
                if triplet_pos.M() < 185 or triplet_pos.M() > 165 :
                    notpassDRpos=0
                    if Wj1.Pt() > 20 and Wj2.Pt() > 20 and bHadToppos[0].Pt() >20 :
                        foundtriplet=foundtriplet+1
                        passpt=passpt+1
                        if abs(Wj1.Eta()) < 5 and abs(Wj2.Eta()) < 5 and abs(bHadToppos[0].Eta()) < 5 :
                            passeta=passeta+1
                            if abs(bHadToppos[0].Eta() < 2.4 ) :
                                passetaB=passetaB+1
                                if Wj1.Pt() > 25 and Wj2.Pt() > 25 and bHadToppos[0].Pt() >25 :
                                    passptHard=passptHard+1
                                    if abs(Wj1.Eta()) < 2.4 and abs(Wj2.Eta()) < 2.4 and abs(bHadToppos[0].Eta()) < 2.4 :
                                        passetaHard=passetaHard+1
                                        positive = 1
                                        if bHadToppos[0].DeltaR(Wj1) < 0.4 : notpassDRpos=notpassDRpos+1
                                        if bHadToppos[0].DeltaR(Wj2) < 0.4 : notpassDRpos=notpassDRpos+1
                                        if Wj2.DeltaR(Wj1) < 0.4 : notpassDRpos=notpassDRpos+1
                                        #if notpassDRpos !=0 : print("positive",bHadToppos[0].DeltaR(Wj1),bHadToppos[0].DeltaR(Wj2),Wj2.DeltaR(Wj1),notpassDRpos)
            if len(bHadTopneg)==1 :
                triplet_neg = Wj1+Wj2+bHadTopneg[0]
                if triplet_neg.M() < 185 or triplet_neg.M() > 165 :
                    notpassDRneg=0
                    if Wj1.Pt() > 20 and Wj2.Pt() > 20 and bHadTopneg[0].Pt() >20 :
                        foundtriplet=foundtriplet+1
                        passpt=passpt+1
                        if abs(Wj1.Eta()) < 5 and abs(Wj2.Eta()) < 5 and abs(bHadTopneg[0].Eta()) < 5 :
                            passeta=passeta+1
                            if abs(bHadTopneg[0].Eta() < 2.4 ) :
                                passetaB=passetaB+1
                                if Wj1.Pt() > 25 and Wj2.Pt() > 25 and bHadTopneg[0].Pt() >25 :
                                    passptHard=passptHard+1
                                    if abs(Wj1.Eta()) < 2.4 and abs(Wj2.Eta()) < 2.4 and abs(bHadTopneg[0].Eta()) < 2.4 :
                                        passetaHard=passetaHard+1
                                        positive = -1
                                        if bHadTopneg[0].DeltaR(Wj1) < 0.4 : notpassDRneg=notpassDRneg+1
                                        if bHadTopneg[0].DeltaR(Wj2) < 0.4 : notpassDRneg=notpassDRneg+1
                                        if  Wj2.DeltaR(Wj1) < 0.4 : notpassDRneg=notpassDRneg+1
                                        #if notpassDRneg !=0 : print("negative",bHadTopneg[0].DeltaR(Wj1),bHadTopneg[0].DeltaR(Wj2),Wj2.DeltaR(Wj1),notpassDRneg)

    if foundtriplet>0 :  pass13=pass13+1
    else : continue
    if passpt >0 : pass14=pass14+1
    else : continue
    if passeta >0 : pass15=pass15+1
    else : continue
    if passetaB >0 : pass16=pass16+1
    else : continue
    if passptHard > 0 : pass17=pass17+1
    else : continue
    if passetaHard > 0 : pass18=pass18+1
    else : continue
    if (positive==1 and notpassDRpos==0) or (positive==-1 and notpassDRneg==0) : pass19=pass19+1
    else : continue

print (tree.GetEntries(),pass1,pass2,pass3,pass4,pass5,pass6,pass8,pass13,pass14,pass15,pass16,pass17,pass18,pass19)

print ("1)",pass1, int(100*pass1/pass1), "% #       2 genTopQuarks = 728 (weighted = 728)")
print ("2)",pass2, int(100*pass2/pass1), "% #       2 genBJets = 728 (weighted = 728)")
print ("3)",pass3, int(100*pass3/pass1), "% #       >= 2 genWBosons = 728 (weighted = 728)")
print ("4)",pass4, int(100*pass4/pass1), "% #       >= 2 genWJets = 586 (weighted = 586)", int(100*586./728) ,"%")
print ("5)",pass5, int(100*pass5/pass1), "% #       genTopQuark && genAntiTopQuark = 586 (weighted = 586)", int(100*586./728) ,"%")
print ("6)",pass6, int(100*pass6/pass1), "% #       genBJetFromTop && genBJetFromAntiTop = 586 (weighted = 586)", int(100*586./728) ,"%")
print ("8)",pass8, int(100*pass8/pass1), "% #       genTopQuark mass  genAntiTopQuark mass # I put 20 GeV window ", int(100*569./728) ,"%")
print ("13)",pass13, int(100*pass13/pass1), "% #      genJet triplet = 527 (weighted = 527)", int(100*527./728) ,"%")
print ("14)",pass14, int(100*pass14/pass1), "% #      genJet triplet passes pT > 20 GeV = 512 (weighted = 512)", int(100*512./728) ,"%")
print ("15)",pass15, int(100*pass15/pass1), "% #      genJet triplet passes abs(eta) < 5.0 = 512 (weighted = 512)", int(100*512./728) ,"%")
print ("16)",pass16, int(100*pass16/pass1), "% #      genBJet passes abs(eta) < 2.4 = 511 (weighted = 511)", int(100*511./728) ,"%")
print ("17)",pass17, int(100*pass17/pass1), "% #      genJet triplet passes pT > 25 GeV = 501 (weighted = 501)", int(100*501./728) ,"%")
print ("18)",pass18, int(100*pass18/pass1), "% #      genJet triplet passes abs(eta) < 2.4 = 494 (weighted = 494)", int(100*494./728) ,"%")
print ("19)",pass19, int(100*pass19/pass1), "% #      dR(jet1,jet2) > 0.4 for any pair of genJets in triplet = 483 (weighted = 483)", int(100*483./728) ,"%")


if 0> 1 :
    dataTight=load_dataGen(inputPathTight,channelInTreeTight,genVars,[],testtruth,key)
    datatotal=dataTight.ix[(dataTight.key.values == key)]
    ################################
    ntotal=len(datatotal)
    print (key, "total", len(datatotal))
    datatotal=datatotal.ix[(
    (datatotal.genPtAntiTopWj1.values >0) | (datatotal.genPtTopWj1.values >0)
    )]
    print (key, "total 1 Had-top", len(datatotal),float(len(datatotal))/ntotal)
    datatotal=dataTight.ix[(
                ((dataTight.genPtAntiTopWj1.values >0) & (dataTight.genPtTopWj1.values >0))
                ) & (dataTight.passtrigger.values >0)]
    print (key, "total 2 Had-top", len(datatotal),float(len(datatotal))/ntotal)
    ################
    datatotaltrig=dataTight.ix[(dataTight.passtrigger.values >0)]
    ntotalTrig=len(datatotaltrig)
    print (key, "trigger",  len(datatotaltrig),float(len(datatotaltrig))/ntotalTrig)
    datatotaltrig1=datatotaltrig.ix[(
                ((datatotaltrig.genPtAntiTopWj1.values >0) & (datatotaltrig.genPtTopWj1.values >0)) |
                ((datatotaltrig.genPtTopWj1.values >0) & (datatotaltrig.genPtTopWj1.values >0))
                )]
    print (key, "total 1 or 2 Had-top + trigger",  len(datatotaltrig1),float(len(datatotaltrig1))/ntotalTrig)
    print (key, "total 0 Had-top", len(datatotaltrig.ix[((datatotaltrig.genPtAntiTopWj1.values >0) | (datatotaltrig.genPtTopWj1.values >0))==False ]))
    datatotaltrig1=datatotaltrig.ix[(
                ((datatotaltrig.genPtAntiTopWj1.values >0) & (datatotaltrig.genPtAntiTopB.values >25)) |
                ((datatotaltrig.genPtTopWj1.values >0) & (datatotaltrig.genPtTopB.values >25))
                ) | (
                ((datatotaltrig.genPtAntiTopWj1.values >0) & (datatotaltrig.genPtAntiTopB.values >25)) &
                ((datatotaltrig.genPtTopWj1.values >0) & (datatotaltrig.genPtTopB.values >25))
                )]
    print (key, "total 1 or 2 Had-top + trigger + bpt >25 GeV",  len(datatotaltrig1),float(len(datatotaltrig1))/ntotalTrig)
    datatotaltrig1=datatotaltrig.ix[(
                (
                   (datatotaltrig.genPtAntiTopWj1.values >0) & (datatotaltrig.genPtAntiTopB.values >25) &
                   (abs(datatotaltrig.genEtaAntiTopB.values) <2.5)
                   ) | (
                   (datatotaltrig.genPtTopWj1.values >0) & (datatotaltrig.genPtTopB.values >25)  &
                   (abs(datatotaltrig.genEtaTopB.values) <2.5)
                   )
                ) | (
                  (
                  (datatotaltrig.genPtAntiTopWj1.values >0) & (datatotaltrig.genPtAntiTopB.values >25)  &
                  (abs(datatotaltrig.genEtaAntiTopB.values) <2.5)
                  ) & (
                    (datatotaltrig.genPtTopWj1.values >0) & (datatotaltrig.genPtTopB.values >25)  &
                    (abs(datatotaltrig.genEtaTopB.values) <2.5)
                    )
                )]
    print (key, "total 1 or 2 Had-top + trigger + bpt >25 GeV + beta <2.5",  len(datatotaltrig1),float(len(datatotaltrig1))/ntotalTrig)
    datatotaltrig1=datatotaltrig.ix[(
                (
                   (datatotaltrig.genPtAntiTopWj1.values >25) &
                   (datatotaltrig.genPtAntiTopWj2.values >25) &
                   (datatotaltrig.genPtAntiTopB.values >25) &
                   (abs(datatotaltrig.genEtaAntiTopB.values) <2.5)
                ) | (
                   (datatotaltrig.genPtTopWj1.values >25) &
                   (datatotaltrig.genPtTopWj2.values >25) &
                   (datatotaltrig.genPtTopB.values >25)  & (abs(datatotaltrig.genEtaTopB.values) <2.5)
                  )
                ) | (
                (
                   (datatotaltrig.genPtAntiTopWj1.values >25) &
                   (datatotaltrig.genPtAntiTopWj2.values >25) &
                   (datatotaltrig.genPtAntiTopB.values >25) &
                   (abs(datatotaltrig.genEtaAntiTopB.values) <2.5)
                ) & (
                   (datatotaltrig.genPtTopWj1.values >25) &
                   (datatotaltrig.genPtTopWj2.values >25) &
                   (datatotaltrig.genPtTopB.values >25)  & (abs(datatotaltrig.genEtaTopB.values) <2.5)
                  )
                )]
    print (key, "total 1 or 2 Had-top + trigger + bpt >25 GeV + beta <2.5 + Wj>25",  len(datatotaltrig1),float(len(datatotaltrig1))/ntotalTrig)
    datatotaltrig1=datatotaltrig.ix[(
                    (
                    (datatotaltrig.genPtAntiTopWj1.values >25) & (abs(datatotaltrig.genEtaAntiTopWj1.values) <2.5) &
                    (datatotaltrig.genPtAntiTopWj2.values >25) & (abs(datatotaltrig.genEtaAntiTopWj2.values) <2.5) &
                    (datatotaltrig.genPtAntiTopB.values >25) & (abs(datatotaltrig.genEtaAntiTopB.values) <2.5)
                    ) | (
                        (datatotaltrig.genPtTopWj1.values >25) & (abs(datatotaltrig.genEtaTopWj1.values) <2.5) &
                        (datatotaltrig.genPtTopWj2.values >25) & (abs(datatotaltrig.genEtaTopWj2.values) <2.5) &
                        (datatotaltrig.genPtTopB.values >25)  & (abs(datatotaltrig.genEtaTopB.values) <2.5)
                    )
                ) | (
                    (
                    (datatotaltrig.genPtAntiTopWj1.values >25) & (abs(datatotaltrig.genEtaAntiTopWj1.values) <2.5) &
                    (datatotaltrig.genPtAntiTopWj2.values >25) & (abs(datatotaltrig.genEtaAntiTopWj2.values) <2.5) &
                    (datatotaltrig.genPtAntiTopB.values >25) & (abs(datatotaltrig.genEtaAntiTopB.values) <2.5)
                    ) & (
                    (datatotaltrig.genPtTopWj1.values >25) & (abs(datatotaltrig.genEtaTopWj1.values) <2.5) &
                    (datatotaltrig.genPtTopWj2.values >25) & (abs(datatotaltrig.genEtaTopWj2.values) <2.5) &
                    (datatotaltrig.genPtTopB.values >25) & (abs(datatotaltrig.genEtaTopB.values) <2.5)
                    )
                ) ]
    print (key, "total 1 or 2 Had-top + trigger + bpt >25 GeV + beta <2.5 + Wjpt>25 + Wjeta<2.5",  len(datatotaltrig1),float(len(datatotaltrig1))/ntotalTrig)
    #########
    ## plot DR
    datatotaltrig1['DeltaR']=  ((datatotaltrig.genEtaTopWj1-datatotaltrig.genEtaTopWj2).pow(2.)+(datatotaltrig.genPhiTopWj1-datatotaltrig.genPhiTopWj2).pow(2.)).pow(1./2)

    hist_params = {'normed': True, 'alpha': 0.4}
    plt.figure(figsize=(5, 5))
    feature='DeltaR'
    min_value, max_value = np.percentile(datatotaltrig1[feature], [1, 99])
    print (min_value, max_value,feature)
    values, bins, _ = plt.hist(datatotaltrig1[feature] ,
                               range=(min_value, max_value),
    						   label=key,
    						   bins=30,
    						   **hist_params )

    plt.legend(loc='best')
    plt.title(feature)
    plt.savefig("GenDR_"+key+"_BDT.pdf")
    plt.clf()
    datatotaltrig1Cone=datatotaltrig1.ix[(datatotaltrig1.DeltaR.values <0.6)]
    print (key, "total 1 or 2 Had-top + trigger + bpt >25 GeV + beta <2.5 + Wjpt>25 + Wjeta<2.5 , dr(wj1,wj2) < 0.6",  len(datatotaltrig1Cone),float(len(datatotaltrig1Cone))/ntotalTrig,float(len(datatotaltrig1Cone))/ntotal)



"""
TFile**		tthAnalysis/HiggsToTauTau/data/FR_tau_2016.root
 TFile*		tthAnalysis/HiggsToTauTau/data/FR_tau_2016.root
  TDirectoryFile*		jetToTauFakeRate	jetToTauFakeRate
   KEY: TDirectoryFile	dR05isoLoose;1	dR05isoLoose
   KEY: TDirectoryFile	dR05isoMedium;1	dR05isoMedium
   KEY: TDirectoryFile	dR05isoTight;1	dR05isoTight
   KEY: TDirectoryFile	dR03mvaMedium;1	dR03mvaMedium
   KEY: TDirectoryFile	dR03mvaTight;1	dR03mvaTight
   KEY: TDirectoryFile	dR03mvaVTight;1	dR03mvaVTight
   KEY: TDirectoryFile	dR03mvaVVTight;1	dR03mvaVVTight
  KEY: TDirectoryFile	jetToTauFakeRate;1	jetToTauFakeRate
"""
