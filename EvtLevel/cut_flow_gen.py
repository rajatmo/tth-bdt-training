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
key='TTWJetsToLNu'
dataTight=load_dataGen(inputPathTight,channelInTreeTight,genVars,[],testtruth,key)

# ['genPtTop', 'genPtTopB', 'genPtTopW', 'genPtTopWj1', 'genPtTopWj2', 'genEtaTop', 'genEtaTopB', 'genEtaTopW', 'genEtaTopWj1', 'genEtaTopWj2', 'genPhiTopB', 'genPhiTopWj1', 'genPhiTopWj2', 'genMTopB', 'genMTopWj1', 'genMTopWj2', 'genPtAntiTop', 'genPtAntiTopB', 'genPtAntiTopW', 'genPtAntiTopWj1', 'genPtAntiTopWj2', 'genEtaAntiTop', 'genEtaAntiTopB', 'genEtaAntiTopW', 'genEtaAntiTopWj1', 'genEtaAntiTopWj2', 'genPhiAntiTopB', 'genPhiAntiTopWj1', 'genPhiAntiTopWj2', 'genMAntiTopB', 'genMAntiTopWj1', 'genMAntiTopWj2', 'gencount', 'passtrigger']

if 1> 0 :
    datatotal=dataTight.ix[(dataTight.key.values == key)]
    ################################
    ntotal=len(datatotal)
    print (key, "total", len(datatotal))
    datatotal=datatotal.ix[((datatotal.genPtAntiTopWj1.values >0) | (datatotal.genPtTopWj1.values >0))]
    print (key, "total 1 Had-top", len(datatotal),float(len(datatotal))/ntotal)
    datatotal=dataTight.ix[(
                ((dataTight.genPtAntiTopWj1.values >0) & (dataTight.genPtTopWj1.values >0))
                ) & (dataTight.passtrigger.values >0)]
    print (key, "total 2 Had-top", len(datatotal),float(len(datatotal))/ntotal)
    ################
    datatotaltrig=dataTight.ix[(dataTight.passtrigger.values >0)]
    ntotalTrig=len(datatotaltrig)
    print (key, "trigger",  len(datatotal),float(len(datatotal))/ntotal)
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
                ((datatotaltrig.genPtAntiTopWj1.values >0) & (datatotaltrig.genPtAntiTopB.values >25) & (abs(datatotaltrig.genEtaAntiTopB.values) <2.5)) |
                ((datatotaltrig.genPtTopWj1.values >0) & (datatotaltrig.genPtTopB.values >25)  & (abs(datatotaltrig.genEtaTopB.values) <2.5))
                ) | (
                ((datatotaltrig.genPtAntiTopWj1.values >0) & (datatotaltrig.genPtAntiTopB.values >25)  & (abs(datatotaltrig.genEtaAntiTopB.values) <2.5)) &
                ((datatotaltrig.genPtTopWj1.values >0) & (datatotaltrig.genPtTopB.values >25)  & (abs(datatotaltrig.genEtaTopB.values) <2.5))
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
