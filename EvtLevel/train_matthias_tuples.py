import sys , time
#import sklearn_to_tmva
import sklearn
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
import pandas
#from pandas import HDFStore,DataFrame
import math
import sklearn_to_tmva
import xgboost2tmva
import skTMVA
import matplotlib
matplotlib.use('agg')
#matplotlib.use('PS')   # generate postscript output by default
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import numpy as np

from rep.estimators import TMVAClassifier

import pickle

from sklearn.externals import joblib
import root_numpy
from root_numpy import root2array, rec2array, array2root, tree2array

import xgboost as xgb
#import catboost as catboost #import CatBoostRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import ROOT
from tqdm import trange
import glob

from keras.models import Sequential, model_from_json
import json

from collections import OrderedDict

#inputPath='matthias_trees/ntuple_tight.root'
inputPath='matthias_trees/ntuple_tight.root'

#  KEY: TTree	ttZ_train;2	object title
#  KEY: TTree	ttZ_train;1	object title
ttjets_dl="ttjets_dl_pow_train"
ttjets_sl="ttjets_sl_pow_train"
ttH="ttH2Nonbb_125_train"

trainVars=[
"ht",#"htmiss",
"tau2lep1_visiblemass",#'mTauTauVis',
"tt_deltaR", #"dr_taus",   #tau1_pt tau2_pt
"jet_deltaRavg", #'avg_dr_jet',
"njets", #"nJet",
"ntags_loose", #"nBJetLoose", #
"tau1_pt",
"tau2_pt"
]

my_cols_list=trainVars+['target'+'w_generator'+'w_csvweight'+'w_puweight'+'w_csvweight'+"totalWeight"]
data = pandas.DataFrame(columns=my_cols_list)
tfile = ROOT.TFile(inputPath)

#############################
treedl = tfile.Get(ttjets_dl)
chunk_arrdl = tree2array(treedl)
chunk_dfdl = pandas.DataFrame(chunk_arrdl) #
#print (chunk_dfdl["jet_deltaRavg"])
chunk_dfdl['target']=0
chunk_dfdl["totalWeight"]=chunk_dfdl['w_generator']*chunk_dfdl['w_csvweight']*chunk_dfdl['w_puweight']*chunk_dfdl['w_csvweight']*chunk_dfdl['w_fake']
data=data.append(chunk_dfdl, ignore_index=True)
############################
treesl = tfile.Get(ttjets_sl)
chunk_arrsl = tree2array(treesl)
chunk_dfsl = pandas.DataFrame(chunk_arrdl) #
#print (chunk_dfsl["jet_deltaRavg"])
chunk_dfsl['target']=0
chunk_dfsl["totalWeight"]=chunk_dfsl['w_generator']*chunk_dfsl['w_csvweight']*chunk_dfsl['w_puweight']*chunk_dfsl['w_csvweight']*chunk_dfsl['w_fake']
data=data.append(chunk_dfsl, ignore_index=True)
############################
treettH = tfile.Get(ttH)
chunk_arrttH = tree2array(treettH)
chunk_dfttH = pandas.DataFrame(chunk_arrttH) #
#print (chunk_dfttH.columns.values.tolist())
#print (chunk_dfttH["jet_deltaRavg"])
chunk_dfttH['target']=1
chunk_dfttH["totalWeight"]=chunk_dfttH['w_generator']*chunk_dfttH['w_csvweight']*chunk_dfttH['w_puweight']*chunk_dfttH['w_csvweight']
data=data.append(chunk_dfttH, ignore_index=True)
##########################

nS = len(data.ix[(data.target.values == 0)])
nB = len(data.ix[(data.target.values == 1)])
print "length of sig, bkg: ", nS, nB

weights="totalWeight"
hist_params = {'normed': True, 'bins': 40, 'alpha': 0.4}
plt.figure(figsize=(50, 50))
for n, feature in enumerate(trainVars):
    # add sub plot on our figure
	plt.subplot(10, 10, n+1)
    # define range for histograms by cutting 1% of data from both ends
	min_value, max_value = np.percentile(data[feature], [0.5, 99])
	print (min_value, max_value,feature)
	values, bins, _ = plt.hist(abs(data.ix[data.target.values == 0, feature].values) ,
							   weights= abs(data.ix[data.target.values == 0, weights].values.astype(np.float64)) ,
                               range=(min_value, max_value),
							   label="TT", **hist_params )
	values, bins, _ = plt.hist(abs(data.ix[data.target.values == 1, feature].values),
							   weights= abs(data.ix[data.target.values == 1, weights].values.astype(np.float64)) ,
                               range=(min_value, max_value), label='Signal', **hist_params)
	areaSig = sum(np.diff(bins)*values)
	#print areaBKG, " ",areaBKG2 ," ",areaSig
	if n == 0 : plt.legend(loc='best')
	plt.title(feature)
plt.savefig("ntuples_Matthias_Variables_BDT.pdf")
plt.clf()

traindataset, valdataset  = train_test_split(data[trainVars+["target","totalWeight"]], test_size=0.2, random_state=7)

clf = GradientBoostingClassifier(max_depth=3,learning_rate=0.01,n_estimators=2500,min_samples_leaf=100) # ,min_samples_leaf=10,min_samples_split=10
clf.fit(traindataset[trainVars].values,
	traindataset.target.astype(np.bool),
	sample_weight=(traindataset["totalWeight"].astype(np.float64))
	)
print ("GradientBoosting trained")
proba = clf.predict_proba(traindataset[trainVars].values  )
fprf, tprf, thresholdsf = roc_curve(traindataset["target"], proba[:,1] )
train_aucf = auc(fprf, tprf, reorder = True)
print("GradientBoosting train set auc - {}".format(train_aucf))
proba = clf.predict_proba(valdataset[trainVars].values)
fprtf, tprtf, thresholdsf = roc_curve(valdataset["target"], proba[:,1] )
test_auctf = auc(fprtf, tprtf, reorder = True)
print("GradientBoosting test set auc - {}".format(test_auctf))


"""
*Tree    :ttH2Nonbb_125_train: ntuple                                                 *
*Entries :    40577 : Total =        45177091 bytes  File  Size =   24565749 *
*        :          : Tree compression factor =   1.85                       *
******************************************************************************
*Br    0 :category  : category/I                                             *
*Entries :    40577 : Total  Size=     163460 bytes  File Size  =      23264 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   6.88     *
*............................................................................*
*Br    1 :category_lj : category_lj/I                                        *
*Entries :    40577 : Total  Size=     163493 bytes  File Size  =      21924 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   7.30     *
*............................................................................*
*Br    2 :event     : event/I                                                *
*Entries :    40577 : Total  Size=     163427 bytes  File Size  =      61386 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   2.61     *
*............................................................................*
*Br    3 :events    : events/F                                               *
*Entries :    40577 : Total  Size=     163438 bytes  File Size  =       1400 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression= 114.27     *
*............................................................................*
*Br    4 :ht        : ht/F                                                   *
*Entries :    40577 : Total  Size=     163394 bytes  File Size  =     142016 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.13     *
*............................................................................*
*Br    5 :ht_notau  : ht_notau/F                                             *
*Entries :    40577 : Total  Size=     163460 bytes  File Size  =     141879 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.13     *
*............................................................................*
*Br    6 :ht_old    : ht_old/F                                               *
*Entries :    40577 : Total  Size=     163438 bytes  File Size  =     142040 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.13     *
*............................................................................*
*Br    7 :jet1_csv  : jet1_csv/F                                             *
*Entries :    40577 : Total  Size=     163460 bytes  File Size  =     142632 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.12     *
*............................................................................*
*Br    8 :jet1_eta  : jet1_eta/F                                             *
*Entries :    40577 : Total  Size=     163460 bytes  File Size  =     148290 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.08     *
*............................................................................*
*Br    9 :jet1_pt   : jet1_pt/F                                              *
*Entries :    40577 : Total  Size=     163449 bytes  File Size  =     142762 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.12     *
*............................................................................*
*Br   10 :jet2_csv  : jet2_csv/F                                             *
*Entries :    40577 : Total  Size=     163460 bytes  File Size  =     142641 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.12     *
*............................................................................*
*Br   11 :jet2_eta  : jet2_eta/F                                             *
*Entries :    40577 : Total  Size=     163460 bytes  File Size  =     148395 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.08     *
*............................................................................*
*Br   12 :jet2_pt   : jet2_pt/F                                              *
*Entries :    40577 : Total  Size=     163449 bytes  File Size  =     141774 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.13     *
*............................................................................*
*Br   13 :jet_csv   : vector<float>                                          *
*Entries :    40577 : Total  Size=    1263798 bytes  File Size  =     814959 *
*Baskets :       44 : Basket Size=      32000 bytes  Compression=   1.53     *
*............................................................................*
*Br   14 :jet_deltaRavg ntags_loose w_generator w_csvweight w_puweight: w_csvweight : jet_deltaRavg/F                                    *
*Entries :    40577 : Total  Size=     163515 bytes  File Size  =     140321 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.14     *
*............................................................................*
*Br   15 :jet_deltaRmax : jet_deltaRmax/F                                    *
*Entries :    40577 : Total  Size=     163515 bytes  File Size  =     137988 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.16     *
*............................................................................*
*Br   16 :jet_deltaRmin : jet_deltaRmin/F                                    *
*Entries :    40577 : Total  Size=     163515 bytes  File Size  =     140799 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.14     *
*............................................................................*
*Br   17 :jet_eta   : vector<float>                                          *
*Entries :    40577 : Total  Size=    1263798 bytes  File Size  =     828921 *
*Baskets :       44 : Basket Size=      32000 bytes  Compression=   1.50     *
*............................................................................*
*Br   18 :jet_geneta : vector<float>                                         *
*Entries :    40577 : Total  Size=    1263948 bytes  File Size  =     770803 *
*Baskets :       44 : Basket Size=      32000 bytes  Compression=   1.62     *
*............................................................................*
*Br   19 :jet_genpt : vector<float>                                          *
*Entries :    40577 : Total  Size=    1263898 bytes  File Size  =     746001 *
*Baskets :       44 : Basket Size=      32000 bytes  Compression=   1.67     *
*............................................................................*
*Br   20 :jet_ideff : vector<int>                                            *
*Entries :    40577 : Total  Size=     860547 bytes  File Size  =     154977 *
*Baskets :       31 : Basket Size=      32000 bytes  Compression=   5.38     *
*............................................................................*
*Br   21 :jet_ideff_loose : vector<int>                                      *
*Entries :    40577 : Total  Size=     860769 bytes  File Size  =     141064 *
*Baskets :       31 : Basket Size=      32000 bytes  Compression=   5.92     *
*............................................................................*
*Br   22 :jet_mistag : vector<int>                                           *
*Entries :    40577 : Total  Size=     974692 bytes  File Size  =     165938 *
*Baskets :       35 : Basket Size=      32000 bytes  Compression=   5.78     *
*............................................................................*
*Br   23 :jet_pt    : vector<float>                                          *
*Entries :    40577 : Total  Size=    1263748 bytes  File Size  =     806204 *
*Baskets :       44 : Basket Size=      32000 bytes  Compression=   1.55     *
*............................................................................*
*Br   24 :jet_ptsum : jet_ptsum/F                                            *
*Entries :    40577 : Total  Size=     163471 bytes  File Size  =     143472 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.12     *
*............................................................................*
*Br   25 :jet_ptsum_scalar : jet_ptsum_scalar/F                              *
*Entries :    40577 : Total  Size=     163548 bytes  File Size  =     141813 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.13     *
*............................................................................*
*Br   26 :lep1_abseta : lep1_abseta/F                                        *
*Entries :    40577 : Total  Size=     163493 bytes  File Size  =     144021 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.11     *
*............................................................................*
*Br   27 :lep1_chargediso : lep1_chargediso/F                                *
*Entries :    40577 : Total  Size=     163537 bytes  File Size  =      27391 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   5.84     *
*............................................................................*
*Br   28 :lep1_dz   : lep1_dz/F                                              *
*Entries :    40577 : Total  Size=     163449 bytes  File Size  =     145590 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.10     *
*............................................................................*
*Br   29 :lep1_eta  : lep1_eta/F                                             *
*Entries :    40577 : Total  Size=     163460 bytes  File Size  =     147938 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.08     *
*............................................................................*
*Br   30 :lep1_geneta : lep1_geneta/F                                        *
*Entries :    40577 : Total  Size=     163493 bytes  File Size  =     147821 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.08     *
*............................................................................*
*Br   31 :lep1_genid : lep1_genid/F                                          *
*Entries :    40577 : Total  Size=     163482 bytes  File Size  =      19037 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   8.40     *
*............................................................................*
*Br   32 :lep1_genparentid : lep1_genparentid/F                              *
*Entries :    40577 : Total  Size=     163548 bytes  File Size  =      12373 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=  12.93     *
*............................................................................*
*Br   33 :lep1_genpt : lep1_genpt/F                                          *
*Entries :    40577 : Total  Size=     163482 bytes  File Size  =     142729 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.12     *
*............................................................................*
*Br   34 :lep1_id   : lep1_id/I                                              *
*Entries :    40577 : Total  Size=     163449 bytes  File Size  =      30341 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   5.27     *
*............................................................................*
*Br   35 :lep1_ip   : lep1_ip/F                                              *
*Entries :    40577 : Total  Size=     163449 bytes  File Size  =     144234 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.11     *
*............................................................................*
*Br   36 :lep1_match : lep1_match/I                                          *
*Entries :    40577 : Total  Size=     163482 bytes  File Size  =      20976 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   7.63     *
*............................................................................*
*Br   37 :lep1_mt   : lep1_mt/F                                              *
*Entries :    40577 : Total  Size=     163449 bytes  File Size  =     144527 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.11     *
*............................................................................*
*Br   38 :lep1_mva  : lep1_mva/F                                             *
*Entries :    40577 : Total  Size=     163460 bytes  File Size  =     125610 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.27     *
*............................................................................*
*Br   39 :lep1_neutraliso : lep1_neutraliso/F                                *
*Entries :    40577 : Total  Size=     163537 bytes  File Size  =      50001 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   3.20     *
*............................................................................*
*Br   40 :lep1_nontrig : lep1_nontrig/F                                      *
*Entries :    40577 : Total  Size=     163504 bytes  File Size  =     115540 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.38     *
*............................................................................*
*Br   41 :lep1_phi  : lep1_phi/F                                             *
*Entries :    40577 : Total  Size=     163460 bytes  File Size  =     148258 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.08     *
*............................................................................*
*Br   42 :lep1_pt   : lep1_pt/F                                              *
*Entries :    40577 : Total  Size=     163449 bytes  File Size  =     142798 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.12     *
*............................................................................*
*Br   43 :lep1_reliso : lep1_reliso/F                                        *
*Entries :    40577 : Total  Size=     163493 bytes  File Size  =      68194 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   2.35     *
*............................................................................*
*Br   44 :lep1jet_deltaRmin : lep1jet_deltaRmin/F                            *
*Entries :    40577 : Total  Size=     163559 bytes  File Size  =     140886 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.14     *
*............................................................................*
*Br   45 :lep1tau1_cosDeltaPhi : lep1tau1_cosDeltaPhi/F                      *
*Entries :    40577 : Total  Size=     163592 bytes  File Size  =     146428 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   46 :lep1tau1_deltaR : lep1tau1_deltaR/F                                *
*Entries :    40577 : Total  Size=     163537 bytes  File Size  =     142120 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.13     *
*............................................................................*
*Br   47 :lep1tau2_cosDeltaPhi : lep1tau2_cosDeltaPhi/F                      *
*Entries :    40577 : Total  Size=     163592 bytes  File Size  =     146655 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   48 :lep1tau2_deltaR : lep1tau2_deltaR/F                                *
*Entries :    40577 : Total  Size=     163537 bytes  File Size  =     142117 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.13     *
*............................................................................*
*Br   49 :lep1tauOS_cosDeltaPhi : lep1tauOS_cosDeltaPhi/F                    *
*Entries :    40577 : Total  Size=     163603 bytes  File Size  =     146528 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   50 :lep1tauOS_deltaR : lep1tauOS_deltaR/F                              *
*Entries :    40577 : Total  Size=     163548 bytes  File Size  =     142138 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.13     *
*............................................................................*
*Br   51 :lep1tauSS_cosDeltaPhi : lep1tauSS_cosDeltaPhi/F                    *
*Entries :    40577 : Total  Size=     163603 bytes  File Size  =     146552 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   52 :lep1tauSS_deltaR : lep1tauSS_deltaR/F                              *
*Entries :    40577 : Total  Size=     163548 bytes  File Size  =     142109 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.13     *
*............................................................................*
*Br   53 :lumi      : lumi/I                                                 *
*Entries :    40577 : Total  Size=     163416 bytes  File Size  =     100992 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.58     *
*............................................................................*
*Br   54 :met       : met/F                                                  *
*Entries :    40577 : Total  Size=     163405 bytes  File Size  =     143620 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.11     *
*............................................................................*
*Br   55 :mht       : mht/F                                                  *
*Entries :    40577 : Total  Size=     163405 bytes  File Size  =     143635 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.11     *
*............................................................................*
*Br   56 :nalltaus  : nalltaus/I                                             *
*Entries :    40577 : Total  Size=     163460 bytes  File Size  =       7192 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=  22.25     *
*............................................................................*
*Br   57 :nalltausjetfake : nalltausjetfake/I                                *
*Entries :    40577 : Total  Size=     163537 bytes  File Size  =       5314 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=  30.11     *
*............................................................................*
*Br   58 :nalltausreal : nalltausreal/I                                      *
*Entries :    40577 : Total  Size=     163504 bytes  File Size  =      13953 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=  11.47     *
*............................................................................*
*Br   59 :nelectrons : nelectrons/F                                          *
*Entries :    40577 : Total  Size=     163482 bytes  File Size  =      18078 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   8.85     *
*............................................................................*
*Br   60 :njets     : njets/F                                                *
*Entries :    40577 : Total  Size=     163427 bytes  File Size  =      35630 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   4.49     *
*............................................................................*
*Br   61 :njets_inclusive : njets_inclusive/F                                *
*Entries :    40577 : Total  Size=     163537 bytes  File Size  =      30302 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   5.28     *
*............................................................................*
*Br   62 :nmuons    : nmuons/F                                               *
*Entries :    40577 : Total  Size=     163438 bytes  File Size  =      18430 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   8.68     *
*............................................................................*
*Br   63 :notag1_pt : notag1_pt/F                                            *
*Entries :    40577 : Total  Size=     163471 bytes  File Size  =     143044 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.12     *
*............................................................................*
*Br   64 :notag2_pt : notag2_pt/F                                            *
*Entries :    40577 : Total  Size=     163471 bytes  File Size  =     133194 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.20     *
*............................................................................*
*Br   65 :notag_ptsum : notag_ptsum/F                                        *
*Entries :    40577 : Total  Size=     163493 bytes  File Size  =     143759 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.11     *
*............................................................................*
*Br   66 :notag_ptsum_scalar : notag_ptsum_scalar/F                          *
*Entries :    40577 : Total  Size=     163570 bytes  File Size  =     143509 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.11     *
*............................................................................*
*Br   67 :npv       : npv/F                                                  *
*Entries :    40577 : Total  Size=     163405 bytes  File Size  =      50808 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   3.15     *
*............................................................................*
*Br   68 :ntags     : ntags/F                                                *
*Entries :    40577 : Total  Size=     163427 bytes  File Size  =      23490 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   6.81     *
*............................................................................*
*Br   69 :ntags_loose w_generator w_csvweight w_puweight: w_csvweight : ntags_loose/F                                        *
*Entries :    40577 : Total  Size=     163493 bytes  File Size  =      27435 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   5.83     *
*............................................................................*
*Br   70 :ntaus     : ntaus/I                                                *
*Entries :    40577 : Total  Size=     163427 bytes  File Size  =       4511 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=  35.47     *
*............................................................................*
*Br   71 :ntausjetfake : ntausjetfake/I                                      *
*Entries :    40577 : Total  Size=     163504 bytes  File Size  =       2882 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=  55.51     *
*............................................................................*
*Br   72 :ntausreal : ntausreal/I                                            *
*Entries :    40577 : Total  Size=     163471 bytes  File Size  =      13656 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=  11.72     *
*............................................................................*
*Br   73 :run       : run/I                                                  *
*Entries :    40577 : Total  Size=     163405 bytes  File Size  =       1380 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression= 115.93     *
*............................................................................*
*Br   74 :tag1_pt   : tag1_pt/F                                              *
*Entries :    40577 : Total  Size=     163449 bytes  File Size  =     140735 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.14     *
*............................................................................*
*Br   75 :tag2_pt   : tag2_pt/F                                              *
*Entries :    40577 : Total  Size=     163449 bytes  File Size  =      80559 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.99     *
*............................................................................*
*Br   76 :tag_ptsum : tag_ptsum/F                                            *
*Entries :    40577 : Total  Size=     163471 bytes  File Size  =     141099 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.13     *
*............................................................................*
*Br   77 :tag_ptsum_scalar : tag_ptsum_scalar/F                              *
*Entries :    40577 : Total  Size=     163548 bytes  File Size  =     141202 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.13     *
*............................................................................*
*Br   78 :taggedjet_csv : vector<float>                                      *
*Entries :    40577 : Total  Size=     803440 bytes  File Size  =     350556 *
*Baskets :       30 : Basket Size=      32000 bytes  Compression=   2.28     *
*............................................................................*
*Br   79 :taggedjet_eta : vector<float>                                      *
*Entries :    40577 : Total  Size=     803440 bytes  File Size  =     387449 *
*Baskets :       30 : Basket Size=      32000 bytes  Compression=   2.06     *
*............................................................................*
*Br   80 :taggedjet_geneta : vector<float>                                   *
*Entries :    40577 : Total  Size=     803548 bytes  File Size  =     382922 *
*Baskets :       30 : Basket Size=      32000 bytes  Compression=   2.09     *
*............................................................................*
*Br   81 :taggedjet_genpt : vector<float>                                    *
*Entries :    40577 : Total  Size=     803512 bytes  File Size  =     369840 *
*Baskets :       30 : Basket Size=      32000 bytes  Compression=   2.16     *
*............................................................................*
*Br   82 :taggedjet_pt : vector<float>                                       *
*Entries :    40577 : Total  Size=     803404 bytes  File Size  =     373874 *
*Baskets :       30 : Basket Size=      32000 bytes  Compression=   2.14     *
*............................................................................*
*Br   83 :tau1_abseta : tau1_abseta/F                                        *
*Entries :    40577 : Total  Size=     163493 bytes  File Size  =     144441 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.11     *
*............................................................................*
*Br   84 :tau1_decaymode : tau1_decaymode/F                                  *
*Entries :    40577 : Total  Size=     163526 bytes  File Size  =      25461 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   6.28     *
*............................................................................*
*Br   85 :tau1_eta  : tau1_eta/F                                             *
*Entries :    40577 : Total  Size=     163460 bytes  File Size  =     148344 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.08     *
*............................................................................*
*Br   86 :tau1_gen_deltaR : tau1_gen_deltaR/F                                *
*Entries :    40577 : Total  Size=     163537 bytes  File Size  =     144380 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.11     *
*............................................................................*
*Br   87 :tau1_geneta : tau1_geneta/F                                        *
*Entries :    40577 : Total  Size=     163493 bytes  File Size  =     148364 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.08     *
*............................................................................*
*Br   88 :tau1_genid : tau1_genid/F                                          *
*Entries :    40577 : Total  Size=     163482 bytes  File Size  =       8242 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=  19.41     *
*............................................................................*
*Br   89 :tau1_genparentid : tau1_genparentid/F                              *
*Entries :    40577 : Total  Size=     163548 bytes  File Size  =      13318 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=  12.01     *
*............................................................................*
*Br   90 :tau1_genpt : tau1_genpt/F                                          *
*Entries :    40577 : Total  Size=     163482 bytes  File Size  =     142138 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.13     *
*............................................................................*
*Br   91 :tau1_genviseta : tau1_genviseta/F                                  *
*Entries :    40577 : Total  Size=     163526 bytes  File Size  =     145006 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.10     *
*............................................................................*
*Br   92 :tau1_genvispt : tau1_genvispt/F                                    *
*Entries :    40577 : Total  Size=     163515 bytes  File Size  =     138614 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.15     *
*............................................................................*
*Br   93 :tau1_iso3hits03 : tau1_iso3hits03/F                                *
*Entries :    40577 : Total  Size=     163537 bytes  File Size  =       1345 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression= 118.95     *
*............................................................................*
*Br   94 :tau1_iso3hits05 : tau1_iso3hits05/F                                *
*Entries :    40577 : Total  Size=     163537 bytes  File Size  =      30363 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   5.27     *
*............................................................................*
*Br   95 :tau1_isoMVA03 : tau1_isoMVA03/F                                    *
*Entries :    40577 : Total  Size=     163515 bytes  File Size  =      14147 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=  11.31     *
*............................................................................*
*Br   96 :tau1_isoMVA03_raw : tau1_isoMVA03_raw/F                            *
*Entries :    40577 : Total  Size=     163559 bytes  File Size  =     139515 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.15     *
*............................................................................*
*Br   97 :tau1_isoMVA05 : tau1_isoMVA05/F                                    *
*Entries :    40577 : Total  Size=     163515 bytes  File Size  =      31522 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   5.08     *
*............................................................................*
*Br   98 :tau1_leadingtrackpt : tau1_leadingtrackpt/F                        *
*Entries :    40577 : Total  Size=     163581 bytes  File Size  =      96654 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.66     *
*............................................................................*
*Br   99 :tau1_match : tau1_match/I                                          *
*Entries :    40577 : Total  Size=     163482 bytes  File Size  =       8033 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=  19.92     *
*............................................................................*
*Br  100 :tau1_nprongs : tau1_nprongs/F                                      *
*Entries :    40577 : Total  Size=     163504 bytes  File Size  =      13667 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=  11.71     *
*............................................................................*
*Br  101 :tau1_phi  : tau1_phi/F                                             *
*Entries :    40577 : Total  Size=     163460 bytes  File Size  =     148376 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.08     *
*............................................................................*
*Br  102 :  njets tau1_pt   : tau1_pt/F                                              *
*Entries :    40577 : Total  Size=     163449 bytes  File Size  =     141069 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.13     *
*............................................................................*
*Br  103 :tau1_vetoelectron : tau1_vetoelectron/F                            *
*Entries :    40577 : Total  Size=     163559 bytes  File Size  =      26470 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   6.04     *
*............................................................................*
*Br  104 :tau1_vetomuon : tau1_vetomuon/F                                    *
*Entries :    40577 : Total  Size=     163515 bytes  File Size  =       5717 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=  27.99     *
*............................................................................*
*Br  105 :tau1jet_deltaRmin : tau1jet_deltaRmin/F                            *
*Entries :    40577 : Total  Size=     163559 bytes  File Size  =     141161 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.13     *
*............................................................................*
*Br  106 :tau1lep1_deltaR : tau1lep1_deltaR/F                                *
*Entries :    40577 : Total  Size=     163537 bytes  File Size  =     142120 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.13     *
*............................................................................*
*Br  107 :tau1lep1_visiblemass : tau1lep1_visiblemass/F                      *
*Entries :    40577 : Total  Size=     163592 bytes  File Size  =     143165 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.12     *
*............................................................................*
*Br  108 :tau2_abseta : tau2_abseta/F                                        *
*Entries :    40577 : Total  Size=     163493 bytes  File Size  =     144321 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.11     *
*............................................................................*
*Br  109 :tau2_decaymode : tau2_decaymode/F                                  *
*Entries :    40577 : Total  Size=     163526 bytes  File Size  =      24713 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   6.47     *
*............................................................................*
*Br  110 :tau2_eta  : tau2_eta/F                                             *
*Entries :    40577 : Total  Size=     163460 bytes  File Size  =     148220 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.08     *
*............................................................................*
*Br  111 :tau2_gen_deltaR : tau2_gen_deltaR/F                                *
*Entries :    40577 : Total  Size=     163537 bytes  File Size  =     144481 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.11     *
*............................................................................*
*Br  112 :tau2_geneta : tau2_geneta/F                                        *
*Entries :    40577 : Total  Size=     163493 bytes  File Size  =     148215 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.08     *
*............................................................................*
*Br  113 :tau2_genid : tau2_genid/F                                          *
*Entries :    40577 : Total  Size=     163482 bytes  File Size  =      11416 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=  14.01     *
*............................................................................*
*Br  114 :tau2_genparentid : tau2_genparentid/F                              *
*Entries :    40577 : Total  Size=     163548 bytes  File Size  =      15512 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=  10.31     *
*............................................................................*
*Br  115 :tau2_genpt : tau2_genpt/F                                          *
*Entries :    40577 : Total  Size=     163482 bytes  File Size  =     142087 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.13     *
*............................................................................*
*Br  116 :tau2_genviseta : tau2_genviseta/F                                  *
*Entries :    40577 : Total  Size=     163526 bytes  File Size  =     141807 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.13     *
*............................................................................*
*Br  117 :tau2_genvispt : tau2_genvispt/F                                    *
*Entries :    40577 : Total  Size=     163515 bytes  File Size  =     136510 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.17     *
*............................................................................*
*Br  118 :tau2_iso3hits03 : tau2_iso3hits03/F                                *
*Entries :    40577 : Total  Size=     163537 bytes  File Size  =       1345 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression= 118.95     *
*............................................................................*
*Br  119 :tau2_iso3hits05 : tau2_iso3hits05/F                                *
*Entries :    40577 : Total  Size=     163537 bytes  File Size  =      30331 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   5.27     *
*............................................................................*
*Br  120 :tau2_isoMVA03 : tau2_isoMVA03/F                                    *
*Entries :    40577 : Total  Size=     163515 bytes  File Size  =      14012 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=  11.42     *
*............................................................................*
*Br  121 :tau2_isoMVA03_raw : tau2_isoMVA03_raw/F                            *
*Entries :    40577 : Total  Size=     163559 bytes  File Size  =     139515 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.15     *
*............................................................................*
*Br  122 :tau2_isoMVA05 : tau2_isoMVA05/F                                    *
*Entries :    40577 : Total  Size=     163515 bytes  File Size  =      31821 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   5.03     *
*............................................................................*
*Br  123 :tau2_leadingtrackpt : tau2_leadingtrackpt/F                        *
*Entries :    40577 : Total  Size=     163581 bytes  File Size  =      95630 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.67     *
*............................................................................*
*Br  124 :tau2_match : tau2_match/I                                          *
*Entries :    40577 : Total  Size=     163482 bytes  File Size  =      10999 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=  14.54     *
*............................................................................*
*Br  125 :tau2_nprongs : tau2_nprongs/F                                      *
*Entries :    40577 : Total  Size=     163504 bytes  File Size  =      11812 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=  13.54     *
*............................................................................*
*Br  126 :tau2_phi  : tau2_phi/F                                             *
*Entries :    40577 : Total  Size=     163460 bytes  File Size  =     148246 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.08     *
*............................................................................*
*Br  127 :njets tau1_pt tau2_pt   : tau2_pt/F                                              *
*Entries :    40577 : Total  Size=     163449 bytes  File Size  =     142042 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.13     *
*............................................................................*
*Br  128 :tau2_vetoelectron : tau2_vetoelectron/F                            *
*Entries :    40577 : Total  Size=     163559 bytes  File Size  =      28426 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   5.63     *
*............................................................................*
*Br  129 :tau2_vetomuon : tau2_vetomuon/F                                    *
*Entries :    40577 : Total  Size=     163515 bytes  File Size  =       8203 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=  19.50     *
*............................................................................*
*Br  130 :tau2jet_deltaRmin : tau2jet_deltaRmin/F                            *
*Entries :    40577 : Total  Size=     163559 bytes  File Size  =     141104 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.13     *
*............................................................................*
*Br  131 :tau2lep1_deltaR : tau2lep1_deltaR/F                                *
*Entries :    40577 : Total  Size=     163537 bytes  File Size  =     142117 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   1.13     *
*............................................................................*
*Br  132 :tau2lep1_visiblemass : tau2lep1_visiblemass/F                      *
*Br  133 :tt_cosDeltaPhi : tt_cosDeltaPhi/F                                  *
*Br  134 :tt_deltaR : tt_deltaR/F                                            *
*Br  135 :tt_sumpt  : tt_sumpt/F                                             *
*Br  136 :tt_visiblemass : tt_visiblemass/F                                  *
*Br  137 :untaggedjet_csv : vector<float>                                    *
*Br  138 :untaggedjet_eta : vector<float>                                    *
*Br  139 :untaggedjet_geneta : vector<float>                                 *
*Br  140 :untaggedjet_genpt : vector<float>                                  *
*Br  141 :untaggedjet_pt : vector<float>                                     *
*Br  142 :vtx_z     : vtx_z/F                                                *
*Br  143 :vtx_zerr  : vtx_zerr/F                                             *
*Br  144 :w_cms_tthl_btag_cerr1down : w_cms_tthl_btag_cerr1down/F            *
*Br  145 :w_cms_tthl_btag_cerr1up : w_cms_tthl_btag_cerr1up/F                *
*Br  146 :w_cms_tthl_btag_cerr2down : w_cms_tthl_btag_cerr2down/F            *
*Br  147 :w_cms_tthl_btag_cerr2up : w_cms_tthl_btag_cerr2up/F                *
*Br  148 :w_cms_tthl_btag_hfdown : w_cms_tthl_btag_hfdown/F                  *
*Br  149 :w_cms_tthl_btag_hfstats1down : w_cms_tthl_btag_hfstats1down/F      *
*Br  150 :w_cms_tthl_btag_hfstats1up : w_cms_tthl_btag_hfstats1up/F          *
*Br  151 :w_cms_tthl_btag_hfstats2down : w_cms_tthl_btag_hfstats2down/F      *
*Br  152 :w_cms_tthl_btag_hfstats2up : w_cms_tthl_btag_hfstats2up/F          *
*Br  153 :w_cms_tthl_btag_hfup : w_cms_tthl_btag_hfup/F                      *
*Br  154 :w_cms_tthl_btag_lfdown : w_cms_tthl_btag_lfdown/F                  *
*Br  155 :w_cms_tthl_btag_lfstats1down : w_cms_tthl_btag_lfstats1down/F      *
*Br  156 :w_cms_tthl_btag_lfstats1up : w_cms_tthl_btag_lfstats1up/F          *
*Br  157 :w_cms_tthl_btag_lfstats2down : w_cms_tthl_btag_lfstats2down/F      *
*Br  158 :w_cms_tthl_btag_lfstats2up : w_cms_tthl_btag_lfstats2up/F          *
*Br  159 :w_cms_tthl_btag_lfup : w_cms_tthl_btag_lfup/F                      *
*Br  160 :w_cms_tthl_frjt_normdown : w_cms_tthl_frjt_normdown/F              *
*Br  161 :w_cms_tthl_frjt_normup : w_cms_tthl_frjt_normup/F                  *
*Br  162 :w_cms_tthl_frjt_shapedown : w_cms_tthl_frjt_shapedown/F            *
*Br  163 :w_cms_tthl_frjt_shapeup : w_cms_tthl_frjt_shapeup/F                *
*Br  164 :w_cms_tthl_thu_shape_tth_x1down : w_cms_tthl_thu_shape_tth_x1down/F*
*Br  165 :w_cms_tthl_thu_shape_tth_x1up : w_cms_tthl_thu_shape_tth_x1up/F    *
*Br  166 :w_cms_tthl_thu_shape_tth_y1down : w_cms_tthl_thu_shape_tth_y1down/F*
*Br  167 :w_cms_tthl_thu_shape_tth_y1up : w_cms_tthl_thu_shape_tth_y1up/F    *
*Br  168 :w_cms_tthl_thu_shape_ttw_x1down : w_cms_tthl_thu_shape_ttw_x1down/F*
*Br  169 :w_cms_tthl_thu_shape_ttw_x1up : w_cms_tthl_thu_shape_ttw_x1up/F    *
*Br  170 :w_cms_tthl_thu_shape_ttw_y1down : w_cms_tthl_thu_shape_ttw_y1down/F*
*Br  171 :w_cms_tthl_thu_shape_ttw_y1up : w_cms_tthl_thu_shape_ttw_y1up/F    *
*Br  172 :w_cms_tthl_thu_shape_ttz_x1down : w_cms_tthl_thu_shape_ttz_x1down/F*
*Br  173 :w_cms_tthl_thu_shape_ttz_x1up : w_cms_tthl_thu_shape_ttz_x1up/F    *
*Br  174 :w_cms_tthl_thu_shape_ttz_y1down : w_cms_tthl_thu_shape_ttz_y1down/F*
*Br  175 :w_cms_tthl_thu_shape_ttz_y1up : w_cms_tthl_thu_shape_ttz_y1up/F    *
*Br  176 :w_generator w_csvweight w_puweight: w_csvweight/F                                        *
*Br  177 :w_fake    : w_fake/F                                               *
*Br  178 :w_generator : w_generator/F                                        *
*Br  179 :w_leptonsf : w_leptonsf/F                                          *
*Br  180 :w_puweight : w_puweight/F                                          *
*Br  181 :w_tauideff : w_tauideff/F                                          *
*Br  182 :w_triggersf : w_triggersf/F                                        *
"""
