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

import pickle

from sklearn.externals import joblib
import root_numpy
from root_numpy import root2array, rec2array, array2root, tree2array

#from xgboost import XGBClassifier
import xgboost as xgb

#import numpy
import catboost as catboost #import CatBoostRegressor
#from catboost import CatBoostRegressor

#import seaborn as sns
#import gnosis.xml.pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import ROOT
from tqdm import trange
import glob
#channel="2lss_1tau"
channel="1l_2tau"
if channel=='1l_2tau':channelInTree='1l_2tau_OS_Tight'
bdtType="evtLevelTT_TTH"
trainvar="allVar" # 
#trainvar="oldVar" 
#trainvar="notForbidenVar"
#trainvar="notForbidenVarNoMEM"
inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Oct17/histograms/1l_2tau/forBDTtraining_OS/'

def trainVars(all):

        if trainvar=="allVar" and channel=="2lss_1tau" :return [
		'avg_dr_jet', 
		'dr_lep1_tau', 
		'dr_lep2_tau', 
		'dr_leps', 
		'htmiss', 
		'ptmiss',  
		#'lep1_frWeight', 
		#'lep1_isTight', 
		'lep1_pt',
		'lep2_pt', 
		'lep1_abs_eta',
		'lep2_abs_eta',
		'lep1_conePt',		
		'lep2_conePt', 
		#'lep2_frWeight', 
		#'lep2_isTight',  
		'lep1_tth_mva', 
		'lep2_tth_mva', 
		#'log_memOutput_tt', 
		#'log_memOutput_ttH', 
		#'log_memOutput_ttZ', 
		#'log_memOutput_ttZ_Zll', 
		#'lumiScale', 
		'mT_lep1', 
		'mT_lep2', 
		'mTauTauVis1', 
		'mTauTauVis2', 
		'max_lep_eta', 
		'memOutput_LR', 
		#'memOutput_isValid', 
		#'evtWeight', 
		#'memOutput_ttZ_Zll_LR', 
		'memOutput_tt_LR', 
		#'memOutput_type', 
		#'mindr_lep1_jet', 
		'mindr_lep2_jet', 
		'mindr_tau_jet', 
		'nBJetLoose', 
		'nBJetMedium', 
		'nJet', 
		'tau_abs_eta', 
		#'tau_frWeight', 
		#'tau_isTight', 
		'tau_mva', 
		'tau_pt' 
		#'target' 
		#'totalWeight'
		]

        if trainvar=="notForbidenVar" and channel=="2lss_1tau" :return [
		'avg_dr_jet', 
		'dr_lep1_tau', 
		'dr_lep2_tau', 
		'dr_leps', 
		'htmiss', 
		'ptmiss',  
		#'lep1_frWeight', 
		#'lep1_isTight', 
		#'lep1_pt',
		#'lep2_pt', 
		#'lep1_abs_eta',
		#'lep2_abs_eta',
		'lep1_conePt',		
		'lep2_conePt', 
		#'lep2_frWeight', 
		#'lep2_isTight',  
		#'lep1_tth_mva', 
		#'lep2_tth_mva', 
		#'log_memOutput_tt', 
		#'log_memOutput_ttH', 
		#'log_memOutput_ttZ', 
		#'log_memOutput_ttZ_Zll', 
		#'lumiScale', 
		#'mT_lep1', 
		'mT_lep2', 
		'mTauTauVis1', 
		'mTauTauVis2', 
		'max_lep_eta', 
		'memOutput_LR', 
		#'memOutput_isValid', 
		#'evtWeight', 
		#'memOutput_ttZ_Zll_LR', 
		'memOutput_tt_LR', 
		#'memOutput_type', 
		#'mindr_lep1_jet', 
		'mindr_lep2_jet', 
		'mindr_tau_jet', 
		#'nBJetLoose', 
		'nBJetMedium', 
		'nJet', 
		'tau_abs_eta'
		#'tau_frWeight', 
		#'tau_isTight', 
		#'tau_mva', 
		#'tau_pt' 
		#'target' 
		#'totalWeight'
		]

	if trainvar=="notForbidenVarNoMEM" and channel=="2lss_1tau" :return [
		'avg_dr_jet', 
		'dr_lep1_tau', 
		'dr_lep2_tau', 
		'dr_leps', 
		'htmiss', 
		'ptmiss',  
		#'lep1_frWeight', 
		#'lep1_isTight', 
		#'lep1_pt',
		#'lep2_pt', 
		#'lep1_abs_eta',
		#'lep2_abs_eta',
		'lep1_conePt',		
		'lep2_conePt', 
		#'lep2_frWeight', 
		#'lep2_isTight',  
		#'lep1_tth_mva', 
		#'lep2_tth_mva', 
		#'log_memOutput_tt', 
		#'log_memOutput_ttH', 
		#'log_memOutput_ttZ', 
		#'log_memOutput_ttZ_Zll', 
		#'lumiScale', 
		#'mT_lep1', 
		'mT_lep2', 
		'mTauTauVis1', 
		'mTauTauVis2', 
		'max_lep_eta', 
		'memOutput_LR', 
		#'memOutput_isValid', 
		#'evtWeight', 
		#'memOutput_ttZ_Zll_LR', 
		'memOutput_tt_LR', 
		#'memOutput_type', 
		#'mindr_lep1_jet', 
		'mindr_lep2_jet', 
		'mindr_tau_jet', 
		#'nBJetLoose', 
		'nBJetMedium', 
		'nJet', 
		'tau_abs_eta'
		#'tau_frWeight', 
		#'tau_isTight', 
		#'tau_mva', 
		#'tau_pt' 
		#'target' 
		#'totalWeight'
		]

        if trainvar=="oldVar"  and channel=="2lss_1tau" :return [
		'avg_dr_jet', 
		'dr_lep1_tau',  
		'dr_leps', 		
		'lep2_conePt',  
		'max_lep_eta', 
		'mindr_lep1_jet', 
		'nJet', 
		'tau_pt' 
		]

	if trainvar=="allVar" and channel=="1l_2tau" and all==True :return [
		'avg_dr_jet',
		'dr_lep_fittedHadTop',
		'dr_lep_tau_os',
		'dr_lep_tau_ss',
		'dr_taus',
		'evtWeight',
		'fittedHadTop_eta',
		'fittedHadTop_pt',
		'genWeight',
		'htmiss',
		'lep_conePt',
		'lep_eta',
		'lep_pt',
		'lep_tth_mva',
		'lumiScale',
		'mT_lep',
		'mTauTauVis',
		'mindr_lep_jet',
		'mindr_tau1_jet',
		'mindr_tau2_jet',
		'mvaOutput_hadTopTagger',
		'ptmiss',
		'tau1_eta',
		'tau1_mva',
		'tau1_pt',
		'tau2_eta',
		'tau2_mva',
		'tau2_pt',
		'nBJetLoose',
		'nBJetMedium',
		'nJet'
		'run',
		'lumi',
		'evt'
		]
		
	if trainvar=="allVar" and channel=="1l_2tau"  and all==False :return [
		'avg_dr_jet',
		'dr_lep_fittedHadTop',
		'dr_lep_tau_os',
		'dr_lep_tau_ss',
		'dr_taus',
		#'evtWeight',
		'fittedHadTop_eta',
		'fittedHadTop_pt',
		#'genWeight',
		'htmiss',
		'lep_conePt',
		'lep_eta',
		'lep_pt',
		'lep_tth_mva',
		#'lumiScale',
		'mT_lep',
		'mTauTauVis',
		'mindr_lep_jet',
		'mindr_tau1_jet',
		'mindr_tau2_jet',
		'mvaOutput_hadTopTagger',
		'ptmiss',
		'tau1_eta',
		'tau1_mva',
		'tau1_pt',
		'tau2_eta',
		'tau2_mva',
		'tau2_pt',
		'nBJetLoose',
		'nBJetMedium',
		'nJet'
		'run',
		'lumi',
		'evt'
		]

	if trainvar=="notForbidenVar" and channel=="1l_2tau" :return [
		'avg_dr_jet',
		'dr_lep_fittedHadTop',
		'dr_lep_tau_os',
		'dr_lep_tau_ss',
		'dr_taus',
		'evtWeight',
		'fittedHadTop_eta',
		'fittedHadTop_pt',
		'genWeight',
		'htmiss',
		'lep_conePt',
		'lep_eta',
		'lep_pt',
		'lep_tth_mva',
		'lumiScale',
		'mT_lep',
		'mTauTauVis',
		'mindr_lep_jet',
		'mindr_tau1_jet',
		'mindr_tau2_jet',
		'mvaOutput_hadTopTagger',
		'ptmiss',
		'tau1_eta',
		'tau1_mva',
		'tau1_pt',
		'tau2_eta',
		'tau2_mva',
		'tau2_pt',
		'nBJetLoose',
		'nBJetMedium',
		'nJet',
		'run',
		'lumi',
		'evt'
		]

	if trainvar=="HadTopTaggerVar" and channel=="1l_2tau" :return [
		'avg_dr_jet',
		'dr_lep_fittedHadTop',
		'dr_lep_tau_os',
		'dr_lep_tau_ss',
		'dr_taus',
		'evtWeight',
		'fittedHadTop_eta',
		'fittedHadTop_pt',
		'genWeight',
		'htmiss',
		'lep_conePt',
		'lep_eta',
		'lep_pt',
		'lep_tth_mva',
		'lumiScale',
		'mT_lep',
		'mTauTauVis',
		'mindr_lep_jet',
		'mindr_tau1_jet',
		'mindr_tau2_jet',
		'mvaOutput_hadTopTagger',
		'ptmiss',
		'tau1_eta',
		'tau1_mva',
		'tau1_pt',
		'tau2_eta',
		'tau2_mva',
		'tau2_pt',
		'nBJetLoose',
		'nBJetMedium',
		'nJet',
		'run',
		'lumi',
		'evt'
		]

		
"""
data_signal = pandas.read_csv(inputPath+channel+"_ttHToNonbb_21Oct2017.csv")
data_signal["target"] = 1

if channel=="2lss_1tau" : data_signal["totalWeight"] = data_signal.evtWeight * data_signal.tau_frWeight * data_signal.lep1_frWeight * data_signal.lep2_frWeight 
if channel=="1l_2tau" : data_signal["totalWeight"] = data_signal.evtWeight  
print (data_signal["totalWeight"].sum())
data_signal["totalWeight"] = 10000*data_signal["totalWeight"] /data_signal["totalWeight"].sum()

#data_signal["evtWeight"] = data_signal["evtWeight"] /data_signal["evtWeight"].sum()
#data_signal = data_signal[data_signal.eval("(memOutput_errorFlag==0)")]

data_tt = pandas.read_csv(inputPath+channel+"_TTTo2L2Nu_21Oct2017.csv") 
data_tt.append(inputPath+channel+"_TTToSemilepton_21Oct2017.csv")
data_tt["target"] = 0
if channel=="2lss_1tau" : data_tt["totalWeight"] = data_tt.evtWeight * data_tt.tau_frWeight * data_tt.lep1_frWeight * data_tt.lep2_frWeight #
if channel=="1l_2tau" : data_tt["totalWeight"] = data_tt.evtWeight  
print (data_tt["totalWeight"].sum())
data_tt["totalWeight"] = 10000*data_tt["totalWeight"] /data_tt["totalWeight"].sum()
#data_tt["evtWeight"] = data_tt["evtWeight"] /data_tt["evtWeight"].sum()
#data_tt = data_tt[data_tt.eval("(memOutput_errorFlag==0)")]
nS = len(data_signal)
nB = len(data_tt)
print "length of sig, bkg: ", nS, nB
data = pandas.concat([data_signal, data_tt])
print "length of sig+bkg: ", len(data)
#targets = np.concatenate((np.ones(nS),np.zeros(nB)))
#data_train = np.array(data[trainVars(False)])
#data_weights = np.array(data['evtWeight'])

#data["totalWeight"] = data.evtWeight * data.tau_frWeight * data.lep1_frWeight * data.lep2_frWeight   
#data.ix[data.target.values == 0, "totalWeight"] = data["totalWeight"]/

# evtWeight *= ( prob_chargeMisId_lead + prob_chargeMisId_sublead);
data_weights = np.array(data['totalWeight'])
weights = "totalWeight"
print (list(data))
#print (data.index)
#data["mva1"] = 0.0
#data["mva2"] = 0.0

"""

my_cols_list=trainVars(True)+['key','target'] #,'tau_frWeight','lep1_frWeight','lep1_frWeight'
# those last are onlz for channels where selection is relaxed, but it does not hurt to add
data = pandas.DataFrame(columns=my_cols_list)
#data.reindex(columns=[*df.columns.tolist(), *my_cols_list], fill_value=0)

keys=['ttHToNonbb','TTTo2L2Nu','TTToSemilepton']
for folderName in keys :
	print (folderName)
	if 'TT' in folderName : 
		sampleName='TT'
		target=0
	if folderName=='ttHToNonbb' : 
		sampleName='signal'
		target=1
	inputTree = channelInTree+'/sel/evtntuple/'+sampleName+'/evtTree'	
	outfile = inputPath+channel+'_'+folderName+'_21Oct2017.csv' #%sampleName
	#
	procP1=glob.glob(inputPath+"/"+folderName+"_fastsim_p1/"+folderName+"_fastsim_p1_forBDTtraining_OS_central_*.root")
	procP2=glob.glob(inputPath+"/"+folderName+"_fastsim_p2/"+folderName+"_fastsim_p2_forBDTtraining_OS_central_*.root")
	procP3=glob.glob(inputPath+"/"+folderName+"_fastsim_p3/"+folderName+"_fastsim_p3_forBDTtraining_OS_central_*.root")
	#print (procP1)
	list=procP1+procP2+procP3
	print ("Date: ", time.asctime( time.localtime(time.time()) ))
	for ii in trange(0, len(list)) : #
		tfile = ROOT.TFile(list[ii])
		tree = tfile.Get(inputTree)
		#print (list[ii])
		if tree is not None :
			chunk_arr = tree2array(tree) #,  start=start, stop = stop)
			chunk_df = pandas.DataFrame(chunk_arr) #
			#print (len(chunk_df))
			chunk_df['key']=folderName
			chunk_df['target']=target
			if channel=="2lss_1tau" : data["totalWeight"] = data.evtWeight * data.tau_frWeight * data.lep1_frWeight * data.lep2_frWeight 
			if channel=="1l_2tau" : data["totalWeight"] = data.evtWeight 
			data=data.append(chunk_df, ignore_index=True)
			#store.append(folderName, chunk_df, chunksize=tree.GetEntries())
			#with open_file(outfile,'r') as infile: chunk_df.to_csv(outfile,index=False,header=False,mode='a', chunksize=tree.GetEntries()) 
		tfile.Close()
print (data.columns.values.tolist())
n = len(data)
nS = len(data.ix[data.target.values == 0])
nB = len(data.ix[data.target.values == 1])
print "length of sig, bkg: ", nS, nB
print ("weigths", data.loc[data['target']==0]["totalWeight"].sum() , data.loc[data['target']==1]["totalWeight"].sum() )
################################################################################# 
# balance datasets
#https://stackoverflow.com/questions/34803670/pandas-conditional-multiplication
for tar in [0,1] : data.loc[data['target']==tar, ['totalWeight']] *= 100000/data.loc[data['target']==tar]["totalWeight"].sum()
print ("balanced weigths", data.loc[data['target']==0]["totalWeight"].sum() , data.loc[data['target']==1]["totalWeight"].sum() )


weights="totalWeight"
#print data.loc[data['target']==1]["totalWeight"]
print data.loc[data['target']==0]["totalWeight"]
# print data.loc[data['totalWeight']==np.nan, ['nJet','run','lumi','evt']] .values 

# drop events with NaN weights
#data.dropna(thresh=0)
data.dropna(subset=["totalWeight"], inplace = True)
print data.loc[data['target']==0]["totalWeight"]

nS = len(data.ix[data.target.values == 0])
nB = len(data.ix[data.target.values == 1])
print "length of sig, bkg without NaN: ", nS, nB
#################################################################################
### Plot some histograms
################################################################################# 
### against QCD
hist_params = {'normed': True, 'bins': 18, 'alpha': 0.4}
plt.figure(figsize=(30, 30))
for n, feature in enumerate(trainVars(False)):
    # add sub plot on our figure
	plt.subplot(6, 6, n+1)
    # define range for histograms by cutting 1% of data from both ends
	if n == 0 or n == 1 or n == 2 or n == 4 or n == 5 : min_value, max_value = np.percentile(data[feature], [1, 99])
	else : min_value, max_value = np.percentile(data[feature], [1, 99])
	values, bins, _ = plt.hist(data.ix[data.target.values == 0, feature].values , weights= data.ix[data.target.values == 0, weights].values ,  
                               range=(min_value, max_value), label="TT", **hist_params )
	values, bins, _ = plt.hist(data.ix[data.target.values == 1, feature].values, weights= data.ix[data.target.values == 1, weights].values , 
                               range=(min_value, max_value), label='Signal', **hist_params)
	areaSig = sum(np.diff(bins)*values) 
	#print areaBKG, " ",areaBKG2 ," ",areaSig
	if n == 0 : plt.legend(loc='best')
	plt.title(feature)
plt.savefig(channel+"/"+bdtType+"_"+trainvar+"_Variables_BDT.pdf")
plt.clf()
#########################################################################################
#print (data.ix[data.target.values == 0, weights])
#print (data.ix[data.target.values == 1, weights])
traindataset, valdataset  = train_test_split(data, test_size=0.5, random_state=7)


#raindataset, valdataset, t_train, t_test, w_train, w_test = train_test_split(data, np.array(data['target']), data_weights, test_size=0.5, random_state=7)
#cls.fit(data_train, targets)
#ls.fit(traindataset, t_train, w_train)



keys=["evtLevelTT_TTH","evtLevelTTW_TTH"]

## do the report 
evals_result={}
param=[('max_depth', 6), ('objective', 'binary:logistic'), # ('eval_metric', 'logloss'), 
	('n_estimators', 100),('min_child_weight', 30)] 
num_round = 2
#cls = xgb(param)
"""
evals_result = {}
dtrain = xgb.DMatrix(traindataset, label = traindataset["target"], 
	weight = (traindataset[weights].astype(np.float64)), silent=True)
dtest = xgb.DMatrix(valdataset, label = valdataset["target"],
	weight = (valdataset[weights].astype(np.float64)), silent=True)
watchlist  = [(dtest,'eval'), (dtrain,'train')]
cls=xgb.train(param, dtrain, num_round, watchlist , evals_result=evals_result)
print (evals_result)
model = cls.get_dump() # #pickle.dumps(cls)
"""

if trainvar=="oldVar" : cls = xgb.XGBClassifier(n_estimators = 2000, max_depth = 2, min_child_weight = 1, learning_rate = 0.01) #,max_depth=20,n_estimators=50,learning_rate=0.5)
if trainvar=="notForbidenVar" : cls = xgb.XGBClassifier(n_estimators = 2000, max_depth = 2, min_child_weight = 2, learning_rate = 0.01) #,max_depth=20,n_estimators=50,learning_rate=0.5)
if trainvar=="notForbidenVarNoMEM" : cls = xgb.XGBClassifier(n_estimators = 2000, max_depth = 2, min_child_weight = 2, learning_rate = 0.01)  #,max_depth=20,n_estimators=50,learning_rate=0.5)
if trainvar=="allVar" : cls = xgb.XGBClassifier(n_estimators = 2000, max_depth = 2, min_child_weight = 2, learning_rate = 0.01)  #,max_depth=20,n_estimators=50,learning_rate=0.5)


gridSearchXGB=False
if gridSearchXGB==True :
	param_grid = {
				#'n_estimators': [1200,1500],
				#'min_child_weight': [10,20,30],
				'max_depth': [2,4,6],  
				'learning_rate': [0.01,0.02,0.03]
				}
	scoring = "roc_auc"
	early_stopping_rounds = None
	cv=3
	fit_params = { "eval_set" : [(traindataset[trainVars(False)].values,traindataset["target"])],
                           "eval_metric" : "roc_auc",
                           "early_stopping_rounds" : early_stopping_rounds }
	gs = GridSearchCV(cls, param_grid, scoring, fit_params, cv = cv, verbose = 1) 
	gs.fit(traindataset[trainVars(False)].values,traindataset["target"])
	for i, param in enumerate(gs.cv_results_["params"]): 
		print("params : {} \n    cv auc = {}  +- {} ".format(param,gs.cv_results_["mean_test_score"][i],gs.cv_results_["std_test_score"][i]))
	print(gs.best_params_)
	print(gs.best_score_)
	gs = dm.grid_search_cv(clf, param_grid = param_grid,early_stopping_rounds = None)

cls.fit(
	traindataset[trainVars(False)].values,  
	traindataset.target.astype(np.bool),  
	sample_weight= (traindataset[weights].astype(np.float64)),
	eval_set=[(traindataset[trainVars(False)].values,  traindataset.target.astype(np.bool),traindataset[weights].astype(np.float64)),
	(valdataset[trainVars(False)].values,  valdataset.target.astype(np.bool), valdataset[weights].astype(np.float64))] ,  
	verbose=True,eval_metric="auc"
	)

#model = cls.booster().get_dump(fmap='', with_stats=False) #.get_dump() #pickle.dumps(cls)
#xgboost2tmva.convert_model(model, trainVars(False), channel+"_XGB_wMEMallVars.xml")

# xmllint --format TMVABDT_2lss_1tau_XGB_wMEMallVars.xml

print ("XGBoost trained") 
proba = cls.predict_proba(traindataset[trainVars(False)].values  )
fpr, tpr, thresholds = roc_curve(traindataset["target"], proba[:,1] )
train_auc = auc(fpr, tpr, reorder = True) 
print("XGBoost train set auc - {}".format(train_auc)) 
proba = cls.predict_proba(valdataset[trainVars(False)].values)
fprt, tprt, thresholds = roc_curve(valdataset["target"], proba[:,1] )
test_auct = auc(fprt, tprt, reorder = True)
print("XGBoost test set auc - {}".format(test_auct))
""" 
sklearn_to_tmva.gbr_to_tmva(
    cls,
    data[trainVars(False)],
	trainVars(False),
    "TMVABDT_2lss_1tau_XGB_wMEMallVars.xml",
    coef=2
)
#"""
##################################################
if trainvar=="oldVar" :  clc = catboost.CatBoostClassifier(iterations=1800, depth=4, learning_rate=0.01, loss_function='Logloss',gradient_iterations=3,od_pval=0.01, verbose=True)
if trainvar=="notForbidenVar" : clc = catboost.CatBoostClassifier(iterations=2000, depth=2, learning_rate=0.01, loss_function='Logloss',od_pval=0.01, verbose=False)
if trainvar=="notForbidenVarNoMEM" : clc = catboost.CatBoostClassifier(iterations=1000, depth=3, learning_rate=0.01, loss_function='Logloss',gradient_iterations=3,od_pval=0.01, verbose=False)
if trainvar=="allVar" : clc = catboost.CatBoostClassifier(iterations=1500, depth=2, learning_rate=0.01, loss_function='Logloss',gradient_iterations=3,od_pval=0.01, verbose=False)

clc.fit(
	traindataset[trainVars(False)].values,  
	traindataset.target.astype(np.bool),  
	sample_weight= np.absolute((traindataset[weights].astype(np.float64))),
	#eval_set=[(traindataset[trainVars(False)].values,  traindataset.target.astype(np.bool),traindataset[weights].astype(np.float64)),
	#(valdataset[trainVars(False)].values,  valdataset.target.astype(np.bool), valdataset[weights].astype(np.float64))] 
	)
#print (cls.evals_result())
#print (cls.evals_result()['validation_0']['logloss'])
#"""
#model = pickle.dumps(clc) # clc.get_dump() #
#xgboost2tmva.convert_model(model, trainVars(False), "TMVABDT_2lss_1tau_CB_wMEMallVars.xml")
# xmllint --format TMVABDT_2lss_1tau_XGB_wMEMallVars.xml
print ("CatBoost trained") 
proba = clc.predict_proba(traindataset[trainVars(False)].values  )
fprc, tprc, thresholds = roc_curve(traindataset["target"], proba[:,1] )
train_aucc = auc(fprc, tprc, reorder = True) 
print("CatBoost train set auc - {}".format(train_aucc)) 
proba = clc.predict_proba(valdataset[trainVars(False)].values)
fprtc, tprtc, thresholds = roc_curve(valdataset["target"], proba[:,1] )
test_auctc = auc(fprtc, tprtc, reorder = True)
print("CatBoost test set auc - {}".format(test_auctc))
##################################################
clf = GradientBoostingClassifier(max_depth=3,learning_rate=0.01,n_estimators=100,verbose=True,min_samples_leaf=10,min_samples_split=10)
clf.fit(traindataset[trainVars(False)].values,  
	traindataset.target.astype(np.bool),  
	sample_weight= (traindataset[weights].astype(np.float64))
	)
sklearn_to_tmva.gbr_to_tmva(clf,data[trainVars(False)],trainVars(False),channel+"_GB_wMEMallVars.xml",coef=2)
print ("GradientBoosting trained")
proba = clf.predict_proba(traindataset[trainVars(False)].values  )
fprf, tprf, thresholdsf = roc_curve(traindataset["target"], proba[:,1] )
train_aucf = auc(fprf, tprf, reorder = True) 
print("GradientBoosting train set auc - {}".format(train_aucf)) 
proba = clf.predict_proba(valdataset[trainVars(False)].values)
fprtf, tprtf, thresholdsf = roc_curve(valdataset["target"], proba[:,1] )
test_auctf = auc(fprtf, tprtf, reorder = True)
print("GradientBoosting test set auc - {}".format(test_auctf)) 
##################################################
fig, ax = plt.subplots()
## ROC curve
ax.plot(fprf, tprf, lw=1, label='GB train (area = %0.3f)'%(train_aucf))
ax.plot(fprtf, tprtf, lw=1, label='GB test (area = %0.3f)'%(test_auctf))
ax.plot(fpr, tpr, lw=1, label='XGB train (area = %0.3f)'%(train_auc))
ax.plot(fprt, tprt, lw=1, label='XGB test (area = %0.3f)'%(test_auct))
ax.plot(fprc, tprc, lw=1, label='CB train (area = %0.3f)'%(train_aucc))
ax.plot(fprtc, tprtc, lw=1, label='CB test (area = %0.3f)'%(test_auctc))
ax.set_ylim([0.0,1.0])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
#ax.set_title('Test set ROC')
ax.legend(loc="lower right")
ax.grid()
fig.savefig("{}/{}_{}_roc.png".format(channel,bdtType,trainvar))
fig.savefig("{}/{}_{}_roc.pdf".format(channel,bdtType,trainvar))
###########################################################################
## feature importance plot
fig, ax = plt.subplots()
f_score_dict =cls.booster().get_fscore()
f_score_dict = {trainVars(False)[int(k[1:])] : v for k,v in f_score_dict.items()}
feat_imp = pandas.Series(f_score_dict).sort_values(ascending=True)
feat_imp.plot(kind='barh', title='Feature Importances')
fig.tight_layout()
fig.savefig("{}/{}_{}_XGB_importance.png".format(channel,bdtType,trainvar))
fig.savefig("{}/{}_{}_XGB_importance.pdf".format(channel,bdtType,trainvar))
###########################################################################
"""
## feature importance plot
fig, ax = plt.subplots()
f_score_dict =clf.booster().get_fscore()
f_score_dict = {trainVars(False)[int(k[1:])] : v for k,v in f_score_dict.items()}
feat_imp = pandas.Series(f_score_dict).sort_values(ascending=True)
feat_imp.plot(kind='barh', title='Feature Importances')
fig.tight_layout()
fig.savefig("{}/{}_{}_GB_importance.png".format(channel,bdtType,trainvar))
fig.savefig("{}/{}_{}_GB_importance.pdf".format(channel,bdtType,trainvar))

###########################################################################
## feature importance plot
fig, ax = plt.subplots()
f_score_dict =clc.get_feature_importance()
f_score_dict = {trainVars(False)[int(k[1:])] : v for k,v in f_score_dict.items()}
feat_imp = pandas.Series(f_score_dict).sort_values(ascending=True)
feat_imp.plot(kind='barh', title='Feature Importances')
fig.tight_layout()
fig.savefig("{}/{}_{}_CB_importance.png".format(channel,bdtType,trainvar))
fig.savefig("{}/{}_{}_CB_importance.pdf".format(channel,bdtType,trainvar))
"""
###########################################################################
#print (list(valdataset))
hist_params = {'normed': True, 'bins': 20 , 'histtype':'step'}
plt.clf()
y_pred = cls.predict_proba(valdataset.ix[valdataset.target.values == 0, trainVars(False)].values)[:, 1] #  
y_predS = cls.predict_proba(valdataset.ix[valdataset.target.values == 1, trainVars(False)].values)[:, 1] # 
plt.figure('XGB',figsize=(6, 6)) 
values, bins, _ = plt.hist(y_pred , label="TT (XGB)", **hist_params)
values, bins, _ = plt.hist(y_predS , label="signal", **hist_params )
#plt.xscale('log')
#plt.yscale('log')
plt.legend(loc='best')
plt.savefig(channel+'/'+bdtType+'_'+trainvar+'_XGBclassifier.pdf')  
###########################################################################
plt.clf()
y_pred = clc.predict_proba(valdataset.ix[valdataset.target.values == 0, trainVars(False)].values)[:, 1] #  
y_predS = clc.predict_proba(valdataset.ix[valdataset.target.values == 1, trainVars(False)].values)[:, 1] # 
plt.figure('CB',figsize=(6, 6)) 
values, bins, _ = plt.hist(y_pred , label="TT (CB)", **hist_params)
values, bins, _ = plt.hist(y_predS , label="signal", **hist_params )
#plt.xscale('log')
#plt.yscale('log')
plt.legend(loc='best')
plt.savefig(channel+'/'+bdtType+'_'+trainvar+'_CBclassifier.pdf')  
###########################################################################
plt.clf()
y_pred = clf.predict_proba(valdataset.ix[valdataset.target.values == 0, trainVars(False)].values)[:, 1] #  
y_predS = clf.predict_proba(valdataset.ix[valdataset.target.values == 1, trainVars(False)].values)[:, 1] # 
plt.figure( 'GB',figsize=(6, 6)) 
values, bins, _ = plt.hist(y_pred , label="TT (GB)", **hist_params)
values, bins, _ = plt.hist(y_predS , label="signal", **hist_params )
#plt.xscale('log')
#plt.yscale('log')
plt.legend(loc='best')
plt.savefig(channel+'/'+bdtType+'_'+trainvar+'_GBclassifier.pdf')  

"""
###########################################################################
plt.figure('Correlation Signal '+channel,figsize=(6, 6))
reportAll.features_correlation_matrix(\
    features=dataset.ix[(dataset.target.values == 1), trainVars(False)], cmap=cmap) #.plot(new_plot=True, show_legend=False, figsize=(4, 4))
plt.savefig(channel+'_CorrelationsMatrix_signal.pdf')  #_by_class
plt.clf() 
"""
########################################################################
# plot correlation matrix
print (len(trainVars(False)))

for ii in [1,2] :
	if ii == 1 :
		datad=data_signal
		label="signal"
	else :
		datad=data_tt
		label="BKG"
	data = datad[trainVars(False)] #.loc[:,trainVars(False)] #dataHToNobbCSV[[features]]
	correlations = data.corr()
	fig = plt.figure()
	
	ax = fig.add_subplot(111) 
	#ax.xticks(rotation=90)
	#ax.plot()
	cax = ax.matshow(correlations, vmin=-1, vmax=1)
	ticks = np.arange(0,len(trainVars(False)),1)
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)
	ax.set_xticklabels(trainVars(False),rotation=45)
	ax.set_yticklabels(trainVars(False),rotation=-45)
	fig.colorbar(cax)
	plt.savefig("{}/{}_{}_corr_{}.png".format(channel,bdtType,trainvar,label))
	plt.savefig("{}/{}_{}_corr_{}.pdf".format(channel,bdtType,trainvar,label))
	ax.clear()
###################################################################
"""
cmap = cm.get_cmap('jet', 30)
sns.set(style="white")
plt.clf()
plt.figure('Features',figsize=(6, 6))
# Compute the correlation matrix
corr = data_signal[trainVars(False)]
corrBKG = data_tt[trainVars(False)]
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig(channel+'_CorrelationsMatrix_signal.pdf')  #_by_class
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corrBKG, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig(channel+'_CorrelationsMatrix_TT.pdf')  #_by_class
plt.clf()
############################################################################
"""


#save the training to file for later use
filename = 'sklearn_2lss_1tau_maxDepth3_8Var_frWt_wMEMall.pkl'
obj=pickle.dump(cls, open(filename, 'wb'))
#xml_str = gnosis.xml.pickle.dumps(obj)
#o2 = gnosis.xml.pickle.loads('test.xml')
