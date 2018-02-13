import sys , time 
#import sklearn_to_tmva
import sklearn
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
import pandas
#from pandas import HDFStore,DataFrame
import math , array

import matplotlib
#matplotlib.use('PS')   # generate postscript output by default
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import numpy as np

import pickle
from tqdm import trange
from sklearn.externals import joblib
import root_numpy
from root_numpy import root2array, rec2array, array2root, tree2array
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import ROOT
from ROOT import TFile
from ROOT import TCut

def trainVars():
        return [
		'CSV_b',  
		'alphaKinFit', 
		'dR_Wj1Wj2', 
		'dR_bW',   
		'm_Wj1Wj2', 
		'm_bWj1Wj2', 
		'nllKinFit', 
		'pT_Wj1',  
		'pT_Wj1Wj2', 
		'pT_Wj2', 
		'pT_b', 
		'pT_bWj1Wj2', 
		'qg_Wj1', 
		'qg_Wj2'
		]


print ("Date: ", time.asctime( time.localtime(time.time()) ))
##################################################################
### pickle 
# HadTopTagger/HadTopTagger_XGB_allVar_lessBKG_CSV_screening.pkl
loaded_model = pickle.load(open('HadTopTagger_baseline_1000trees_allBKG/HadTopTagger_XGB_allVar_allBKG_CSV_screening.pkl', 'rb'))
target="bWj1Wj2_isGenMatched"

tfile = ROOT.TFile("/hdfs/local/acaan/HadTopTagger/2017Aug31/structured_histograms_harvested_stage1_hadTopTagger_ttHToNonbb_fastsim_p1.root")
tree0 = tfile.Get("TCVARS")
n_entries0=100000 # tree0.GetEntries()
counttrue=0
for ev in trange(0,n_entries0): #, desc="{} ({})".format(key, n_entries0)) :
	tree0.GetEntry(ev) 
	data = pandas.DataFrame()
	#data = data.assign(CSV_b=pandas.Series(tree0.CSV_b).values)
	data['CSV_b'] = pandas.Series(np.array(tree0.CSV_b))
	data['alphaKinFit'] = pandas.Series(np.array(tree0.alphaKinFit))
	data['dR_Wj1Wj2'] = pandas.Series(np.array(tree0.dR_Wj1Wj2))
	data['dR_bW'] = pandas.Series(np.array(tree0.dR_bW))
	data['m_Wj1Wj2'] = pandas.Series(np.array(tree0.m_Wj1Wj2))
	data['m_bWj1Wj2'] = pandas.Series(np.array(tree0.m_bWj1Wj2))
	data['nllKinFit'] = pandas.Series(np.array(tree0.nllKinFit))
	data['pT_Wj1'] = pandas.Series(np.array(tree0.pT_Wj1))
	data['pT_Wj1Wj2'] = pandas.Series(np.array(tree0.pT_Wj1Wj2))
	data['pT_Wj2'] = pandas.Series(np.array(tree0.pT_Wj2))
	data['pT_b'] = pandas.Series(np.array(tree0.pT_b))
	data['pT_bWj1Wj2'] = pandas.Series(np.array(tree0.pT_bWj1Wj2))
	data['qg_Wj1'] = pandas.Series(np.array(tree0.qg_Wj1))
	data['qg_Wj2'] = pandas.Series(np.array(tree0.qg_Wj2))
	data['bWj1Wj2_isGenMatched'] = pandas.Series(np.array(tree0.bWj1Wj2_isGenMatched))
	#result = loaded_model.score(data)
	#print(result)
	#"""
	#model = model.booster().get_dump(fmap='', with_stats=False) 
	#NTrees = len(model) 
	#
	proba = loaded_model.predict_proba(data[trainVars()].values)
	if data['bWj1Wj2_isGenMatched'][np.argmax(proba[:,1])] >0 : counttrue= counttrue+1
	#print (ii , data['bWj1Wj2_isGenMatched'][ii],proba[:,1][ii])
	#fprt, tprt, thresholds = roc_curve(data[target], proba[:,1] )
	#test_auct = auc(fprt, tprt, reorder = True)
	#print (len(data),test_auct)
	#"""
	# df1 = df1.assign(e=np.random.randn(sLength))
	#
print (counttrue,n_entries0, float(counttrue/n_entries0))
