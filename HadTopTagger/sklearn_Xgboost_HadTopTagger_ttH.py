import sys , time
#run: python sklearn_Xgboost_HadTopTagger_ttH.py --process 'all' --evaluateFOM --HypOpt --doXML &
import os
#os.environ['PYTHONUSERBASE'] = '/cvmfs/cms.cern.ch/slc6_amd64_gcc530/external/py2-scikit-learn/0.17.1-ikhhed'
#os.environ['PYTHONPATH'] = '/cvmfs/cms.cern.ch/slc6_amd64_gcc530/external/py2-scikit-learn/0.17.1-ikhhed/lib/python2.7/site-packages:'+os.environ['PYTHONPATH']
import sklearn
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split

import pandas
#from pandas import HDFStore,DataFrame
import math

#from mlglue.tree import tree_to_tmva, BDTxgboost, BDTsklearn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import numpy as np

import pickle

from sklearn.externals import joblib
import root_numpy
from root_numpy import root2array, rec2array, array2root, tree2array
print('The root_numpy version is {}.'.format(root_numpy.__version__))

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import ROOT
#from tqdm import trange
import glob


import xgboost as xgb
print('The xgb version is {}.'.format(xgb.__version__))

# if we have many trees
# https://stackoverflow.com/questions/38238139/python-prevent-ioerror-errno-5-input-output-error-when-running-without-stdo

from optparse import OptionParser
parser = OptionParser()
parser.add_option("--process ", type="string", dest="process", help="process", default='T')
parser.add_option("--evaluateFOM", action="store_true", dest="evaluateFOM", help="evaluateFOM", default=False)
parser.add_option("--HypOpt", action="store_true", dest="HypOpt", help="If you call this will not do plots with repport", default=False)
parser.add_option("--withKinFit", action="store_true", dest="withKinFit", help="BDT variables with kinfit", default=False)
parser.add_option("--doXML", action="store_true", dest="doXML", help="BDT variables with kinfit", default=False)
parser.add_option("--ntrees ", type="int", dest="ntrees", help="hyp", default=1000)
parser.add_option("--treeDeph", type="int", dest="treeDeph", help="hyp", default=3)
parser.add_option("--lr", type="float", dest="lr", help="hyp", default=0.01)
(options, args) = parser.parse_args()

channel="HadTopTagger_HypOpt" #options.channel #"1l_2tau"
inputPath='structured/'
process=options.process
keys=['ttHToNonbb','TTToSemilepton','TTZToLLNuNu','TTWJetsToLNu']
bdtType="CSV_sort"
withKinFit=options.withKinFit
doXML=options.doXML
trainvar="ntrees_"+str(options.ntrees)+"_deph_"+str(options.treeDeph)+"_lr_0o0"+str(int(options.lr*100)) #options.variables #

inputTree="TCVARSbfilter"
target="bWj1Wj2_isGenMatched"

import shutil,subprocess
proc=subprocess.Popen(['mkdir '+channel],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()

def trainVars(all):
	if all==True :
		return [ # ['CSV_b', 'qg_Wj2', 'pT_bWj1Wj2', 'm_Wj1Wj2', 'nllKinFit', 'pT_b_o_kinFit_pT_b', 'pT_Wj2']
				'CSV_b',
				'qg_Wj2',
				'pT_bWj1Wj2',
				'pT_Wj2',
				'm_Wj1Wj2',
				'nllKinFit',
				'pT_b_o_kinFit_pT_b'#,
		]
	if all==False :
				# ['CSV_b', 'qg_Wj2', 'qg_Wj1', 'm_bWj1Wj2', 'pT_bWj1Wj2', 'm_Wj1Wj2', 'pT_Wj2']
				return [
				'CSV_b',
				'qg_Wj2',
				'm_bWj1Wj2',
				'pT_bWj1Wj2',
				'pT_Wj2',
				'm_Wj1Wj2',
				'pT_Wj2',
				'ncombo',
				"cosThetaW_rest",
				"cosTheta_leadEWj_restTop",
				"cosThetaWj1_restW"
				]

def evaluateFOM(clf,keys,features,tag,train,test,nBdeplet,nB,nS,f_score_dicts):
	for process in keys :
		datatest=pandas.read_csv('structured/'+process+'_Structured_from_20000sig_1.csv')
		datatest['pT_b_o_kinFit_pT_b']=datatest['pT_b']/datatest['kinFit_pT_b']
		datatest['pT_Wj2_o_kinFit_pT_Wj2']=datatest['pT_Wj2']/datatest['kinFit_pT_Wj2']
		datatest['pT_Wj1_o_kinFit_pT_Wj1']=datatest['pT_Wj1']/datatest['kinFit_pT_Wj1']
		datatest['cosTheta_leadEWj_restTop'] = datatest['cosTheta_leadEWj_restTop'].abs()
		# make angles abs
		countTruth=0
		countEvt=0
		#print ("events raw: ",int(datatest['eventRaw'].min()),int(datatest['eventRaw'].max()))
		for ii in range(int(datatest['eventRaw'].min(axis=0)),int(datatest['eventRaw'].max())) :
			row=datatest.loc[datatest['eventRaw'].values == ii]
			if len(row)>0 :
				countEvt=countEvt+1
				row=datatest.loc[datatest['eventRaw'].values == ii]
				proba = clf.predict_proba(row[features].values)
				max= np.argmax(proba[:,1] )
				if row["bWj1Wj2_isGenMatched"].iloc[max] == 1 : countTruth=countTruth+1
		print ("process"+\
					" truthRatio(%)"+\
					" hyp Nfeat"+\
					" trainROC testROC ratioROC"+\
					" nB nBdeplet nS"+\
					" variables"+\
					" totEvt"+\
					" EvtThruth")
		print (str(process)+\
					" "+str(round(100*float(countTruth)/float(countEvt), 2))+\
					" "+trainvar+" "+str(len(features))+\
					" "+str(train)+" "+str(test)+" "+str(round(100.0*float(test)/train,2))+\
					" "+str(nB)+" "+str(nBdeplet)+" "+str(nS)+\
					" "+str(f_score_dicts)+\
					" "+str(countEvt)+\
					" "+str(countTruth))
		file = open(channel+'/'+options.process+'_in_'+process+'_'+bdtType+'_'+trainvar+'_tag_'+tag+'_XGB_FOM'+'_nvar'+str(len(trainVars(withKinFit)))+'.txt',"w")
		file.write(
					" "+str(round(100*float(countTruth)/float(countEvt), 2))+\
					" "+trainvar+" "+str(len(features))+\
					" "+str(train)+" "+str(test)+" "+str(round(100.0*float(test)/train,2))+\
					" "+str(nB)+" "+str(nBdeplet)+" "+str(nS)+\
					" "+str(f_score_dicts)+\
					" "+str(countEvt)+\
					" "+str(countTruth)
					)
		file.close()
		print ("Date: ", time.asctime( time.localtime(time.time()) ))
####################################################################################################
## Load data
data = pandas.DataFrame()
if process=="all" :
	for jj in range(0,len(keys)) :
		maxfile=1 #20
		for ii in range(0,maxfile) :
			print 'structured/'+str(keys[jj])+'_CSVsort_from_20000sig_'+str(ii)+'.csv'
			try : dumb=pandas.read_csv('structured/'+str(keys[jj])+'_CSVsort_from_20000sig_'+str(ii)+'.csv')
			except :
				print('Oops!',sys.exc_info()[0],'occured', str(keys[jj])+'_CSVsort_from_20000sig_'+str(ii)+'.csv')
				continue
			if len(dumb) > 0 :
				print (ii,str(keys[jj]),len(dumb.loc[dumb[target] == 0]),len(dumb.loc[dumb[target] == 1]))
				data=data.append(dumb, ignore_index=True)
		print ("partial ",keys[jj], len(data.loc[data[target]==0]) , len(data.loc[data[target]==1]) )
else :
	for ii in range(0,20) :
		print 'structured/'+process+'_CSVsort_from_10000sig_'+str(ii)+'.csv'
		try : dumb=pandas.read_csv('structured/'+process+'_CSVsort_from_10000sig_'+str(ii)+'.csv')
		except :
			print('Oops!',sys.exc_info()[0],'occured', process+'_CSVsort_from_10000sig_'+str(ii)+'.csv')
			break
		if len(dumb) > 0 : data=data.append(dumb, ignore_index=True)
data["totalWeight"] = 1
data['pT_b_o_kinFit_pT_b']=data['pT_b']/data['kinFit_pT_b']
data['pT_Wj2_o_kinFit_pT_Wj2']=data['pT_Wj2']/data['kinFit_pT_Wj2']
data['pT_Wj1_o_kinFit_pT_Wj1']=data['pT_Wj1']/data['kinFit_pT_Wj1']

data['ncombo']=1
print ("Do combo row: ", time.asctime( time.localtime(time.time()) ),int(data['eventRaw'].min()),int(data['eventRaw'].max()))
for ii in range(int(data['eventRaw'].min()),int(data['eventRaw'].max())): data.loc[data['eventRaw']==ii,'ncombo'] = len(data.loc[data['eventRaw']==ii])
print ("Did combo row: ", time.asctime( time.localtime(time.time()) ))

n = len(data)
nB = len(data.loc[data[target] == 0])
nS = len(data.loc[data[target] == 1])
print "length of sig, bkg: ", nS, nB
print ("weigths", data.loc[data[target]==0]["totalWeight"].sum() , data.loc[data[target]==1]["totalWeight"].sum() )
#################################################################################
print ("Throw away BKG: ", time.asctime( time.localtime(time.time()) ))
removeN=9.5*len(data.loc[data[target] == 0] )/10
drop_indices = np.random.choice(data.loc[data[target] == 0].index, int(removeN), replace=False)
data.loc[data[target] == 0] = data.loc[data[target] == 0].drop(drop_indices)
print ("weigths after throw away BKG", data.loc[data[target]==0]["totalWeight"].sum() , data.loc[data[target]==1]["totalWeight"].sum() )
nBdeplet=data.loc[data[target]==0]["totalWeight"].sum()
########################################################################################
## drop events with NaN weights = not needded now, but precaution
data.dropna(subset=[target],inplace = True) # data
print ("weigths after drop NaN", data.loc[data[target]==0]["totalWeight"].sum() , data.loc[data[target]==1]["totalWeight"].sum() )
#################################################################################
print ("Balance datasets ", time.asctime( time.localtime(time.time()) ))
#https://stackoverflow.com/questions/34803670/pandas-conditional-multiplication
print ("norm", data.loc[data[target]==0]["totalWeight"].sum(),data.loc[data[target]==1]["totalWeight"].sum())
for tar in [0,1] : data.loc[data[target]==tar, ["totalWeight"]] *= 100000./float(data.loc[data[target]==tar]["totalWeight"].sum())
weights="totalWeight"
#############################################################
## make angles absolute
angles=[
	"cosThetaW_rest",
	"cosTheta_leadEWj_restTop",
	"cosTheta_subleadEWj_restTop",
	"cosThetaWj1_restW"
	]
for angle in angles : data[angle] = data[angle].abs()
print "length of sig, bkg without NaN: ", nS, nB
#################################################################################
if options.HypOpt==False :
	print ("Plot histograms ", time.asctime( time.localtime(time.time()) ))
	hist_params = {'normed': False, 'bins': 18, 'alpha': 0.4}
	plt.figure(figsize=(20, 10))
	maxVar=[None] * len(trainVars(False))
	minVar=[None] * len(trainVars(False))
	for n, feature in enumerate(trainVars(True)):
		# add sub plot on our figure
		plt.subplot(2,4, n+1)
		# define range for histograms by cutting 1% of data from both ends
		min_value, max_value = np.percentile(data[feature], [1, 99])
		maxVar[n]=max_value
		minVar[n]=min_value
		if 'qg_' in feature :
			min_value=0.0
			max_value=1.0
		print (min_value, max_value,feature)
		values, bins, _ = plt.hist(data.ix[data[target].values == 0, feature].values , weights= data.ix[data[target].values == 0, weights].values ,
								   range=(min_value, max_value),
								   label="BKG", **hist_params )
		values, bins, _ = plt.hist(data.ix[data[target].values == 1, feature].values, weights= data.ix[data[target].values == 1, weights].values ,
								   range=(min_value, max_value), label='Signal', **hist_params)
		areaSig = sum(np.diff(bins)*values)
		#print areaBKG, " ",areaBKG2 ," ",areaSig
		if n == 0 : plt.legend(loc='best')
		plt.title(feature)
	plt.savefig(channel+"/"+process+'_'+bdtType+"_"+trainvar+"_Variables_BDT.pdf")
	plt.savefig(channel+"/"+process+'_'+bdtType+"_"+trainvar+"_Variables_BDT.png")
	plt.clf()

	print (len(trainVars(False)))
	for ii in [1,2] :
		if ii == 1 :
			datad=data.loc[data[target].values == 1]
			label="signal"
		else :
			datad=data.loc[data[target].values == 0]
			label="BKG"
		datacorr = datad[trainVars(False)]
		correlations = datacorr.corr()
		fig = plt.figure(figsize=(10, 10))
		ax = fig.add_subplot(111)
		cax = ax.matshow(correlations, vmin=-1, vmax=1)
		ticks = np.arange(0,len(trainVars(False)),1)
		plt.rc('axes', labelsize=8)
		ax.set_xticks(ticks)
		ax.set_yticks(ticks)
		ax.set_xticklabels(trainVars(False),rotation=-90)
		ax.set_yticklabels(trainVars(False))
		fig.colorbar(cax)
		fig.tight_layout()
		plt.savefig("{}/{}_{}_{}_corr_{}.png".format(channel,process,bdtType,str(len(trainVars(False))),label))
		plt.savefig("{}/{}_{}_{}_corr_{}.pdf".format(channel,process,bdtType,str(len(trainVars(False))),label))
		ax.clear()
#########################################################################################
traindataset, valdataset  = train_test_split(data[trainVars(withKinFit)+[target,"totalWeight"]], test_size=0.3, random_state=7)
print (traindataset.columns.values.tolist())
## Training parameters
hypOptOnsite=False ### this takes too much time to be done in series, run True only if you know what you are doing
if hypOptOnsite==True :
    print ("HypOptOnSite ", time.asctime( time.localtime(time.time()) ))
    param_grid = {
    			'n_estimators': [1000,1500,2000],
    			'min_child_weight': [3,10,15],
    			'max_depth': [1,2,3,4,5,6],
    			'learning_rate': [0.01,0.02,0.03,0.4]
    			}

    early_stopping_rounds = None
    cv=3
    cls = xgb.XGBClassifier()
    fit_params = { "eval_set" : [(traindataset[trainVars(withKinFit)].values,traindataset[target])],
                           "eval_metric" : "roc_auc",
                           "early_stopping_rounds" : early_stopping_rounds }
    gs = GridSearchCV(cls, param_grid, fit_params, cv = cv, verbose = 1)
    gs.fit(traindataset[trainVars(withKinFit)].values,traindataset[target])
    for i, param in enumerate(gs.cv_results_["params"]):
    	print("params : {} \n    cv auc = {}  +- {} ".format(param,gs.cv_results_["mean_test_score"][i],gs.cv_results_["std_test_score"][i]))
    print(gs.best_params_)
    print(gs.best_score_)
    gs = dm.grid_search_cv(clf, param_grid = param_grid,early_stopping_rounds = None)
    file = open(channel+"HTT_result"+str(len(trainVars(withKinFit)))+"var_WithKinFit"+str(WithKinfit)+".dat","w")
    file.write(gs.best_params_)
    file.write(gs.best_score_)
    file.close()
    print "wrote "+channel+"/HTT_result"+str(len(trainVars(withKinFit)))+"var_WithKinFit"+str(WithKinfit)+".dat"
##########################################################################################
print ("Finally training: ", time.asctime( time.localtime(time.time()) ))
cls = xgb.XGBClassifier(n_estimators = options.ntrees, max_depth = options.treeDeph, min_child_weight = 1, learning_rate = options.lr)
cls.fit(
	traindataset[trainVars(withKinFit)].values,
	traindataset[target].astype(np.bool),
	sample_weight= (traindataset[weights].astype(np.float64)) #,
	# this is another diagnosis trick
	#eval_set=[(traindataset[trainVars(False)].values,  traindataset[target].astype(np.bool),traindataset[weights].astype(np.float64)),
	#(valdataset[trainVars(False)].values,  valdataset[target].astype(np.bool), valdataset[weights].astype(np.float64))] ,
	#verbose=True ,eval_metric="auc"
	)
print ("XGBoost with KinFit trained",withKinFit, time.asctime( time.localtime(time.time()) ))
proba = cls.predict_proba(traindataset[trainVars(withKinFit)].values  )
evttest=2
print ("One event",traindataset[trainVars(withKinFit)].iloc[evttest])
print ("predict BDT to one event",withKinFit,cls.predict_proba(traindataset[trainVars(withKinFit)].iloc[evttest])[:,1])
proba = cls.predict_proba(traindataset[trainVars(withKinFit)].values)
fpr, tpr, thresholds = roc_curve(traindataset[target], proba[:,1] )
train_auc = auc(fpr, tpr, reorder = True)
print("XGBoost train set auc - {}".format(train_auc))
proba = cls.predict_proba(valdataset[trainVars(withKinFit)].values)
fprt, tprt, thresholds = roc_curve(valdataset[target], proba[:,1] )
test_auct = auc(fprt, tprt, reorder = True)
print("XGBoost test set auc - {}".format(test_auct))
################################################################################
if doXML==True :
	bdtpath=channel+"/"+process+"_"+channel+"_XGB_"+trainvar+"_"+bdtType+"_nvar"+str(len(trainVars(withKinFit)))
	print ("Output pkl ", time.asctime( time.localtime(time.time()) ))
	if withKinFit :
		pickle.dump(cls, open(bdtpath+"_withKinFit.pkl", 'wb'))
		print ("saved "+bdtpath+"_withKinFit.pkl")
	else :
		pickle.dump(cls, open(bdtpath+".pkl", 'wb'))
		print ("saved "+bdtpath+".pkl")
	print ("starting xml conversion")
###########################################################################
if options.evaluateFOM==True :
	print ("evaluateFOM with kinfit",withKinFit, time.asctime( time.localtime(time.time()) ))
	## feature importance plot
	fig, ax = plt.subplots()
	f_score_dicts =cls.booster().get_fscore()
	f_score_dicts = {trainVars(withKinFit)[int(k[1:])] : v for k,v in f_score_dicts.items()}
	if options.HypOpt==False :
		feat_imp = pandas.Series(f_score_dicts).sort_values(ascending=True)
		feat_imp.plot(kind='barh', title='Feature Importances')
		fig.tight_layout()
		if withKinFit :
			fig.savefig("{}/{}_{}_{}_XGB_importance_withKinFit.png".format(channel,process,bdtType,trainvar))
			fig.savefig("{}/{}_{}_{}_XGB_importance_withKinFit.pdf".format(channel,process,bdtType,trainvar))
		else :
			fig.savefig("{}/{}_{}_{}_XGB_importance.png".format(channel,process,bdtType,trainvar))
			fig.savefig("{}/{}_{}_{}_XGB_importance.pdf".format(channel,process,bdtType,trainvar))
	# the bellow takes time: you may want to comment if you are setting up
	evaluateFOM(cls,keys,trainVars(withKinFit),"WithKinfit"+str(withKinFit), train_auc , test_auct,nBdeplet,nB,nS,f_score_dicts)
########################################################################
if options.HypOpt==False :
	print ("Start plotting repport: ", time.asctime( time.localtime(time.time()) ))
	fig, ax = plt.subplots()
	## ROC curve
	ax.plot(fpr, tpr, lw=1, label='XGB train + angles (area = %0.3f)'%(train_auc))
	ax.plot(fprt, tprt, lw=1, label='XGB test + angles (area = %0.3f)'%(test_auct))
	ax.set_ylim([0.0,1.0])
	ax.set_xlabel('False Positive Rate')
	ax.set_ylabel('True Positive Rate')
	ax.legend(loc="lower right")
	ax.grid()
	fig.savefig("{}/{}_{}_{}_roc.png".format(channel,process,bdtType,trainvar))
	fig.savefig("{}/{}_{}_{}_roc.pdf".format(channel,process,bdtType,trainvar))
	###########################################################################
