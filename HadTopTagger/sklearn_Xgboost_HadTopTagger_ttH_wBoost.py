import sys , time
#run: python sklearn_Xgboost_HadTopTagger_ttH.py --process 'all' --evaluateFOM --HypOpt --doXML &
import os
import sklearn
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier

import pandas
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import numpy as np

import pickle

from sklearn.externals import joblib
import root_numpy
from root_numpy import root2array, rec2array, array2root, tree2array

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import ROOT
#from tqdm import trange
import glob
import xgboost as xgb
execfile("../python/data_manager.py")

from optparse import OptionParser
parser = OptionParser()
parser.add_option("--process ", type="string", dest="process", help="process", default='ttH')
parser.add_option("--evaluateFOM", action="store_true", dest="evaluateFOM", help="evaluateFOM", default=False)
parser.add_option("--HypOpt", action="store_true", dest="HypOpt", help="If you call this will not do plots with repport", default=False)
parser.add_option("--withKinFit", action="store_true", dest="withKinFit", help="BDT variables with kinfit", default=False)
parser.add_option("--doXML", action="store_true", dest="doXML", help="BDT variables with kinfit", default=False)
parser.add_option("--ntrees ", type="int", dest="ntrees", help="hyp", default=1000)
parser.add_option("--treeDeph", type="int", dest="treeDeph", help="hyp", default=1)
parser.add_option("--lr", type="float", dest="lr", help="hyp", default=0.01)
parser.add_option("--mcw", type="float", dest="mcw", help="hyp", default=10)
(options, args) = parser.parse_args()

channel="HadTopTagger_wBoost" #options.channel #"1l_2tau"
inputPath='structured/'
process=options.process
keys=['ttHToNonbb','TTToSemilepton','TTZToLLNuNu','TTWJetsToLNu']
bdtType="CSV_sort"
withKinFit=options.withKinFit
doXML=options.doXML
trainvar="ntrees_"+str(options.ntrees)+"_deph_"+str(options.treeDeph)+"_lr_0o0"+str(int(options.lr*100)) #options.variables #

inputTree="TCVARSbfilter"
target="bWj1Wj2_isGenMatched"

oldPKL = "HadTopTagger_sklearnV0o17o1_HypOpt/all_HadTopTagger_sklearnV0o17o1_HypOpt_XGB_ntrees_1000_deph_3_lr_0o01_CSV_sort_withKinFit_withKinFit.pkl"
from sklearn.externals import joblib
oldclf = joblib.load(oldPKL)


import shutil,subprocess
proc=subprocess.Popen(['mkdir '+channel],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()

kinvars = [
        #'Dphi_KinWb_lab',
        #'Dphi_KinWb_rest',
        #'Dphi_KinWj1_KinWj2_lab',
        ##'Dphi_Wb_lab',
        #'Dphi_Wb_rest',
        ##'Dphi_Wj1_Wj2_lab',
        #'cosThetaKinW_lab',
        #'cosThetaKinW_rest',
        #'cosThetaKinWj_restW',
        #'cosThetaKinb_lab',
        #'cosThetaKinb_rest',
        #'cosThetaW_lab',
        #'cosThetaW_rest',
        'cosThetaWj1_restW',
        #'cosTheta_Kin_leadEWj_restTop',
        #'cosTheta_Kin_leadWj_restTop',
        #'cosTheta_Kin_subleadEWj_restTop',
        #'cosTheta_Kin_subleadWj_restTop',
        ##'cosTheta_leadEWj_restTop',
        ##'cosTheta_leadWj_restTop',
        ##'cosTheta_subleadEWj_restTop',
        ##'cosTheta_subleadWj_restTop',
        ##'cosThetab_lab',
        #'cosThetab_rest',
        'dR_Wj1Wj2',
        'dR_bW',
        'dR_bWj1',
        'dR_bWj2',
        #'eta_Wj1',
        #'eta_Wj2',
        #'eta_b',
        'm_Wj1Wj2',
        'm_Wj1Wj2_div_m_bWj1Wj2',
        'm_bWj1Wj2',
        'm_bWj1',
        'm_bWj2',
        'pT_Wj1',
        'pT_Wj1Wj2',
        'pT_Wj2',
        'pT_b',
        'pT_bWj1Wj2',
        ##'alphaKinFit',
        'nllKinFit',
        'kinFit_pT_Wj1',
        'kinFit_pT_Wj2',
        'kinFit_pT_b',
        'pT_b_o_kinFit_pT_b'
]

def trainVars(cat,train):
	if cat==1 and train==False :
		return [
        #'genTopPt',
        #'bjet_tag_position',
        'massTop',
        'tau32Top',
        #"btagDisc",
        "drT_gen", "drWj1_gen", "drWj2_gen", "drB_gen", "drW_gen",
        "etaWj1_gen", "etaWj2_gen", "etaB_gen",
        "ptWj1_gen", "ptWj2_gen", "ptB_gen",
        "dr_b_wj1", "dr_b_wj2", "dr_wj1_wj2",
        "genFatPtAll", "genFatEtaAll"
        #'collectionSize',
        #"fatjet_isGenMatched"
		] #+ kinvars

	if cat==2 and train==False :
		return [
        'genTopPt',
        #'bWj1Wj2_isGenMatched',
        'bjet_tag_position',
        #'massTop',
        'massW_SD',
        'tau21W',
        #'tau32Top',
        "btagDisc",
        #"qg_Wj1",
        #"qg_Wj2",
        'collectionSize',
        "fatjet_isGenMatched",
        "drT_gen", "drWj1_gen", "drWj2_gen", "drB_gen", "drW_gen",
        "etaWj1_gen", "etaWj2_gen", "etaB_gen",
        "ptWj1_gen", "ptWj2_gen", "ptB_gen",
        "dr_b_wj1", "dr_b_wj2", "dr_wj1_wj2",
        "dr_b_wj1_gen", "dr_b_wj2_gen", "dr_wj1_wj2_gen",
        #'typeTop'
		] #+ kinvars

	if cat==3 and train==False :
		return [
        'genTopPt',
        #'bWj1Wj2_isGenMatched',
        'bjet_tag_position',
        #'massTop',
        #'massW_SD',
        #'tau21W',
        #'tau32Top',
        "btagDisc",
        "qg_Wj1",
        "qg_Wj2",
        'collectionSize',
        #"fatjet_isGenMatched",
        #'typeTop'
		] #+ kinvars

	if cat==1 and train==True :
		return [
        'massTop',
        'tau32Top',
        #"btagDisc",
        #'bjet_tag_position',
		] + kinvars

	if cat==2 and train==True :
		return [
        #'genTopPt',
        #'bWj1Wj2_isGenMatched',
        #'bjet_tag_position',
        #'massTop',
        'massW_SD',
        'tau21W',
        #'tau32Top',
        #"btagDisc",
        #"qg_Wj1",
        #"qg_Wj2",
        #'collectionSize',
        #"fatjet_isGenMatched",
        #'typeTop'
		] #+ kinvars

	if cat==3 and train==True :
		return [
        #'genTopPt',
        #'bWj1Wj2_isGenMatched',
        'bjet_tag_position',
        #'massTop',
        #'massW_SD',
        #'tau21W',
        #'tau32Top',
        "btagDisc",
        "qg_Wj1",
        "qg_Wj2",
        #'collectionSize',
        #"fatjet_isGenMatched",
        #'typeTop',
        'cosThetaWj1_restW',
        #'cosTheta_Kin_leadEWj_restTop',
        #'cosTheta_Kin_leadWj_restTop',
        #'cosTheta_Kin_subleadEWj_restTop',
        #'cosTheta_Kin_subleadWj_restTop',
        ##'cosTheta_leadEWj_restTop',
        ##'cosTheta_leadWj_restTop',
        ##'cosTheta_subleadEWj_restTop',
        ##'cosTheta_subleadWj_restTop',
        ##'cosThetab_lab',
        #'cosThetab_rest',
        ##'dR_Wj1Wj2',
        ##'dR_bW',
        ##'dR_bWj1',
        ##'dR_bWj2',
        #'eta_Wj1',
        #'eta_Wj2',
        #'eta_b',
        'm_Wj1Wj2',
        ##'m_Wj1Wj2_div_m_bWj1Wj2',
        ##'m_bWj1Wj2',
        ##'m_bWj1',
        ##'m_bWj2',
        'pT_Wj1',
        ##'pT_Wj1Wj2',
        'pT_Wj2',
        ##'pT_b',
        'pT_bWj1Wj2',
        ##'alphaKinFit',
        'nllKinFit',
        ##'kinFit_pT_Wj1',
        ##'kinFit_pT_Wj2',
        ##'kinFit_pT_b',
        ##'pT_b_o_kinFit_pT_b'
		]

category = 1
jet = 12
btagRank = 4
keystoDraw=['ttHToNonbb','TTToSemilepton','TTWJetsToLNu']
treetoread="analyze_hadTopTagger/evtntuple/signal/evtTree"
sourceA="/hdfs/local/acaan/HTT_withBoost/ttHJetToNonbb_M125_amcatnlo_ak"+str(jet)+"_noCleaning_HTTv2loop_fatter_RJet_higestBtagHTTv2.root"
# ttHJetToNonbb_M125_amcatnlo_ak12_noCleaning_HTTv2loop_fatter_R0o1
data = pandas.DataFrame(columns=trainVars(category,False)+['key','weights'], index=['bWj1Wj2_isGenMatched',"fatjet_isGenMatched"])

tfile = ROOT.TFile(sourceA)
tree = tfile.Get(treetoread)
chunk_arr = tree2array(tree,
    selection='typeTop == {} && bjet_tag_position <= {}'.format(category,btagRank)) # && collectionSize == 1
chunk_df = pandas.DataFrame(chunk_arr)
chunk_df['key'] = keystoDraw[0]
chunk_df['weights'] = 1.0
chunk_df['CSV_b'] = chunk_df["btagDisc"]
chunk_df['pT_b_o_kinFit_pT_b'] =  chunk_df['pT_b']/chunk_df['kinFit_pT_b']
chunk_df['cosThetaWj1_restW'] = abs(chunk_df['cosThetaWj1_restW'])
chunk_df['bjet_tag_position']= np.where(chunk_df['bjet_tag_position'] > 3, 4, chunk_df['bjet_tag_position'])
chunk_df['target'] = 0
#chunk_df['target'] = np.where((chunk_df["drT_genTriplet"] < 1.5) & (chunk_df["drB_gen"] < 0.5) , 1, chunk_df['target'])
chunk_df['target'] = np.where((chunk_df["drWj1_gen"] < 0.3) & (chunk_df["drWj2_gen"] < 0.3) &  (chunk_df["drB_gen"] < 0.3) , 1, chunk_df['target'])
# & (chunk_df["drWj1_gen"] < 0.75) & (chunk_df["drWj2_gen"] < 0.75) & (chunk_df["bjet_tag_position"] == 1)
data=data.append(chunk_df, ignore_index=True)
data.dropna(subset=['bWj1Wj2_isGenMatched',"fatjet_isGenMatched","counter"],inplace = True) # data
print list(data)
print len(data)
target = 'target' #"bWj1Wj2_isGenMatched" #
weights = "weights"

doOld = False
if doOld :
    oldVars = [ # ['CSV_b', 'qg_Wj2', 'pT_bWj1Wj2', 'm_Wj1Wj2', 'nllKinFit', 'pT_b_o_kinFit_pT_b', 'pT_Wj2']
            'CSV_b',
            'qg_Wj2',
            'pT_bWj1Wj2',
            'pT_Wj2',
            'm_Wj1Wj2',
            'nllKinFit',
            'pT_b_o_kinFit_pT_b'#,
    ]
    evaluateFOM(oldclf,keys[0], oldVars ,"oldWithKinfit"+str(True), "train_auc" , "test_auct", 1, 1, 1, "f_score_dict", data)

print ("len sig/BKG",len(data.loc[data[target]==1]),len(data.loc[data[target]==0]))
print ("len sig (gen pt > 200)",len(data.loc[(data[target]==1) & (data["genFatPtAll"] > 200 )]))

#print  len(np.unique(data["counter"].values))
#print len(data["counter"].values)

print ("len sig/BKG (fattag)",
    len(data.loc[(data["fatjet_isGenMatched"]==1) ]), # & (data["b_isGenMatched"]==1)
    len(data.loc[(data["fatjet_isGenMatched"]==0) ]) # | (data["b_isGenMatched"]==0)
    )

#print ("len sig/BKG (target)",
#    len(data.loc[(data["target"]==1) ]), # & (data["b_isGenMatched"]==1)
#    len(data.loc[(data["target"]==0) ]) # | (data["b_isGenMatched"]==0)
#    )

print ("sum weights sig/BKG",data.loc[data[target]==1]['weights'].sum(),data.loc[data[target]==0]['weights'].sum())

#df_y_count = data.groupby(labels).size().reset_index().rename(columns={0:'bWj1Wj2_isGenMatched'})
#print data.index.get_values()
#print data[['bWj1Wj2_isGenMatched',"fatjet_isGenMatched"]]

## Balance datasets
data.loc[data[target]==0, ['weights']] *= 100000/data.loc[data[target]==0]['weights'].sum()
data.loc[data[target]==1, ['weights']] *= 100000/data.loc[data[target]==1]['weights'].sum()

## make plots
nbins=8
color1='g'
color2='b'
printmin=True
plotResiduals=False

make_plots(
    trainVars(category,True), 20,
    data.loc[data[target]==1], "signal", color1,
    data.loc[data[target]==0], "BKG", color2,
    channel+"/HTT_withBoost_cat"+str(category)+"_ak"+str(jet)+".pdf",
    printmin,
    plotResiduals
    )

make_plots(
    [
    #"dr_wj1_wj2_gen", #
    # "dr_b_wj1_gen", "dr_b_wj2_gen",
    "drWj1_gen", "drWj2_gen", "drB_gen",
    "drW_gen", "drT_gen", "drT_genTriplet", "drT_genJ_max",
    # "etaWj1_gen", "etaWj2_gen", "etaB_gen",
    # "ptWj1_gen", "ptWj2_gen", "ptB_gen",
    #"genFatPtAll", "genFatEtaAll", #"drB_gen",
    ], 20,
    data.loc[data[target]==1], "signal", color1,
    data.loc[data[target]==0], "BKG", color2,
    channel+"/HTT_withBoost_cat"+str(category)+"_ak"+str(jet)+"_genvars.pdf",
    printmin,
    plotResiduals
    )

traindataset, valdataset  = train_test_split(data[trainVars(category,True)+[target,'weights',"fatjet_isGenMatched", "counter"]], test_size=0.3, random_state=7)

cls = xgb.XGBClassifier(
			n_estimators = options.ntrees,
			max_depth = options.treeDeph,
			min_child_weight = options.mcw, # min_samples_leaf
			learning_rate = options.lr,
            #objective="multi:softmax"
			)
cls.fit(
	traindataset[trainVars(category,True)].values,
	traindataset[target].astype(np.bool),
	sample_weight= (traindataset[weights].astype(np.float64)) #,
	)

print trainVars(category,True)
print traindataset[trainVars(category,True)].columns.values.tolist()
print ("XGBoost trained")
proba = cls.predict_proba(traindataset[trainVars(category,True)].values )
print proba
fpr, tpr, thresholds = roc_curve(traindataset[target], proba[:,1],
	sample_weight=(traindataset[weights].astype(np.float64)) )
train_auc = auc(fpr, tpr, reorder = True)
print("XGBoost train set auc - {}".format(train_auc))
proba = cls.predict_proba(valdataset[trainVars(category,True)].values )
fprt, tprt, thresholds = roc_curve(valdataset[target], proba[:,1], sample_weight=(valdataset[weights].astype(np.float64))  )
test_auct = auc(fprt, tprt, reorder = True)
print("XGBoost test set auc - {}".format(test_auct))
################################################################################
if doXML==True :
	bdtpath=channel+"/"+process+"_"+channel+"_XGB_"+trainvar+"_"+bdtType+"_nvar"+str(len(trainVars(category,True)))
	print ("Output pkl ", time.asctime( time.localtime(time.time()) ))
	if withKinFit :
		pickle.dump(cls, open(bdtpath+"_withKinFit.pkl", 'wb'))
		print ("saved "+bdtpath+"_withKinFit.pkl")
	else :
		pickle.dump(cls, open(bdtpath+".pkl", 'wb'))
		print ("saved "+bdtpath+".pkl")
	print ("starting xml conversion")
###########################################################################

###########################################################################
## feature importance plot
fig, ax = plt.subplots()
f_score_dict =cls.booster().get_fscore()
f_score_dict = {trainVars(category,True)[int(k[1:])] : v for k,v in f_score_dict.items()}
feat_imp = pandas.Series(f_score_dict).sort_values(ascending=True)
feat_imp.plot(kind='barh', title='Feature Importances')
fig.tight_layout()
fig.savefig("{}/cat_{}_nvar_{}_ak{}_XGB_importance.pdf".format(channel,str(category),str(len(trainVars(category,True))),str(jet)))
###########################################################################
# the bellow takes time: you may want to comment if you are setting up
if options.evaluateFOM==True :
    evaluateFOM(cls,keys[0],trainVars(category,True),"WithKinfit"+str(True), train_auc , test_auct, 1, 1, 1, f_score_dict, valdataset)
##########################################################################
# plot correlation matrix
for ii in [1,2] :
	if ii == 1 :
		datad=data.loc[data[target].values == 1]
		label="signal"
	else :
		datad=data.loc[data[target].values == 0]
		label="BKG"
	datacorr = datad[trainVars(category,True)] #.loc[:,trainVars(False)] #dataHToNobbCSV[[trainVars(True)]]
	correlations = datacorr.corr()
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111)
	cax = ax.matshow(correlations, vmin=-1, vmax=1)
	ticks = np.arange(0,len(trainVars(category,True)),1)
	plt.rc('axes', labelsize=8)
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)
	ax.set_xticklabels(trainVars(category,True),rotation=-90)
	ax.set_yticklabels(trainVars(category,True))
	fig.colorbar(cax)
	fig.tight_layout()
	#plt.subplots_adjust(left=0.9, right=0.9, top=0.9, bottom=0.1)
	fig.savefig("{}/cat_{}_nvar_{}_ak{}_{}_corr.pdf".format(channel,str(category),str(len(trainVars(category,True))),str(jet),label))
	ax.clear()
###################################################################
