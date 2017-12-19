# python map_2D_evtLevel_ttH.py --channel '2lss_1tau' --variables "oldVar" --nbins-start 5 --nbins-target 5
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--channel ", type="string", dest="channel", help="The ones whose variables implemented now are:\n   - 1l_2tau\n   - 2lss_1tau\n It will create a local folder and store the report*/xml", default='T')
parser.add_option("--variables", type="string", dest="variables", help="  Set of variables to use -- it shall be put by hand in the code", default=1000)
parser.add_option("--nbins-start ", type="int", dest="start", help="for the squared 2D histogram", default=20)
parser.add_option("--nbins-target", type="int", dest="target", help="hyp", default=5)
#parser.add_option("--lr", type="float", dest="lr", help="hyp", default=0.01)
#parser.add_option("--mcw", type="int", dest="mcw", help="hyp", default=1)
(options, args) = parser.parse_args()

#channel="2lss_1tau"
channel=options.channel #"1l_2tau"
if channel=='1l_2tau':
	channelInTree='1l_2tau_OS_Tight'
	inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Dec-BDT-noMEM-LooseLepMedTau-TagT-fakeR/histograms/1l_2tau/forBDTtraining_OS/'#  - tight lepton, medium tau

if channel=='2lss_1tau':
	#channelInTree='2lss_1tau_lepSS_sumOS_Loose'
	#inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Dec13-BDT-noMEM-LooseLepMedTau-TagT-fakeR/histograms/2lss_1tau/forBDTtraining_SS_OS/' # with charge tag, 2017Dec13-BDT-noMEM-LooseLepMedTau-TagT-fakeR
	channelInTree='2lss_1tau_lepSS_sumOS_Tight'
	inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Dec08-BDT-noMEM-tighLep/histograms/2lss_1tau/forBDTtraining_SS_OS/' #  2017Dec08-BDT-noMEM-tighLep
	#channelInTree='2lss_1tau_lepSS_sumOS_Fakeable_wFakeRateWeights'
	#inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Dec08-BDT-noMEM-fakableLepLooseTau/histograms/2lss_1tau/forBDTtraining_SS_OS/' # 2017Dec08-BDT-noMEM-fakableLepMedTau

import sys , time
#import sklearn_to_tmva
import sklearn
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
import pandas
import ROOT
#from pandas import HDFStore,DataFrame
import sklearn_to_tmva
import xgboost2tmva
import skTMVA
import matplotlib
matplotlib.use('agg')
#matplotlib.use('PS')   # generate postscript output by default
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from matplotlib import colors as colors
import math , array
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
ROOT.gStyle.SetOptStat(0)
from tqdm import trange
import glob

from keras.models import Sequential, model_from_json
import json

from collections import OrderedDict

def reverse_colourmap(cmap, name = 'my_cmap_r'):
    reverse = []
    k = []
    for key in cmap._segmentdata:
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []
        for t in channel:
            data.append((1-t[0],t[2],t[1]))
        reverse.append(sorted(data))
    LinearL = dict(zip(k,reverse))
    my_cmap_r = colors.LinearSegmentedColormap(name, LinearL)
    return my_cmap_r





import shutil,subprocess
proc=subprocess.Popen(['mkdir '+options.channel],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()

Variables_all=[
'HadTop_eta', 'HadTop_pt',
'MT_met_lep1', 'avg_dr_jet', 'avg_dr_lep', 'dr_lep1_HTfitted', 'dr_lep1_HTunfitted', 'dr_lep1_tau',
'dr_lep2_HTfitted', 'dr_lep2_HTunfitted', 'dr_lep2_tau', 'dr_leps',
'dr_tau_HTfitted', 'dr_tau_HTunfitted', 'evtWeight', 'fitHTptoHTmass', 'fitHTptoHTpt',
'fittedHadTop_eta', 'fittedHadTop_pt', 'hadtruth', 'htmiss', 'lep1_conePt', 'lep1_eta',
'lep1_frWeight', 'lep1_genLepPt', 'lep1_pt', 'lep1_tth_mva', 'lep2_conePt', 'lep2_eta',
'lep2_frWeight', 'lep2_genLepPt', 'lep2_pt', 'lep2_tth_mva', 'mT_lep1',
'mT_lep2', 'mTauTauVis1', 'mTauTauVis2', 'mass_lep1_HTfitted',
'mass_lep2_HTfitted', 'max_lep_eta', 'mbb',
 'min(met_pt,400)', 'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet',
'mvaOutput_hadTopTaggerWithKinFit', 'nJet25_Recl',
'ptmiss', 'tau_eta',
'tau_mva', 'tau_pt', 'unfittedHadTop_pt',
 'lep1_isTight', 'lep2_isTight', 'nBJetLoose', 'nJet', 'nLep',  'tau_isTight'
]


def trainVarsTT(trainvar):

        if trainvar=="noHTT" and channel=="2lss_1tau" :
			return [
		'avg_dr_jet', 'dr_lep1_tau', 'dr_lep2_tau',
		'dr_leps',
		'lep1_conePt', 'lep2_conePt',
		'mT_lep1',
		'mT_lep2', 'mTauTauVis1', 'mTauTauVis2', 'max_lep_eta',
		'mbb',
		'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet',
		'nJet25_Recl',
		'ptmiss', 'tau_pt',
		]

        if trainvar=="HTT" and channel=="2lss_1tau" :
			return [
			'dr_lep1_tau', 'dr_lep2_tau', 'dr_leps',
			'mT_lep1', 'mT_lep2', 'mTauTauVis1', 'mTauTauVis2', 'mindr_lep1_jet',
			'mindr_lep2_jet', 'mindr_tau_jet', 'ptmiss', 'tau_pt',
			'mvaOutput_hadTopTaggerWithKinFit', 'mvaOutput_Hj_tagger', 'unfittedHadTop_pt',
			'nJet25_Recl', 'avg_dr_jet'
			]


        if trainvar=="HTT_LepID" and channel=="2lss_1tau" :
			return [
		'mTauTauVis1', 'mTauTauVis2',
		'tau_pt',
		'mvaOutput_hadTopTaggerWithKinFit',
		'lep1_tth_mva',
		'lep2_tth_mva'
		]

        if trainvar=="oldVar"  and channel=="2lss_1tau" :return [
		"max_lep_eta",
		"nJet25_Recl",
		"mindr_lep1_jet",
		"mindr_lep2_jet",
		"min(met_pt,400)",
		"avg_dr_jet",
		"MT_met_lep1"
		]


def trainVarsTTV(trainvar):

        if trainvar=="oldVar"  and channel=="2lss_1tau" :return [
		"max_lep_eta",
		"MT_met_lep1",
		"nJet25_Recl",
		"mindr_lep1_jet",
		"mindr_lep2_jet",
		"lep1_conePt",
		"lep2_conePt"
		]

        if trainvar=="HTT_LepID" and channel=="2lss_1tau"  :
			return [
		'avg_dr_jet', 'dr_lep1_tau', 'dr_lep2_tau', 'dr_leps',
		'lep1_conePt', 'lep2_conePt', 'mT_lep1', 'mT_lep2',
		'mTauTauVis1', 'mTauTauVis2', 'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet',
		'ptmiss', 'tau_pt',
		'mvaOutput_hadTopTaggerWithKinFit', 'mvaOutput_Hj_tagger'
		]

        if trainvar=="HTT" and channel=="2lss_1tau" :
			return [
		'avg_dr_jet', 'dr_lep1_tau', 'dr_lep2_tau', 'dr_leps',
		'lep1_conePt', 'lep2_conePt', 'mT_lep1', 'mT_lep2',
		'mTauTauVis1', 'mTauTauVis2', 'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet',
		'ptmiss', 'tau_pt',
		'mvaOutput_hadTopTaggerWithKinFit', 'mvaOutput_Hj_tagger'
		]

        if trainvar=="noHTT" and channel=="2lss_1tau":
			return [
		'avg_dr_jet', 'dr_lep1_tau', 'dr_lep2_tau', 'dr_leps',
		'lep1_conePt', 'lep2_conePt',
		'mT_lep1', 'mT_lep2', 'mTauTauVis1', 'mTauTauVis2',
		'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet', 'ptmiss', 'tau_pt'
		]

####################################################################################################
## Load data
my_cols_list=Variables_all+['key','target','file']
testtruth="bWj1Wj2_isGenMatchedWithKinFit"
weights="totalWeight"
data = pandas.DataFrame(columns=my_cols_list)
keys=['ttHToNonbb','TTTo2L2Nu','TTToSemilepton','TTZToLLNuNu','TTWJetsToLNu']
for folderName in keys :
	print (folderName)
	if 'TTT' in folderName :
		sampleName='TT'
		target=0
	if folderName=='ttHToNonbb' :
		sampleName='signal'
		target=1
	if 'TTW' in folderName :
		sampleName='TTW'
		target=0
	if 'TTZ' in folderName :
		sampleName='TTZ'
		target=0
	inputTree = channelInTree+'/sel/evtntuple/'+sampleName+'/evtTree'
	if ('TTT' in folderName) or folderName=='ttHToNonbb' :
		procP1=glob.glob(inputPath+"/"+folderName+"_fastsim_p1/"+folderName+"_fastsim_p1_forBDTtraining*OS_central_*.root")
		procP2=glob.glob(inputPath+"/"+folderName+"_fastsim_p2/"+folderName+"_fastsim_p2_forBDTtraining*OS_central_*.root")
		procP3=glob.glob(inputPath+"/"+folderName+"_fastsim_p3/"+folderName+"_fastsim_p3_forBDTtraining*OS_central_*.root")
		list=procP1+procP2+procP3
	else :
		procP1=glob.glob(inputPath+"/"+folderName+"_fastsim/"+folderName+"_fastsim_forBDTtraining*OS_central_*.root")
		list=procP1
	print ("Date: ", time.asctime( time.localtime(time.time()) ))
	for ii in range(0, len(list)) : #
		#print (list[ii],inputTree)
		try: tfile = ROOT.TFile(list[ii])
		except :
			#print "Doesn't exist"
			#print ('file ', list[ii],' corrupt')
			continue
		try: tree = tfile.Get(inputTree)
		except :
			#print "Doesn't exist"
			#print ('file ', list[ii],' corrupt')
			continue
		if tree is not None :
			try:
				chunk_arr = tree2array(tree) #,  start=start, stop = stop)
			except :
				#print "Doesn't exist"
				#print ('file ', list[ii],' corrupt')
				continue
			else :
				chunk_df = pandas.DataFrame(chunk_arr) #
				if ii ==0 : print (chunk_df.columns.values.tolist())
				chunk_df['key']=folderName
				chunk_df['target']=target
				#chunk_df['file']=list[ii].split("_")[10]
				if channel=="2lss_1tau" :
					chunk_df[weights] = chunk_df["evtWeight"]*chunk_df['tau_frWeight']*chunk_df['lep1_frWeight']*chunk_df['lep2_frWeight']
				if channel=="1l_2tau" : chunk_df[weights] = chunk_df.evtWeight
				###########
				#if channel=="2lss_1tau" :
				#	data=data.append(chunk_df.ix[(chunk_df.lep1_isTight.values == 1) &
				#						(chunk_df.lep2_isTight.values == 1) &
				#						(chunk_df.tau_isTight.values == 1) &
				#						(chunk_df.failsTightChargeCut.values == 0 )], ignore_index=True)
				#else : #
				if 1>0 :
					data=data.append(chunk_df, ignore_index=True)
		else : print ("file "+list[ii]+"was empty")
		tfile.Close()
	if len(data) == 0 : continue
	nS = len(data.ix[(data.target.values == 0) & (data.key.values==folderName)])
	nB = len(data.ix[(data.target.values == 1) & (data.key.values==folderName)])
	print "length of sig, bkg: ", nS, nB
	if channel=="1l_2tau" or channel=="2lss_1tau" :
		nSthuthKin = len(data.ix[(data.target.values == 0) & (data[testtruth].values==1) & (data.key.values==folderName)])
		nBtruthKin = len(data.ix[(data.target.values == 1) & (data[testtruth].values==1) & (data.key.values==folderName)])
		nShadthuth = len(data.ix[(data.target.values == 0) & (data.hadtruth.values==1) & (data.key.values==folderName)])
		nBhadtruth = len(data.ix[(data.target.values == 1) & (data.hadtruth.values==1) & (data.key.values==folderName)])
		print "truth Kin:          ", nSthuthKin, nBtruthKin
		print "hadtruth:           ", nShadthuth, nBhadtruth
print (data.columns.values.tolist())
n = len(data)
nS = len(data.ix[data.target.values == 0])
nB = len(data.ix[data.target.values == 1])
print "length of sig, bkg: ", nS, nB
#################################################################################
# drop events with NaN weights - for safety
data.dropna(subset=["totalWeight"],inplace = True) # data
data.fillna(0)
nS = len(data.loc[data.target.values == 0])
nB = len(data.loc[data.target.values == 1])
print "length of sig, bkg without NaN: ", nS, nB
#########################################################################################
## Load the BDTs - do 2D plot

BDTvar=["oldVar","HTT_LepID","HTT","noHTT"]
nbins= options.start # 15
nbinsout= options.target #5
BDTvar=options.variables #"oldVar"

if BDTvar=="oldVar" : ttV_file=channel+"/2lss_1tau_XGB_oldVar_evtLevelTTV_TTH_7Var.pkl"
if BDTvar=="HTT_LepID" : ttV_file=channel+"/2lss_1tau_XGB_HTT_evtLevelTTV_TTH_17Var.pkl"
if BDTvar=="HTT" : ttV_file=channel+"/2lss_1tau_XGB_HTT_evtLevelTTV_TTH_17Var.pkl"
if BDTvar=="noHTT" : ttV_file=channel+"/2lss_1tau_XGB_noHTT_evtLevelTTV_TTH_15Var.pkl"

if BDTvar=="oldVar" : tt_file=channel+"/2lss_1tau_XGB_oldVar_evtLevelTT_TTH_7Var.pkl"
if BDTvar=="HTT_LepID" : tt_file=channel+"/2lss_1tau_XGB_HTT_LepID_evtLevelTT_TTH_6Var.pkl"
if BDTvar=="HTT" : tt_file=channel+"/2lss_1tau_XGB_HTT_evtLevelTT_TTH_17Var.pkl"
if BDTvar=="noHTT" : tt_file=channel+"/2lss_1tau_XGB_noHTT_evtLevelTT_TTH_18Var.pkl"

#print (tt_files[n],BDTvar,trainVarsTT(BDTvar))
dataTT =data.ix[(data.key.values=='TTTo2L2Nu') | (data.key.values=='TTToSemilepton')]
dataTTV=data.ix[(data.key.values=='TTZToLLNuNu') | (data.key.values=='TTWJetsToLNu')]
dataTTH=data.ix[(data.key.values=='ttHToNonbb')]
clsTT=pickle.load(open(tt_file,'rb'))
####
tt_datainTT =clsTT.predict_proba(dataTT[trainVarsTT(BDTvar)].values)[:, 1]
ttV_datainTT=clsTT.predict_proba(dataTTV[trainVarsTT(BDTvar)].values)[:, 1]
ttH_datainTT=clsTT.predict_proba(dataTTH[trainVarsTT(BDTvar)].values)[:, 1]
###
testROC=clsTT.predict_proba(data.ix[
		(data.key.values=='TTTo2L2Nu') |
		(data.key.values=='TTToSemilepton') |
		(data.key.values=='ttHToNonbb')][trainVarsTT(BDTvar)].values)
fpr, tpr, thresholds = roc_curve(data.ix[(
		data.key.values=='TTTo2L2Nu') |
		(data.key.values=='TTToSemilepton') |
		(data.key.values=='ttHToNonbb')]["target"], testROC[:,1] )
print ("ROC TT",BDTvar,auc(fpr, tpr, reorder = True))
############
clsTTV=pickle.load(open(ttV_file,'rb'))
tt_datainTTV =clsTTV.predict_proba(dataTT[trainVarsTTV(BDTvar)].values)[:, 1]
ttV_datainTTV=clsTTV.predict_proba(dataTTV[trainVarsTTV(BDTvar)].values)[:, 1]
ttH_datainTTV=clsTTV.predict_proba(dataTTH[trainVarsTTV(BDTvar)].values)[:, 1]
testROCttV=clsTTV.predict_proba(
	data.ix[(data.key.values=='TTZToLLNuNu') |
		(data.key.values=='TTWJetsToLNu') |
		(data.key.values=='ttHToNonbb'),trainVarsTTV(BDTvar)].values)
fpr, tpr, thresholds = roc_curve(data.ix[
		(data.key.values=='TTZToLLNuNu') |
		(data.key.values=='TTWJetsToLNu') |
		(data.key.values=='ttHToNonbb')]["target"], testROCttV[:,1])
print ("ROC TTV",BDTvar,auc(fpr, tpr, reorder = True))
##############
plt.figure(figsize=(20, 3))
fig, ax = plt.subplots(figsize=(18, 6))

plt.subplot(1, 3, 0+1)
histtt, xbins, ybins, im  = plt.hist2d(tt_datainTT, tt_datainTTV,
									weights=dataTT["totalWeight"].values.astype(np.float64),
									bins=nbins,
									cmap=reverse_colourmap(cm.hot))
plt.xlabel('BDT for tt')
plt.ylabel('BDT for ttV')
plt.text(0.05, 0.95, "tt sample",  fontweight='bold')
plt.text(0.85, 0.95, BDTvar,  fontweight='bold')
plt.axis((0.,1.,0.,1.))
plt.colorbar()

plt.subplot(1, 3, 1+1)
histttV, xbins, ybins, im  = plt.hist2d(ttV_datainTT, ttV_datainTTV,
										weights=dataTTV["totalWeight"].values.astype(np.float64),
										bins=nbins,
										cmap=reverse_colourmap(cm.hot))
plt.xlabel('BDT for tt')
plt.ylabel('BDT for ttV')
plt.axis((0.,1.,0.,1.))
plt.text(0.05, 0.95, "ttV sample",  fontweight='bold')
plt.text(0.85, 0.95, BDTvar,  fontweight='bold')
plt.colorbar()

plt.subplot(1, 3, 2+1)
histttH, xbins, ybins, im  = plt.hist2d(ttH_datainTT,
					ttH_datainTTV,
					weights=dataTTH["totalWeight"].values.astype(np.float64),
					bins=nbins,
					cmap=reverse_colourmap(cm.hot)
					)
plt.xlabel('BDT for tt')
plt.ylabel('BDT for ttV')
plt.axis((0.,1.,0.,1.))
plt.text(0.05, 0.95, "ttH sample",  fontweight='bold')
plt.text(0.65, 0.95, BDTvar,  fontweight='bold')
plt.colorbar()

plt.savefig(channel+"/"+BDTvar+"_2D_"+str(nbins)+"bins.pdf")
plt.clf()
#################################
### not doing: Smooth(1,"k5b");
#################################
## 1D simply linearized
fig, ax = plt.subplots(figsize=(18, 6))
hist1DttH = []
hist1DttV = []
hist1Dtt = []
for i in range(0,len(ybins)-1):
	for j in range(0,len(xbins)-1):
		hist1DttH.append(max(1e-5,histttH[i][j]+histttV[i][j]+histtt[i][j]))
		hist1DttV.append(max(1e-5,histttV[i][j]+histtt[i][j]))
		hist1Dtt.append( max(1e-5,histtt[i][j]))
xaxis = np.arange(1, len(hist1DttH)+1, 1)
print ("size 1D",len(hist1DttH),len(xaxis) ,hist1DttH[0])
plt.text(2, 18, BDTvar+" (stacked plots)",  fontweight='bold')
plt.step(xaxis,hist1DttH, label="ttH", lw=3, color='r')
plotttttV=plt.step(xaxis,hist1DttV, label="ttV",  color= 'g', lw=3)
plottt=plt.step(xaxis,hist1Dtt, label="tt", color= 'b', lw=3)
#plt.fill_between(plotttttV[0].get_xdata(orig=False), plotttttV[0].get_ydata(orig=False), plottt[0].get_ydata(orig=False),  color= 'g')
#plt.fill_between(plottt[0].get_xdata(orig=False), 0, plottt[0].get_ydata(orig=False), color= 'b')
plt.axis((0.,len(hist1DttH),-0.5,25.0))
plt.legend(loc='upper right')
plt.savefig(channel+"/"+BDTvar+"_1D_"+str(nbins)+"bins.pdf")
plt.clf()
######################################################################
## do cummulative
norm=(max(histtt.flatten())*max(histttV.flatten()))/max(histttH.flatten())
histRatio= histttH/(histtt+histttV)
histRatio = np.nan_to_num(histRatio)
histRatio[histRatio<0]=0
h = ROOT.TH1F("h","",1000,0,1.3)
h.GetXaxis().SetTitle("Likelihood ratio");
h.GetYaxis().SetTitle("Cumulative");
yTT=clsTT.predict_proba(dataTT[trainVarsTT(BDTvar)].values)[:, 1]
xTT=clsTTV.predict_proba(dataTT[trainVarsTTV(BDTvar)].values)[:, 1]
for ii in range(0,len(dataTT)) : h.Fill(histRatio[int(nbins*xTT[ii])][int(nbins*yTT[ii])])
xTTV=clsTTV.predict_proba(dataTTV[trainVarsTTV(BDTvar)].values)[:, 1]
yTTV=clsTT.predict_proba(dataTTV[trainVarsTT(BDTvar)].values)[:, 1]
for ii in range(0,len(dataTTV)) : h.Fill(histRatio[int(nbins*xTTV[ii])][int(nbins*yTTV[ii])])
# int is used here is because I want to fill the histogram with the bin position, not in the actial value
# as the BDT values range from 0-1 , int(nbins*value) is the bin position
#############################################
c = ROOT.TCanvas("c1","",200,200)
h.Scale(1./ h.Integral());
h.SetLineWidth(3)
h.SetLineColor(6)
h.GetCumulative().Draw();
nq=int(nbinsout)
xq= array.array('d', [0.] * (nq+1)) #[ii/nq for i in range(0,nq-1)] #np.empty(nq+1, dtype=object)
yq= array.array('d', [0.] * (nq+1)) # [0]*nq #np.empty(nq+1, dtype=object)
for  ii in range(0,nq) : xq[ii]=(float(ii)/nq)
xq[nq]=0.99999
h.GetQuantiles(nq+1,yq,xq)
print ("quantiles",nq,len(xq),len(yq))
line = [None for point in range(nq)]
for  jj in range(0,nq) :
		line[jj] = ROOT.TLine(xq[jj],0,xq[jj],1);
		line[jj].SetLineColor(ROOT.kRed);
		line[jj].Draw("same")
		print (xq[jj],yq[jj])
hAuxHisto = ROOT.TH1F("hAuxHisto","",nq,yq)
latex1= ROOT.TLatex();
latex1.SetTextSize(0.04);
latex1.SetTextAlign(13);  #//align at top
latex1.SetTextFont(42);
latex1.DrawLatexNDC(0.1,.95,"from ("+str(nbins)+"^2) to "+str(nbinsout)+" bins");
latex1.DrawLatexNDC(0.7,.95,BDTvar);
c.Modified();
c.Update();
c.SaveAs(channel+"/"+BDTvar+"_from"+str(nbins)+"_to_"+str(nbinsout)+"bins_Cumulative.pdf")
#################################################
## to fed analysis code
hTargetBinning = ROOT.TH2F("hTargetBinning","",100,0.,1.,100,0.,1.)
for ix in range(0,hTargetBinning.GetXaxis().GetNbins()  ):
	for iy in range(1,hTargetBinning.GetYaxis().GetNbins()) :
		bin1 = hTargetBinning.GetBin(ix,iy)
		content=histRatio[int(nbins*hTargetBinning.GetXaxis().GetBinCenter(ix))][int(nbins*hTargetBinning.GetYaxis().GetBinCenter(iy))] #GetLikeLiHood(ii))
		bin = hAuxHisto.FindBin(content)-1;
		if bin < 0 : bin=0;
		if bin+1 > hAuxHisto.GetNbinsX() : bin = hAuxHisto.GetNbinsX()-1
		hTargetBinning.SetBinContent(bin1,bin)
#c2 = ROOT.TCanvas("c2","",1000,1000)
hTargetBinning.GetXaxis().SetTitle("BDT(ttH,tt)");
hTargetBinning.GetYaxis().SetTitle("BDT(ttH,ttV)");
latex1.DrawLatexNDC(.67,.25,"from ()"+str(nbins)+"^2) to "+str(nbinsout)+" bins");
latex1.DrawLatexNDC(.67,.45,BDTvar);
#hTargetBinning.Draw("text")
#c2.SaveAs(channel+"/"+BDTvar+"_from"+str(nbins)+"bins_CumulativeBins.pdf")
#c2.SaveAs(channel+"/"+BDTvar+"_from"+str(nbins)+"bins_CumulativeBins.png")
binning = ROOT.TFile(channel+"/"+BDTvar+"_from"+str(nbins)+"_to_"+str(nbinsout)+"bins_CumulativeBins.root","recreate")
binning.cd()
hTargetBinning.Write()
binning.Close()
#################################################
## VoronoiPlot() 2D
c3 =ROOT.TCanvas("c1","",600,600);
c3.cd();
hDummy = ROOT.TH1F("hDummy","",2,0.,1);
hDummy.SetLineColor(ROOT.kWhite);
hDummy.GetYaxis().SetRangeUser(0.,1.);
hDummy.GetXaxis().SetRangeUser(0.,1.);
hDummy.GetXaxis().SetTitle("BDT(ttH,tt)");
hDummy.GetYaxis().SetTitle("BDT(ttH,ttV)");
hDummy.Draw();
print ("len auxiliary", hAuxHisto.GetNbinsX()+1)
XX=[array.array( 'd' ) for count in xrange(int(hAuxHisto.GetNbinsX()+1))] #
YY=[array.array( 'd' ) for count in xrange(int(hAuxHisto.GetNbinsX()+1))] #
for x in range(0,1000) :
	for y in range(0,1000) :
		content=histRatio[int(nbins*x/1000.)][int(nbins*y/1000.)] #GetLikeLiHood(ii))
		bin = hAuxHisto.FindBin(content)-1;
		if bin < 0 : bin=0;
		if bin+1 > hAuxHisto.GetNbinsX() : bin = hAuxHisto.GetNbinsX()-1
		XX[bin].append(x/1000.);
		YY[bin].append(y/1000.);
print ("nbins" , len(XX))
graphs=[None for count in xrange(int(hAuxHisto.GetNbinsX()+1))]
for k in range(0,hAuxHisto.GetNbinsX()) :
	graphs[k]=ROOT.TGraph(len(XX[k]),  XX[k], YY[k] ); #
	graphs[k].SetMarkerColor(k+1);
	graphs[k].SetMarkerStyle(6);
	graphs[k].Draw("PSAME");
latex= ROOT.TLatex();
latex.SetTextSize(0.04);
latex.SetTextAlign(13);  #//align at top
latex.SetTextFont(62);
latex1.DrawLatexNDC(.1,.95,"from ("+str(nbins)+"^2) to "+str(nbinsout)+" bins");
latex1.DrawLatexNDC(0.7,.95,BDTvar);
c3.Modified();
c3.Update();
c3.SaveAs(channel+"/"+BDTvar+"_from"+str(nbins)+"_to_"+str(nbinsout)+"bins_Voronoi.pdf")
c3.SaveAs(channel+"/"+BDTvar+"_from"+str(nbins)+"_to_"+str(nbinsout)+"bins_Voronoi.png")
################################################################
## VoronoiPlot1D()

hTT = ROOT.TH1F("hTTbar","",hAuxHisto.GetNbinsX(), -0.5, hAuxHisto.GetNbinsX()-0.5);
hTTW   = ROOT.TH1F("hTTW"  ,"",hAuxHisto.GetNbinsX(), -0.5, hAuxHisto.GetNbinsX()-0.5);
hTTH   = ROOT.TH1F("hTTH"  ,"",hAuxHisto.GetNbinsX(), -0.5, hAuxHisto.GetNbinsX()-0.5);
mc  = ROOT.THStack("mc","mc");
weightTTV=dataTTV["totalWeight"].values
weightTT=dataTT["totalWeight"].values
weightTTH=dataTTH["totalWeight"].values

for ii in range(0,len(dataTTV)) :
	#print (hTargetBinning.FindBin(xTTV[ii],yTTV[ii]), hTargetBinning.GetBinContent(hTargetBinning.FindBin(xTTV[ii],yTTV[ii])))
	hTTW.Fill(hTargetBinning.GetBinContent(hTargetBinning.FindBin(xTTV[ii],yTTV[ii]))+0.01,weightTTV[ii])

for ii in range(0,len(dataTT)) :
	hTT.Fill(hTargetBinning.GetBinContent(hTargetBinning.FindBin(xTT[ii],yTT[ii]))+0.01,weightTT[ii])

yTTH=clsTT.predict_proba(dataTTH[trainVarsTT(BDTvar)].values)[:, 1]
xTTH=clsTTV.predict_proba(dataTTH[trainVarsTTV(BDTvar)].values)[:, 1]
for ii in range(0,len(dataTTH)) :
		hTTH.Fill(hTargetBinning.GetBinContent(hTargetBinning.FindBin(xTTH[ii],yTTH[ii]))+0.01,weightTTH[ii])

hTT.SetFillColor( 17 );
hTTH.SetFillColor( ROOT.kRed );
hTTW.SetFillColor( 8 );

mc.Add( hTT );
mc.Add(hTTW);
mc.Add(hTTH);

c4 = ROOT.TCanvas("c1","",600,700);
c4.cd();
c4.Divide(1,2,0,0);
c4.cd(1)
ROOT.gPad.SetBottomMargin(0.001)
ROOT.gPad.SetTopMargin(0.01)
ROOT.gPad.SetRightMargin(0.01)
mc.Draw("HIST");
mc.SetMaximum(1.5* mc.GetMaximum());
mc.GetHistogram().GetYaxis().SetTitle("Expected events/bin");
#mc.GetHistogram().GetXaxis().SetTitle("Bin in the bdt1#times bdt2 plane");
mc.GetHistogram().GetXaxis().SetTitleSize(0.05);
mc.GetHistogram().GetXaxis().SetTitleOffset(1.1);
mc.GetHistogram().GetYaxis().SetTitleSize(0.05);
mc.GetHistogram().GetYaxis().SetTitleOffset(1.1);

l = ROOT.TLegend(0.7,0.8,0.9,0.9);
l.AddEntry(hTTH  , "ttH signal", "f");
l.AddEntry(hTTW  , "ttV"       , "f");
l.AddEntry(hTT, "tt"        , "f");
l.Draw();

latex= ROOT.TLatex();
latex.SetTextSize(0.05);
latex.SetTextAlign(13);  #//align at top
latex.SetTextFont(62);
latex.DrawLatexNDC(.1,.95,"CMS Simulation");
latex.DrawLatexNDC(.7,.95,"#it{36 fb^{-1}}");
latex1.DrawLatexNDC(.2,.77,"from ("+str(nbins)+"^2) to "+str(nbinsout)+" bins");
latex1.DrawLatexNDC(.2,.85,BDTvar);

c4.cd(2)
ROOT.gPad.SetTopMargin(0.001)
ROOT.gPad.SetRightMargin(0.01)
h2 = ROOT.TH1F("hErr","",hAuxHisto.GetNbinsX(), -0.5, hAuxHisto.GetNbinsX()-0.5);
h2.GetYaxis().SetTitle("err/content");
h2.GetXaxis().SetTitle("Bin in the bdt1#times bdt2 plane");
h2=hTT+hTTW
if not h2.GetSumw2N() : h2.Sumw2()
for binn in xrange(h2.GetNbinsX()) :
	if h2.GetBinContent(binn) >0 :
		ratio=h2.GetBinError(binn)/h2.GetBinContent(binn)
		print (binn,h2.GetBinContent(binn),h2.GetBinError(binn),ratio)
		h2.SetBinContent(ratio)
h2.SetLineWidth(3)
h2.SetLineColor(15)
h2.Draw("hist")

c4.Modified();
c4.Update();
c4.SaveAs(channel+"/"+BDTvar+"_from"+str(nbins)+"_to_"+str(nbinsout)+"bins_Voronoi1D.pdf")
c4.SaveAs(channel+"/"+BDTvar+"_from"+str(nbins)+"_to_"+str(nbinsout)+"bins_Voronoi1D.png")




"""
void LikeliHOOD::VoronoiPlot()
{
  vector<TGraph*> graphs; graphs.clear();
  vector<Double_t>* X = new vector<Double_t>[hAuxHisto->GetNbinsX()+1];
  vector<Double_t>* Y = new vector<Double_t>[hAuxHisto->GetNbinsX()+1];

  for (Double_t x = -1; x < 1.; x = x + 1e-3){
      for (Double_t y = -1; y < 1.; y = y + 1e-3){
	Int_t k = GetCluster(Point(x,y,-1));
	cout << k << endl;
	X[k].push_back(x);
	Y[k].push_back(y);
	// cout << "Pushed " << endl;
      }
  }

  cout << "Done " << endl;



  TCanvas* c = new TCanvas();
  c->cd();
  setTDRStyle();

  TH1F* hDummy = new TH1F("hDummy","",2,-1,1);
  hDummy->SetBinContent(1, 1.);
  hDummy->SetBinContent(2,-1.);
  hDummy->SetLineColor(kWhite);
  hDummy->GetYaxis()->SetRangeUser(-1.,1.);
  hDummy->GetXaxis()->SetTitle("BDT(ttH,tt)");
  hDummy->GetYaxis()->SetTitle("BDT(ttH,ttV)");
  hDummy->Draw();
  cout << "Done... now plotting" << endl;

  TText t;
  t.SetTextSize(0.08);
  for (unsigned int k = 0; k < hAuxHisto->GetNbinsX(); ++k){
    graphs.push_back(new TGraph( X[k].size(), &X[k][0], &Y[k][0] ));
    graphs[k]->SetMarkerColor(k);
    graphs[k]->SetMarkerStyle(6);
    graphs[k]->Draw("PSAME");

    //t.SetTextColor(k+1);
    //    t.DrawText(fCentroids[k].fX, fCentroids[k].fY, Form("%d",k));

  }

  c->Print(Form("likelihoodBased_2d_%s.pdf", (nLep_==3 ? "3l" : "2lss")));
  c->Print(Form("likelihoodBased_2d_%s.png", (nLep_==3 ? "3l" : "2lss")));


void LikeliHOOD::Test()
{
  TCanvas* c = new TCanvas();
  c->cd();
  setTDRStyle();

  TH1F* hTTbar = new TH1F("hTTbar","",hAuxHisto->GetNbinsX(), -0.5, hAuxHisto->GetNbinsX()-0.5);
  TH1F* hTTW   = new TH1F("hTTW"  ,"",hAuxHisto->GetNbinsX(), -0.5, hAuxHisto->GetNbinsX()-0.5);
  TH1F* hTTH   = new TH1F("hTTH"  ,"",hAuxHisto->GetNbinsX(), -0.5, hAuxHisto->GetNbinsX()-0.5);
  THStack* mc  = new THStack("mc","mc");
  vector<Point>::iterator point;
  cout << "TTbar size is " << fTTbarMC.size() << endl;
  for (point = fTTbarMC.begin(); point != fTTbarMC.end(); ++point)
    hTTbar->Fill( GetCluster( *point), point->fW);
  for (point = fTTWMC.begin(); point != fTTWMC.end(); ++point)
    hTTW->Fill( GetCluster( *point), point->fW);
  for (point = fTTHMC.begin(); point != fTTHMC.end(); ++point)
    hTTH->Fill( GetCluster( *point), point->fW);
  cout << hTTbar->Integral() << " " << hTTbar->GetEntries() << endl;
  hTTbar->SetFillColor( kRed     );
  hTTH->SetFillColor( kBlue    );
  hTTW->SetFillColor( kMagenta );

  mc->Add( hTTbar ); mc->Add(hTTW); mc->Add(hTTH);
  mc->Draw("HIST");
  mc->SetMaximum(1.5* mc->GetMaximum());
  mc->GetHistogram()->GetYaxis()->SetTitle("Expected events/bin");
  mc->GetHistogram()->GetXaxis()->SetTitle("Bin in the bdt1#times bdt2 plane");
  mc->GetHistogram()->GetXaxis()->SetTitleSize(0.05);
  mc->GetHistogram()->GetXaxis()->SetTitleOffset(1.1);
  mc->GetHistogram()->GetYaxis()->SetTitleSize(0.05);
  mc->GetHistogram()->GetYaxis()->SetTitleOffset(1.1);

  TLegend* l = new TLegend(0.7,0.8,0.9,0.9);
  l->AddEntry(hTTH  , "ttH signal", "f");
  l->AddEntry(hTTW  , "ttV"       , "f");
  l->AddEntry(hTTbar, "tt"        , "f");
  l->Draw();

  TLatex latex;
  latex.SetTextSize(0.05);
  latex.SetTextAlign(13);  //align at top
  latex.SetTextFont(62);
  latex.DrawLatexNDC(.1,.95,"CMS Simulation");
  latex.DrawLatexNDC(.7,.95,"#it{36 fb^{-1}}");

  c->Modified();
  c->Update();

  c->Print(Form("likelihoodBased_1d_%s.pdf", (nLep_==3 ? "3l" : "2lss")));
  c->Print(Form("likelihoodBased_1d_%s.png", (nLep_==3 ? "3l" : "2lss")));
}

"""



###########################################################################
