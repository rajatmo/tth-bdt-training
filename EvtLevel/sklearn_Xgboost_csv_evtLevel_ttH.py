import sys , time
#import sklearn_to_tmva
import sklearn
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
import pandas
import matplotlib.mlab as mlab
from scipy.stats import norm
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
import psutil
import os
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

#from tth-bdt-training-test.data_manager import load_data
#dm = __import__("tth-bdt-training-test.data_manager.py")

#import imp
#dm = imp.load_module("dm_name", "tth-bdt-training-test/data_manager.py")

execfile("../python/data_manager.py")
# run: python sklearn_Xgboost_csv_evtLevel_ttH.py --channel '1l_2tau' --variables "HTTWithKinFitKin" --bdtType "evtLevelTT_TTH" --ntrees  --treeDeph --lr  >/dev/null 2>&1
# we have many trees
# https://stackoverflow.com/questions/38238139/python-prevent-ioerror-errno-5-input-output-error-when-running-without-stdo

#"""
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--channel ", type="string", dest="channel", help="The ones whose variables implemented now are:\n   - 1l_2tau\n   - 2lss_1tau\n It will create a local folder and store the report*/xml", default='T')
parser.add_option("--variables", type="string", dest="variables", help="  Set of variables to use -- it shall be put by hand in the code, in the fuction trainVars(all)\n Example to 2ssl_2tau   \n                              all==True -- all variables that should be loaded (training + weights) -- it is used only once\n                               all==False -- only variables of training (not including weights) \n  For the channels implemented I defined 3 sets of variables/each to confront at limit level\n  trainvar=allVar -- all variables that are avaible to training (including lepton IDs, this is here just out of curiosity) \n  trainvar=oldVar -- a minimal set of variables (excluding lepton IDs and lep pt's)\n  trainvar=notForbidenVar -- a maximal set of variables (excluding lepton IDs and lep pt's) \n  trainvar=notForbidenVarNoMEM -- the same as above, but excluding as well MeM variables", default=1000)
parser.add_option("--bdtType", type="string", dest="bdtType", help=" evtLevelTT_TTH or evtLevelTTV_TTH", default='T')
parser.add_option("--HypOpt", action="store_true", dest="HypOpt", help="If you call this will not do plots with repport", default=False)
parser.add_option("--doXML", action="store_true", dest="doXML", help="Do save not write the xml file", default=False)
parser.add_option("--doPlots", action="store_true", dest="doPlots", help="Fastsim Loose/Tight vs Fullsim variables plots", default=False)
parser.add_option("--oldNtuple", action="store_true", dest="oldNtuple", help="use Matthias", default=False)
parser.add_option("--ntrees ", type="int", dest="ntrees", help="hyp", default=2000)
parser.add_option("--treeDeph", type="int", dest="treeDeph", help="hyp", default=2)
parser.add_option("--lr", type="float", dest="lr", help="hyp", default=0.01)
parser.add_option("--mcw", type="int", dest="mcw", help="hyp", default=1)
(options, args) = parser.parse_args()
#""" bdtType=="evtLevelTTV_TTH"

doPlots=options.doPlots
bdtType=options.bdtType
trainvar=options.variables
hyppar=str(options.variables)+"_ntrees_"+str(options.ntrees)+"_deph_"+str(options.treeDeph)+"_mcw_"+str(options.mcw)+"_lr_0o0"+str(int(options.lr*100))

channel=options.channel

if channel=='1l_2tau':
	channelInTree='1l_2tau_OS_Tight'
	inputPath='/hdfs/local/acaan/ttHAnalysis/2016/1l_2tau_2018Jan26_forBDT_tightLmediumT/histograms/1l_2tau/forBDTtraining_OS/'
	channelInTreeTight='1l_2tau_OS_Tight'
	inputPathTight='/hdfs/local/acaan/ttHAnalysis/2016/1l_2tau_2018Jan26_forBDT_tightLtightT/histograms/1l_2tau/forBDTtraining_OS/'
	channelInTreeFS='1l_2tau_OS_Tight'
	inputPathTightFS='/hdfs/local/acaan/ttHAnalysis/2016/2018Jan28_BDT_toTrees_FS_looseT/histograms/1l_2tau/Tight_OS/'
	criteria=[]
	testtruth="bWj1Wj2_isGenMatchedWithKinFit"
	FullsimWP="TightLep_MediumTau"
	FastsimWP="TightLep_TightTau"
	FastsimTWP="TightLep_TightTau"

# 2los_1tau_2018Mar14_BDT_TLepTTau
if channel=='2los_1tau':
	channelInTree='2los_1tau_Loose'
	inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2los_1tau_2018Mar14_BDT_LLepMTau/histograms/2los_1tau/forBDTtraining/'
	FastsimWP= "LooseLep_TightTau"
	criteria=[]
	testtruth="bWj1Wj2_isGenMatchedWithKinFit"
	channelInTreeTight='2los_1tau_Tight'
	inputPathTight='/hdfs/local/acaan/ttHAnalysis/2016/2los_1tau_2018Mar14_BDT_TLepLTau/histograms/2los_1tau/forBDTtraining/'
	FastsimTWP="TightLep_MediumTau"
	channelInTreeFS='2los_1tau_Tight'
	inputPathTightFS='/hdfs/local/acaan/ttHAnalysis/2016/2los_1tau_2018Mar14_BDT_fullsim_TLepVTTau/histograms/2los_1tau/Tight/'
	FullsimWP= "TightLep_VTightTau"

if channel=='2lss_1tau':
	#channelInTree='2lss_1tau_lepSS_sumOS_Tight'
	channelInTree='2lss_1tau_lepSS_sumOS_Loose'
	inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2lss_1tau_2018Feb27_BDT_LLepTTau/histograms/2lss_1tau/forBDTtraining_SS_OS/'
	FastsimWP= "LooseLep_TightTau"
	#channelInTree='2lss_1tau_lepSS_sumOS_Tight'
	#inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2lss_1tau_2018Feb27_BDT_TLepVVLTau/histograms/2lss_1tau/forBDTtraining_SS_OS/'
	criteria=[]
	testtruth="bWj1Wj2_isGenMatchedWithKinFit"
	channelInTreeTight='2lss_1tau_lepSS_sumOS_Tight'
	#channelInTreeTight='2lss_1tau_lepSS_sumOS_Loose'
	inputPathTight='/hdfs/local/acaan/ttHAnalysis/2016/2lss_1tau_2018Feb26_BDT_TLepTTau/histograms/2lss_1tau/forBDTtraining_SS_OS/'
	FastsimTWP="TightLep_MediumTau"
	if bdtType=="evtLevelSUM_TTH_M" :
		channelInTreeFS='2lss_1tau_lepSS_sumOS_Tight'
		inputPathTightFS='/hdfs/local/acaan/ttHAnalysis/2016/2lss_1tau_2018Feb26_VHbb_trees_TLepMTau/histograms/2lss_1tau/forBDTtraining_SS_OS/'
		FullsimWP= "TightLep_MediumTau"
	if bdtType=="evtLevelSUM_TTH_T" :
		channelInTreeFS='2lss_1tau_lepSS_sumOS_Tight'
		inputPathTightFS='/hdfs/local/acaan/ttHAnalysis/2016/2lss_1tau_2018Feb26_VHbb_trees_TLepTTau/histograms/2lss_1tau/forBDTtraining_SS_OS/'
		FullsimWP= "TightLep_TightTau"
	else :
		channelInTreeFS='2lss_1tau_lepSS_sumOS_Tight'
		inputPathTightFS='/hdfs/local/acaan/ttHAnalysis/2016/2lss_1tau_2018Feb26_VHbb_trees_TLepMTau/histograms/2lss_1tau/forBDTtraining_SS_OS/'
		FullsimWP= "TightLep_MediumTau"

doFS2=False
if channel=="2l_2tau": # see Feb10
	#channelInTree='2l_2tau_sumOS_Loose'
	channelInTree='2l_2tau_sumOS_Tight'
	inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2l_2tau_2018Feb18_BDT_TLepVLTau/histograms/2l_2tau/forBDTtraining_sumOS/'
	FastsimWP="TightLep_VVLooseTau"
	criteria=[]
	testtruth="None"
	channelInTreeTight='2l_2tau_sumOS_Tight'
	inputPathTight='/hdfs/local/acaan/ttHAnalysis/2016/2l_2tau_2018Feb18_BDT_TLepMTau/histograms/2l_2tau/forBDTtraining_sumOS/'
	FastsimTWP="TightLep_MediumTau"
	if bdtType=="evtLevelSUM_TTH_M" :
		channelInTreeFS='2l_2tau_sumOS_Tight'
		inputPathTightFS='/hdfs/local/acaan/ttHAnalysis/2016/2l_2tau_2018Feb18_BDT_VHbb_TLepMTau/histograms/2l_2tau/forBDTtraining_sumOS/'
		FullsimWP="TightLep_MediumTau"
	if bdtType=="evtLevelSUM_TTH_T" :
		channelInTreeFS='2l_2tau_sumOS_Tight'
		inputPathTightFS='/hdfs/local/acaan/ttHAnalysis/2016/2l_2tau_2018Feb18_BDT_VHbb_TLepTTau/histograms/2l_2tau/forBDTtraining_sumOS/'
		FullsimWP="TightLep_TightTau"
	if bdtType=="evtLevelSUM_TTH_VT" :
		channelInTreeFS='2l_2tau_sumOS_Tight' # ???? processing again
		inputPathTightFS='/hdfs/local/acaan/ttHAnalysis/2016/2l_2tau_2018Feb18_BDT_VHbb_TLepTTau/histograms/2l_2tau/forBDTtraining_sumOS/'
		FullsimWP="TightLep_VTightTau"
	else :
		channelInTreeFS='2l_2tau_sumOS_Tight'
		inputPathTightFS='/hdfs/local/acaan/ttHAnalysis/2016/2l_2tau_2018Feb18_BDT_VHbb_TLepMTau/histograms/2l_2tau/forBDTtraining_sumOS/'
		FullsimWP="TightLep_MediumTau"
	channelInTreeFS2='2l_2tau_sumOS_Loose'
	inputPathTightFS2='/hdfs/local/acaan/ttHAnalysis/2016/2l_2tau_2018Feb18_BDT_VHbb_TLepVVLTau/histograms/2l_2tau/forBDTtraining_sumOS/'
	FullsimWP2="TightLep_VVLooseTau"


if channel=="3l_1tau":
	channelInTree='3l_1tau_OS_lepLoose_tauTight'
	inputPath='/hdfs/local/acaan/ttHAnalysis/2016/3l_1tau_2018Feb19_BDT_LLepVLTau/histograms/3l_1tau/forBDTtraining_OS/'
	criteria=[]
	testtruth="None"
	channelInTreeTight='3l_1tau_OS_lepTight_tauTight'
	inputPathTight='/hdfs/local/acaan/ttHAnalysis/2016/3l_1tau_2018Feb19_BDT_TLepMTau/histograms/3l_1tau/forBDTtraining_OS/'
	channelInTreeFS='3l_1tau_OS_lepTight_tauTight'
	inputPathTightFS='/hdfs/local/acaan/ttHAnalysis/2016/3l_1tau_2018Feb19_VHbb_trees_TLepMTau/histograms/3l_1tau/forBDTtraining_OS/'
	FullsimWP="TightLep_MediumTau"
	FastsimWP="LooseLep_VLooseTau"
	FastsimTWP="TightLep_MediumTau"

print "reading "+inputPath
print "reading tight "+inputPathTight
print "reading FS "+inputPathTightFS

import shutil,subprocess
proc=subprocess.Popen(['mkdir '+options.channel],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()

def trainVars(all):

        if channel=="2los_1tau" and all==True  :return [
			'HadTop_eta', 'HadTop_pt',
			'avg_dr_jet', 'dr_lep1_tau_os', 'dr_lep2_tau_ss',
			'dr_lepOS_HTfitted', 'dr_lepOS_HTunfitted', 'dr_lepSS_HTfitted', 'dr_lepSS_HTunfitted',
			'dr_leps', 'dr_tau_HTfitted', 'dr_tau_HTunfitted', #'evtWeight',
			'fitHTptoHTmass', 'fitHTptoHTpt', #'genTopPt', 'genWeight',
			'htmiss', 'lep1_conePt', 'lep1_eta', #'lep1_fake_prob', 'lep1_genLepPt',
			'lep1_pt', 'lep1_tth_mva', 'lep2_conePt', 'lep2_eta', #'lep2_fake_prob', 'lep2_genLepPt',
			'lep2_pt', 'lep2_tth_mva', #'lumiScale',
			'mT_lep1', 'mT_lep2', 'mTauTauVis', #'mass_lepOS_HTfitted', 'mass_lepSS_HTfitted',
			'max_lep_eta', 'mbb', 'mbb_loose',
			'min_lep_eta', 'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet',
			#'mvaDiscr_2lss', 'mvaOutput_2lss_ttV', 'mvaOutput_2lss_ttbar', 'mvaOutput_hadTopTagger',
			'mvaOutput_hadTopTaggerWithKinFit', 'ptbb', 'ptbb_loose', 'ptmiss', 'tau_eta',
			#'tau_fake_prob', 'tau_genTauPt', 'tau_mva', 'lep2_charge', 'lep2_isTight',
			'tau_pt', 'unfittedHadTop_eta', 'unfittedHadTop_pt', #'bWj1Wj2_isGenMatched',
			'bWj1Wj2_isGenMatchedWithKinFit', #'hadtruth', 'lep1_charge', 'lep1_isTight',
			'lep1_tau_charge', 'nBJetLoose', 'nBJetMedium',
			'nJet', 'nLep', 'nTau', #'tau_charge', 'tau_isTight', 'run', 'lumi', 'evt'
		]

        if trainvar=="noHTT" and channel=="2los_1tau" and bdtType=="evtLevelTT_TTH" and all==False :
			return [
			'avg_dr_jet', #'dr_lep1_tau_os',
			'dr_lep2_tau_ss',
			'dr_leps',
			#'lep1_conePt',
			#'lep2_conePt',
			#'mT_lep1',
			'mT_lep2',
			'mTauTauVis',
			'tau_pt', 'tau_eta',
			#'max_lep_eta', #'min_lep_eta', #'lep2_eta','lep1_eta',
			'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet',
			#'mbb', 'ptbb', # (medium b)
			'mbb_loose', #'ptbb_loose',
			'ptmiss', #'htmiss',
			#'nBJetLoose',
			#'nBJetMedium',
			'nJet',
			]

        if trainvar=="noHTT" and channel=="2los_1tau" and bdtType=="evtLevelTTV_TTH" and all==False :
			return [
			'avg_dr_jet', #'dr_lep1_tau_os',
			'dr_lep2_tau_ss',
			'dr_leps',
			#'lep1_conePt',
			#'lep2_conePt',
			#'mT_lep1',
			'mT_lep2',
			'mTauTauVis',
			'tau_pt', 'tau_eta',
			#'max_lep_eta', #'min_lep_eta', #'lep2_eta','lep1_eta',
			'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet',
			#'mbb', 'ptbb', # (medium b)
			'mbb_loose', #'ptbb_loose',
			'ptmiss', #'htmiss',
			#'nBJetLoose',
			#'nBJetMedium',
			'nJet',
			]

        if trainvar=="HTT" and channel=="2los_1tau" and bdtType=="evtLevelTT_TTH" and all==False :
			return [
			'avg_dr_jet', #'dr_lep1_tau_os',
			'dr_lep2_tau_ss',
			'dr_leps',
			#'lep1_conePt',
			#'lep2_conePt',
			#'mT_lep1',
			'mT_lep2',
			'mTauTauVis',
			'tau_pt', 'tau_eta',
			#'max_lep_eta', #'min_lep_eta', #'lep2_eta','lep1_eta',
			'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet',
			#'mbb', 'ptbb', # (medium b)
			'mbb_loose', #'ptbb_loose',
			'ptmiss', #'htmiss',
			#'nBJetLoose',
			#'nBJetMedium',
			'nJet',
			'mvaOutput_hadTopTaggerWithKinFit',
			'unfittedHadTop_pt',
			#'dr_lepOS_HTfitted', 'dr_lepOS_HTunfitted',
			#'dr_lepSS_HTfitted', #'dr_lepSS_HTunfitted',
			'dr_tau_HTfitted', #'dr_tau_HTunfitted',
			'fitHTptoHTmass', #'fitHTptoHTpt',
			#'HadTop_eta', 'HadTop_pt',
			#'mass_lepOS_HTfitted', 'mass_lepSS_HTfitted',
			]

        if channel=="2lss_1tau" and all==True  :return [
		#'HadTop_eta', 'HadTop_pt', 'MT_met_lep1', 'avg_dr_jet',
		#'bWj1Wj2_isGenMatched', 'bWj1Wj2_isGenMatchedWithKinFit',
		#'evtWeight',
		#'fitHTptoHTpt', 'fittedHadTop_eta', 'fittedHadTop_pt', #'genTopPt', 'genWeight', 'hadtruth',
		#'htmiss', 'lep1_conePt',  #'lep1_frWeight',
		#'lep1_genLepPt',
		'lep1_pt', 'lep2_pt', 'tau_pt',  'mTauTauVis1',
		#'lep2_conePt',
		'lep1_eta', 'lep2_eta', 'tau_eta', 'mTauTauVis2',#'lep2_frWeight', 'lep2_genLepPt',
		'dr_lep1_tau', 'dr_lep2_tau', 'dr_leps', 'ptmiss',
		#'lep1_tth_mva','lep2_tth_mva',
		#'log_memOutput_tt', 'log_memOutput_ttH', 'log_memOutput_ttZ', 'log_memOutput_ttZ_Zll',
		#'lumiScale',
		#'mT_lep1', 'mT_lep2', 'max_lep_eta',
		#'mbb', 'ptbb', 'mbb_loose', 'ptbb_loose',
		#'memOutput_LR', 'memOutput_errorFlag', 'memOutput_isValid', 'memOutput_ttZ_LR', 'memOutput_ttZ_Zll_LR', 'memOutput_tt_LR',
		#'mindr_lep1_jet',
		#'mindr_lep2_jet',
		#'mindr_tau_jet',
		#'mvaOutput_2lss_ttbar', 'mvaOutput_Hj_tagger', 'mvaOutput_Hjj_tagger',
		#'mvaOutput_hadTopTagger', 'mvaOutput_hadTopTaggerWithKinFit',
		#'nJet25_Recl', 'ncombo',  #'tau_frWeight', 'tau_genTauPt',
		'tau_mva', #'unfittedHadTop_eta', 'unfittedHadTop_pt', #'lep1_isTight', 'lep2_isTight', 'tau_isTight'
		#"mbb",
		#'nBJetLoose',
		'nBJetMedium', 'nJet', #'nLep',
		"max_eta_Lep"
		#"lep1_fake_prob","lep2_fake_prob"
		]

        if trainvar=="noHTT" and channel=="2lss_1tau" and bdtType=="evtLevelTT_TTH" and all==False :
			return [
			'avg_dr_jet',
			'dr_lep1_tau',
			'dr_lep2_tau',
			'dr_leps',
			'lep1_conePt',
			'lep2_conePt',
			'mT_lep2',
			'mTauTauVis1',
			'mTauTauVis2',
			'mbb',
			'mindr_lep1_jet',
			'mindr_lep2_jet',
			'mindr_tau_jet',
			'nJet',
			'ptmiss',
			'tau_pt',
			]

        if trainvar=="noHTT" and channel=="2lss_1tau" and bdtType=="evtLevelTTV_TTH" and all==False :
			return [
			'avg_dr_jet',
			'dr_lep1_tau',
			'dr_leps',
			'lep1_conePt',
			'lep2_conePt',
			'mT_lep1',
			'mT_lep2',
			'mTauTauVis1',
			'mTauTauVis2',
			'mindr_lep1_jet',
			'mindr_lep2_jet',
			'mindr_tau_jet',
			'ptmiss',
			'max_lep_eta',
			'tau_pt'
			]

        if trainvar=="noHTT" and channel=="2lss_1tau" and "evtLevelSUM_TTH" in bdtType and all==False :
			return [
			'avg_dr_jet',
			'dr_lep1_tau',
			'dr_lep2_tau',
			'dr_leps',
			'lep1_conePt',
			'lep2_conePt',
			'mT_lep1',
			'mT_lep2',
			'mTauTauVis1',
			'mTauTauVis2',
			'max_lep_eta',
			'mbb',
			'mindr_lep1_jet',
			'mindr_lep2_jet',
			'mindr_tau_jet',
			'nJet',
			'ptmiss',
			'tau_pt',
			]

        if trainvar=="HTT" and channel=="2lss_1tau" and bdtType=="evtLevelTT_TTH" and all==False :
			return [
			"avg_dr_jet",
			"dr_lep1_tau",
			"dr_lep2_tau",
			"dr_leps",
			"lep2_conePt",
			"mT_lep1",
			"mT_lep2",
			"mTauTauVis2",
			"max_lep_eta",
			"mbb",
			"mindr_lep1_jet",
			"mindr_lep2_jet",
			"mindr_tau_jet",
			"nJet",
			"ptmiss",
			"tau_pt",
			'mvaOutput_hadTopTaggerWithKinFit',
			'unfittedHadTop_pt'
			]
			"""
			'avg_dr_jet',
			'dr_lep1_tau',
			'dr_lep2_tau',
			'dr_leps',
			'lep1_conePt',
			'lep2_conePt',
			'mT_lep2',
			'mTauTauVis2',
			'mbb',
			'mindr_lep1_jet',
			'mindr_lep2_jet',
			'mindr_tau_jet',
			'ptmiss',
			'tau_pt',
			'mvaOutput_hadTopTaggerWithKinFit',
			'mvaOutput_Hj_tagger',
			'unfittedHadTop_pt',
			"""

        if trainvar=="HTT_LepID" and channel=="2lss_1tau" and bdtType=="evtLevelTT_TTH" and all==False :
			return [
			'mTauTauVis1', 'mTauTauVis2', 'tau_pt',
			'mvaOutput_hadTopTaggerWithKinFit',
			'lep1_tth_mva', 'lep2_tth_mva'
			]

        if trainvar=="oldVar"  and channel=="2lss_1tau"  and bdtType=="evtLevelTT_TTH" and all==False :return [
		"max_lep_eta",
		"nJet25_Recl",
		"mindr_lep1_jet",
		"mindr_lep2_jet",
		"min(met_pt,400)",
		"avg_dr_jet",
		"MT_met_lep1"
		]

        if trainvar=="oldVar"  and channel=="2lss_1tau"  and bdtType=="evtLevelTTV_TTH" and all==False :return [
		"max_lep_eta",
		"MT_met_lep1",
		"nJet25_Recl",
		"mindr_lep1_jet",
		"mindr_lep2_jet",
		"lep1_conePt",
		"lep2_conePt"
		]

        if trainvar=="oldVarA"  and channel=="2lss_1tau"  and bdtType=="evtLevelTT_TTH" and all==False :return [
		"nJet","mindr_lep1_jet","avg_dr_jet",
		"max_lep_eta",
		"lep2_conePt","dr_leps","tau_pt","dr_lep1_tau"
		]

        if trainvar=="oldVarA"  and channel=="2lss_1tau"  and bdtType=="evtLevelTTV_TTH" and all==False :return [
		"mindr_lep1_jet","mindr_lep2_jet", "avg_dr_jet", "max_lep_eta",
		"lep1_conePt", "lep2_conePt", "mT_lep1", "dr_leps", "mTauTauVis1", "mTauTauVis2"
		]

        if trainvar=="HTT_LepID" and channel=="2lss_1tau" and bdtType=="evtLevelTTV_TTH" and all==False :
			return [
			'dr_lep1_tau', 'dr_lep2_tau', 'dr_leps',
			'mT_lep1', 'mT_lep2', 'mTauTauVis1',
			'mTauTauVis2', 'mindr_lep1_jet', 'mindr_lep2_jet',
			'mvaOutput_hadTopTaggerWithKinFit', 'nJet25_Recl', 'ptmiss', 'tau_pt',
			'unfittedHadTop_pt', 'lep1_pt',
			'lep1_tth_mva', 'lep2_tth_mva', 'tau_mva'
			]

        if trainvar=="HTT" and channel=="2lss_1tau" and bdtType=="evtLevelTTV_TTH" and all==False :
			return [
			'avg_dr_jet',
			'dr_lep1_tau',
			'dr_lep2_tau',
			'dr_leps',
			'lep1_conePt',
			'lep2_conePt',
			'mT_lep1',
			'mT_lep2',
			'mTauTauVis1',
			'mTauTauVis2',
			'max_lep_eta',
			'mindr_lep1_jet',
			'mindr_lep2_jet',
			'mindr_tau_jet',
			'nJet',
			'ptmiss',
			'tau_pt',
			'mvaOutput_hadTopTaggerWithKinFit',
			]

        if trainvar=="HTTMEM" and channel=="2lss_1tau" and bdtType=="evtLevelTT_TTH" and all==False :
			return [
			'avg_dr_jet',
			'dr_lep1_tau',
			'dr_lep2_tau',
			'dr_leps',
			'lep2_conePt',
			'mT_lep1',
			'mT_lep2',
			'mTauTauVis2',
			'max_lep_eta',
			'mbb',
			'mindr_lep1_jet',
			'mindr_lep2_jet',
			'mindr_tau_jet',
			'nJet',
			'ptmiss',
			'tau_pt',
			"memOutput_LR",
			'mvaOutput_hadTopTaggerWithKinFit',
			'unfittedHadTop_pt',
			]

        if trainvar=="HTTMEM" and channel=="2lss_1tau" and bdtType=="evtLevelTTV_TTH" and all==False :
			return [
			'avg_dr_jet',
			'dr_lep1_tau',
			'dr_lep2_tau',
			'dr_leps',
			'lep2_conePt',
			'mT_lep1',
			'mT_lep2',
			'mTauTauVis2',
			'max_lep_eta',
			'mbb',
			'mindr_lep1_jet',
			'mindr_lep2_jet',
			'mindr_tau_jet',
			'nJet',
			'ptmiss',
			'tau_pt',
			"memOutput_LR",
			'mvaOutput_hadTopTaggerWithKinFit',
			'unfittedHadTop_pt',
			]

        if trainvar=="HTTMEM" and channel=="2lss_1tau" and bdtType=="evtLevelSUM_TTH_M" and all==False :
			return [
			'avg_dr_jet',
			'dr_lep1_tau',
			'dr_lep2_tau',
			'dr_leps',
			'lep2_conePt',
			'mT_lep1',
			'mT_lep2',
			'mTauTauVis2',
			'max_lep_eta',
			'mbb',
			'mindr_lep1_jet',
			'mindr_lep2_jet',
			'mindr_tau_jet',
			'nJet',
			'ptmiss',
			'tau_pt',
			#"memOutput_LR",
			'mvaOutput_hadTopTaggerWithKinFit',
			#'mvaOutput_Hj_tagger',
			'HadTop_pt', #'unfittedHadTop_pt',
			]

        if trainvar=="HTT" and channel=="2lss_1tau" and "evtLevelSUM_TTH_M" in bdtType and all==False :
			return [
			'avg_dr_jet',
			'dr_lep1_tau',
			'dr_lep2_tau',
			'dr_leps',
			'lep2_conePt',
			'mT_lep1',
			'mT_lep2',
			'mTauTauVis2',
			'max_lep_eta',
			'mbb',
			'mindr_lep1_jet',
			'mindr_lep2_jet',
			'mindr_tau_jet',
			'nJet',
			'ptmiss',
			'tau_pt',
			'mvaOutput_hadTopTaggerWithKinFit',
			#'mvaOutput_Hj_tagger',
			'unfittedHadTop_pt',
			]

        if trainvar=="HTTMEM" and channel=="2lss_1tau" and bdtType=="evtLevelSUM_TTH_T" and all==False :
			return [
			'avg_dr_jet',
			'dr_lep1_tau',
			'dr_lep2_tau',
			'dr_leps',
			'lep2_conePt',
			'mT_lep1',
			'mT_lep2',
			'mTauTauVis2',
			'max_lep_eta',
			'mbb',
			'mindr_lep1_jet',
			'mindr_lep2_jet',
			'mindr_tau_jet',
			'nJet',
			'ptmiss',
			'tau_pt',
			"memOutput_LR",
			'mvaOutput_hadTopTaggerWithKinFit',
			'mvaOutput_Hj_tagger',
			'unfittedHadTop_pt',
			]

        if channel=="2l_2tau" and all==True : return [
		"lep1_pt", "lep1_eta",
		"lep2_pt", "lep2_eta", "dr_leps",
		"tau1_pt",  "tau1_eta",
		"tau2_pt", "tau2_eta",
		"dr_taus", "mTauTauVis", "cosThetaS_hadTau",
		'avr_lep_eta','avr_tau_eta',
		"nJet", "nBJetLoose",
		]

        if trainvar=="noHTT" and channel=="2l_2tau"  and bdtType=="evtLevelTTV_TTH" and all==False :return [
			"mTauTauVis", "cosThetaS_hadTau",
			"lep1_conePt", #"lep1_eta", #"lep1_tth_mva",
			"lep2_conePt", #"lep2_eta", #"lep2_tth_mva",
			"mT_lep1", "mT_lep2",
			"dr_taus", #"dr_leps",
			"min_dr_lep_jet",
			"mindr_tau1_jet",
			"avg_dr_jet",
			"min_dr_lep_tau","max_dr_lep_tau",
			"is_OS",
			"nJet",
			]

        if trainvar=="noHTT" and channel=="2l_2tau"  and bdtType=="evtLevelTT_TTH" and all==False :return [
			"mTauTauVis", "cosThetaS_hadTau",
			'tau1_pt',
			'tau2_pt',
			"tau2_eta",
			"mindr_lep1_jet",
			"mT_lep1", #"mT_lep2",
			"mindr_tau_jet",
			"max_dr_lep_tau",
			"is_OS",
			"nBJetLoose",
			]

        if trainvar=="noHTT" and channel=="2l_2tau"  and bdtType=="evtLevelSUM_TTH_M" and all==False :
			return [
			"mTauTauVis", "cosThetaS_hadTau",
			'tau1_pt','tau2_pt',
			"lep2_conePt", #"lep2_eta", #"lep2_tth_mva",
			"mindr_lep1_jet",
			"mT_lep1", #"mT_lep2",
			"mindr_tau_jet",
			"avg_dr_jet",
			"avr_dr_lep_tau",
			"dr_taus", #"dr_leps",
			"is_OS",
			"nBJetLoose",
			"mbb_loose"
			]

        if trainvar=="noHTT" and channel=="2l_2tau"  and bdtType=="evtLevelSUM_TTH_T" and all==False :
			return [
			"mTauTauVis", "cosThetaS_hadTau",
			'tau1_pt','tau2_pt',
			"lep2_conePt", #"lep2_eta", #"lep2_tth_mva",
			"mindr_lep1_jet",
			"mT_lep1", #"mT_lep2",
			"mindr_tau_jet",
			"avg_dr_jet",
			"avr_dr_lep_tau",
			"dr_taus", #"dr_leps",
			"is_OS",
			"nBJetLoose",
			"mbb_loose"
			]

        if trainvar=="noHTT" and channel=="2l_2tau"  and bdtType=="evtLevelSUM_TTH_VT" and all==False :
			return [
			"mTauTauVis", "cosThetaS_hadTau",
			'tau1_pt','tau2_pt',
			"lep2_conePt", #"lep2_eta", #"lep2_tth_mva",
			"mindr_lep1_jet",
			"mT_lep1", #"mT_lep2",
			"mindr_tau_jet",
			"avg_dr_jet",
			"avr_dr_lep_tau",
			"dr_taus", #"dr_leps",
			"is_OS",
			"nBJetLoose",
			"mbb_loose"
			]

        if channel=="1l_2tau" and all==True :return [
		"lep_conePt", #
		"mindr_lep_jet", "mindr_tau1_jet", "mindr_tau2_jet",
		"avg_dr_jet", "ptmiss",
		"htmiss", "mT_lep",
		"tau1_pt", "tau2_pt",
		"tau1_eta", "tau2_eta",
		"dr_taus",
		"dr_lep_tau_os",
		"dr_lep_tau_ss",
		"dr_lep_tau_lead",
		"dr_lep_tau_sublead",
		"dr_HadTop_tau_lead","dr_HadTop_tau_sublead", "dr_HadTop_tautau",
		"dr_HadTop_lepton","mass_HadTop_lepton",
		"costS_HadTop_tautau",
		"costS_tau",
		"mTauTauVis",
		"mvaOutput_hadTopTagger",
		"mT_lepHadTop", #
		"mT_lepHadTopH",
		"HadTop_pt","HadTop_eta",
		"dr_lep_HadTop",
		"dr_HadTop_tau_OS","dr_HadTop_tau_SS",
		"nJet" ,
		"nBJetLoose" , "nBJetMedium",
		"genWeight", "evtWeight",
		]

        if trainvar=="oldVar"  and channel=="1l_2tau" and bdtType=="evtLevelTT_TTH" and all==False :return [
		"htmiss",
		'mTauTauVis',
		"dr_taus",
		'avg_dr_jet',
		"nJet",
		"nBJetLoose",
		"tau1_pt",
		"tau2_pt"
		]

	if trainvar=="oldVarHTT"  and channel=="1l_2tau"  and bdtType=="evtLevelTT_TTH" and all==False :return [
	"htmiss",
	'mTauTauVis',
	"dr_taus",
	'avg_dr_jet',
	"nJet",
	"nBJetLoose",
	"tau1_pt",
	"tau2_pt",
	'mvaOutput_hadTopTaggerWithKinFit',
	'HadTop_pt'
	]

	if trainvar=="noHTT" and channel=="1l_2tau"  and bdtType=="evtLevelTTV_TTH" and all==False :return [
		'avg_dr_jet',
		'dr_taus',
		'ptmiss',
		'lep_conePt',
		'mT_lep',
		'mTauTauVis',
		'mindr_lep_jet',
		'mindr_tau1_jet',
		'dr_lep_tau_ss',
		"dr_lep_tau_sublead",
		"costS_tau",
		"tau1_pt",
		"tau2_pt"
		]

	if trainvar=="HTTMVAonlyWithKinFit" and channel=="1l_2tau"  and bdtType=="evtLevelTTV_TTH" and all==False :return [
		"lep_conePt", #"lep_eta", #"lep_tth_mva",
		"mindr_lep_jet", #"mindr_tau1_jet",
		"mindr_tau2_jet",
		"avg_dr_jet", #"ptmiss",
		"mT_lep", #"htmiss", #"tau1_mva", "tau2_mva",
		"tau2_pt",
		"dr_taus", #"dr_lep_tau_os",
		"dr_lep_tau_ss", #"dr_lep_tau_lead", #"dr_lep_tau_sublead",
		"mTauTauVis",
		"dr_HadTop_tau_lead", #"dr_HadTop_tau_sublead",
		"mass_HadTop_lepton", #"costS_HadTop_tautau",
		"mvaOutput_hadTopTaggerWithKinFit" #"mvaOutput_hadTopTagger",
		]

	if trainvar=="HTTMVAonlyWithKinFitLepID" and channel=="1l_2tau"  and bdtType=="evtLevelTTV_TTH" and all==False :return [
		"lep_conePt", #"lep_eta",
		"lep_tth_mva",
		"mindr_lep_jet", #"mindr_tau1_jet",
		"mindr_tau2_jet",
		"avg_dr_jet", #"ptmiss",
		"mT_lep", #"htmiss", #
		"tau1_mva", "tau2_mva",
		"tau2_pt",
		"dr_taus", #"dr_lep_tau_os",
		"dr_lep_tau_ss", #"dr_lep_tau_lead", #"dr_lep_tau_sublead",
		"mTauTauVis",
		"dr_HadTop_tau_lead", #"dr_HadTop_tau_sublead",
		"mvaOutput_hadTopTaggerWithKinFit" #"mvaOutput_hadTopTagger",
		]

        if trainvar=="oldVar"  and channel=="1l_2tau" and bdtType=="evtLevelTTV_TTH" and all==False :return [
		"htmiss",
		'mTauTauVis',
		"dr_taus",
		'avg_dr_jet',
		"nJet",
		"nBJetLoose",
		"tau1_pt",
		"tau2_pt"
		]

	if trainvar=="noHTT" and channel=="1l_2tau"  and bdtType=="evtLevelTT_TTH" and all==False :return [
		'avg_dr_jet',
		'dr_taus',
		'ptmiss',
		'mT_lep',
		"nJet",
		'mTauTauVis',
		'mindr_lep_jet',
		'mindr_tau1_jet',
		'mindr_tau2_jet',
		"dr_lep_tau_lead",
		"costS_tau",
		'nBJetLoose',
		"tau1_pt",
		"tau2_pt"
		]

	if trainvar=="HTTMVAonlyWithKinFit" and channel=="1l_2tau"  and bdtType=="evtLevelTT_TTH" and all==False :return [
				'avg_dr_jet',
				'dr_taus',
				'ptmiss',
				'lep_conePt',
				'mT_lep',
				'mTauTauVis',
				'mindr_lep_jet',
				'mindr_tau1_jet',
				'nJet',
				'dr_lep_tau_ss',
				"dr_lep_tau_lead",
				"costS_tau",
				'mvaOutput_hadTopTaggerWithKinFit',
				"mT_lepHadTopH"
		]

	if trainvar=="HTTMVAonlyNoKinFitLepID" and channel=="1l_2tau"  and bdtType=="evtLevelTT_TTH" and all==False :return [
				'avg_dr_jet',
				'htmiss',
				'mT_lep',
				'mTauTauVis',
				'mindr_lep_jet',
				'mindr_tau1_jet',
				'nJet',
				'dr_lep_tau_ss',
				"costS_tau",
				'mvaOutput_hadTopTaggerWithKinFit',
				'lep_tth_mva',
				'tau1_mva',
				'tau2_mva',
				"tau1_pt",
				"tau2_pt",
		]

	if trainvar=="HTTMVAonlyWithKinFit" and channel=="1l_2tau"  and bdtType=="evtLevelTT_TTH"  and all==False :return [
				'avg_dr_jet',
				'dr_taus',
				'htmiss',
				'mT_lep',
				'mTauTauVis',
				'mindr_lep_jet',
				'mindr_tau1_jet',
				'nJet',
				'dr_lep_tau_ss',
				"dr_lep_tau_lead",
				"costS_tau",
				'mvaOutput_hadTopTaggerWithKinFit',
				'nBJetLoose',
				"tau1_pt",
				"tau2_pt"
		]

	if trainvar=="HTT" and channel=="1l_2tau"  and bdtType=="evtLevelTTV_TTH" and all==False :return [
						'avg_dr_jet',
						'dr_taus',
						'ptmiss',
						'lep_conePt',
						'mT_lep',
						'mTauTauVis',
						'mindr_lep_jet',
						'mindr_tau1_jet',
						'dr_lep_tau_ss',
						"dr_lep_tau_sublead",
						"costS_tau",
						"tau1_pt",
						"tau2_pt",
						'mvaOutput_hadTopTaggerWithKinFit',
		]

	if trainvar=="HTT" and channel=="1l_2tau"  and bdtType=="evtLevelTT_TTH" and all==False :return [
						'avg_dr_jet',
						'dr_taus',
						'ptmiss',
						'mT_lep',
						"nJet",
						'mTauTauVis',
						'mindr_lep_jet',
						'mindr_tau1_jet',
						'mindr_tau2_jet',
						"dr_lep_tau_lead",
						"costS_tau",
						'nBJetLoose',
						"tau1_pt",
						"tau2_pt",
						'mvaOutput_hadTopTaggerWithKinFit',
						'HadTop_pt',
						"mvaOutput_Hj_tagger",
		]

	if trainvar=="HTT" and channel=="1l_2tau"  and 'evtLevelSUM_TTH' in bdtType and all==False :return [
						'avg_dr_jet',
						'dr_taus',
						'ptmiss',
						'lep_conePt',
						'mT_lep',
						'mTauTauVis',
						'mindr_lep_jet',
						'mindr_tau1_jet',
						'mindr_tau2_jet',
						'dr_lep_tau_ss',
						"dr_lep_tau_lead",
						"costS_tau",
						'nBJetLoose',
						"tau1_pt",
						"tau2_pt",
						'mvaOutput_hadTopTaggerWithKinFit',
						'HadTop_pt',
	]

	if trainvar=="noHTT" and channel=="1l_2tau" and 'evtLevelSUM_TTH' in bdtType and all==False :return [
						'avg_dr_jet',
						'dr_taus',
						'ptmiss',
						'lep_conePt',
						'mT_lep',
						'mTauTauVis',
						'mindr_lep_jet',
						'mindr_tau1_jet',
						'mindr_tau2_jet',
						'nJet',
						'dr_lep_tau_ss',
						"dr_lep_tau_lead",
						"costS_tau",
						'nBJetLoose',
						"tau1_pt",
						"tau2_pt",
	]

        if channel=="3l_1tau" and all==True : return [
		"lep1_pt", "lep1_eta", #"lep1_conePt", "lep1_tth_mva", "mindr_lep1_jet", "mT_lep1", "dr_lep1_tau",
		"lep2_pt", "lep2_eta", #"lep2_conePt", "lep2_tth_mva", "mindr_lep2_jet", "mT_lep2", "dr_lep2_tau",
		"lep3_pt", "lep3_eta", #"lep3_conePt", "lep3_tth_mva", "mindr_lep3_jet", "mT_lep3", "dr_lep3_tau",
		#"mindr_tau_jet", "avg_dr_jet", "ptmiss",  "htmiss", "tau_mva",
		"tau_pt", "tau_eta", "dr_leps","max_lep_eta",
		"mTauTauVis1", "mTauTauVis2",
		"avr_lep_eta",  #"dr_leps",
		#"lumiScale", "genWeight", "evtWeight",
		#"lep1_genLepPt", "lep2_genLepPt", "lep3_genLepPt", "tau_genTauPt",
		#"lep1_fake_prob", "lep2_fake_prob", "lep3_fake_prob", "tau_fake_prob",
		#"tau_fake_prob_test", "weight_fakeRate",
		#"lep1_frWeight", "lep2_frWeight",  "lep3_frWeight",  "tau_frWeight",
		#"mvaOutput_3l_ttV", "mvaOutput_3l_ttbar", "mvaDiscr_3l",
		"mbb_loose","mbb_medium",
		#"dr_tau_los1", "dr_tau_los2",  "dr_tau_lss",
		"dr_lss", "dr_los1", "dr_los2"
		]

        if trainvar=="noHTT" and channel=="3l_1tau"  and bdtType=="evtLevelTTV_TTH" and all==False :return [
			"lep1_conePt", "lep2_conePt", #"lep1_eta",  "lep2_eta", #"lep1_tth_mva",
			"mindr_lep1_jet",  #"dr_lep1_tau",
			"mindr_lep2_jet", "mT_lep2", "mT_lep1", "max_lep_eta", #"dr_lep2_tau",
			"avg_dr_jet", "ptmiss",  #"htmiss", "tau_mva",
			"tau_pt", #"tau_eta",
			"dr_leps",
			"mTauTauVis1", "mTauTauVis2",
			]

        if trainvar=="noHTT" and channel=="3l_1tau"  and bdtType=="evtLevelTT_TTH" and all==False :return [
			"mindr_lep1_jet",  #"dr_lep1_tau",
			"mindr_lep2_jet", "mT_lep2", "mT_lep1", "max_lep_eta", #"dr_lep2_tau",
			"lep3_conePt",
			"mindr_lep3_jet", #"mT_lep3", #"dr_lep3_tau",
			"mindr_tau_jet",
			"avg_dr_jet", "ptmiss",  #"htmiss", "tau_mva",
			"tau_pt", #"tau_eta",
			"dr_leps",
			"mTauTauVis1", "mTauTauVis2",
			"mbb_loose", #"mbb_medium", #"dr_tau_los1", "dr_tau_los2",
			]

        if trainvar=="noHTT" and channel=="3l_1tau"  and "evtLevelSUM_TTH" in bdtType and all==False :return [
			"lep1_conePt", "lep2_conePt", #"lep1_eta",  "lep2_eta", #"lep1_tth_mva",
			"mindr_lep1_jet",  #"dr_lep1_tau",
			"max_lep_eta", #"dr_lep2_tau",
			"mindr_tau_jet",
			"ptmiss",  #"htmiss", "tau_mva",
			"tau_pt", #"tau_eta",
			"dr_leps",
			"mTauTauVis1", "mTauTauVis2",
			"mbb_loose", #"mbb_medium", #"dr_tau_los1", "dr_tau_los2",
			"nJet", #"nBJetLoose", #"nBJetMedium",
			]

####################################################################################################
## Load data
data=load_data(inputPath,channelInTree,trainVars(True),[],testtruth,bdtType)
dataTight=load_data(inputPathTight,channelInTreeTight,trainVars(True),[],testtruth,bdtType)
doFS=True
if doFS : dataTightFS=load_data_fullsim(inputPathTightFS,channelInTreeFS,trainVars(True),[],testtruth,"all")
if doFS2 : dataTightFS2=load_data_fullsim(inputPathTightFS2,channelInTreeFS2,trainVars(True),[],testtruth,"all")
weights="totalWeight"
target='target'

if channel=="1l_2tau" or channel=="2lss_1tau":
	nSthuth = len(data.ix[(data.target.values == 0) & (data[testtruth].values==1)])
	nBtruth = len(data.ix[(data.target.values == 1) & (data[testtruth].values==1)])
	print "truth:              ", nSthuth, nBtruth
	print ("truth", data.loc[(data[testtruth]==0) & (data[testtruth]==1)][weights].sum() , data.loc[(data[target]==1) & (data[testtruth]==1)][weights].sum() )
#################################################################################
## Balance datasets
#https://stackoverflow.com/questions/34803670/pandas-conditional-multiplication
data.loc[data['target']==0, ['totalWeight']] *= 100000/data.loc[data['target']==0]["totalWeight"].sum()
data.loc[data['target']==1, ['totalWeight']] *= 100000/data.loc[data['target']==1]["totalWeight"].sum()

print ("norm", data.loc[data[target]==0][weights].sum(),data.loc[data[target]==1][weights].sum())
### TT-sample is usually much more than fakes
if channel=="2l_2tau" and 'evtLevelSUM_TTH' in bdtType :
	fastsimTT=4.72
	fastsimTTtight=1.45
	fastsimTTV=6.02
	fastsimTTVtight=4.42
	# balance backgrounds
	if bdtType=="evtLevelSUM_TTH_M" :
		TTdatacard=16.82
		TTVdatacard=4.42
		TTfullsim=0.64
		TTVfullsim=1.4
	if bdtType=="evtLevelSUM_TTH_T" :
		TTdatacard=6.27
		TTVdatacard=1.12
		TTfullsim=0.22
		TTVfullsim=1.13
	if bdtType=="evtLevelSUM_TTH_VT" :
		TTdatacard=0.56
		TTVdatacard=0.83
		#TTfullsim=0.073 -- not sure what happens with VT sample
		#TTVfullsim=0.83
		TTfullsim=0.22
		TTVfullsim=1.13
if channel=="2lss_1tau" and 'evtLevelSUM_TTH' in bdtType  :
	# VTightTau
	fastsimTT=22.14+20.22
	fastsimTTtight=1.06791+1.17204
	fastsimTTV=21.73+13.27
	fastsimTTVtight=17.2712+18.3258
	if bdtType=="evtLevelSUM_TTH_M" :
		TTdatacard=9.82
		TTVdatacard=7.75+10.49
		TTfullsim=1.53
		TTVfullsim=7.56+9.39
	if bdtType=="evtLevelSUM_TTH_T" :
		TTdatacard=7.52
		TTVdatacard=5.95+9.19
		TTfullsim=1.21
		TTVfullsim=5.81+8.30
if channel=="1l_2tau" and 'evtLevelSUM_TTH' in bdtType :
	fastsimTT=490.128+774.698
	fastsimTTV=23.5482+5.0938
	fastsimTTtight=490.128+774.698
	fastsimTTVtight=23.5482+5.0938
	if "_T" in bdtType:
		TTdatacard=192.06
		TTVdatacard=0.52+6.11
		TTfullsim=223.00
		TTVfullsim=1.46+7.03
	if "_VT" in bdtType:
		TTdatacard=91.10
		TTVdatacard=0.39+4.68
		TTfullsim=223.00
		TTVfullsim=1.46+7.03
if channel=="3l_1tau" and 'evtLevelSUM_TTH' in bdtType :
	fastsimTT=15.99+0.11
	fastsimTTtight=0.04
	fastsimTTV=21.23+6.24
	fastsimTTVtight=6.0
	# balance backgrounds
	if bdtType=="evtLevelSUM_TTH_M" :
		TTdatacard=1.08396
		TTVdatacard=0.259286+3.5813
		TTfullsim=0.59
		TTVfullsim=0.24+2.48
	if bdtType=="evtLevelSUM_TTH_T" :
		TTdatacard=0.60
		TTVdatacard=0.15+2.90
		TTfullsim=0.59
		TTVfullsim=0.24+2.48
	if bdtType=="evtLevelSUM_TTH_VT" :
		TTdatacard=0.42
		TTVdatacard=0.08+2.36
		TTfullsim=0.59
		TTVfullsim=0.24+2.48
else :
	TTdatacard=1.0
	TTVdatacard=1.0
	TTfullsim=1.0
	TTVfullsim=1.0
	fastsimTT=1.0
	fastsimTTtight=1.0
	fastsimTTV=1.0
	fastsimTTVtight=1.0
data.loc[(data['key']=='TTTo2L2Nu') | (data['key']=='TTToSemilepton'), [weights]]*=TTdatacard/fastsimTT
data.loc[(data['key']=='TTWJetsToLNu') | (data['key']=='TTZToLLNuNu'), [weights]]*=TTVdatacard/fastsimTTV
dataTight.loc[(dataTight['key']=='TTTo2L2Nu') | (dataTight['key']=='TTToSemilepton'), [weights]]*=TTdatacard/fastsimTTtight
dataTight.loc[(dataTight['key']=='TTWJetsToLNu') | (dataTight['key']=='TTZToLLNuNu'), [weights]]*=TTVdatacard/fastsimTTVtight
if doFS :
	dataTightFS.loc[(dataTightFS['proces']=='TT'), [weights]]*=TTdatacard/TTfullsim
	dataTightFS.loc[(dataTightFS['proces']=='TTW') | (dataTightFS['proces']=='TTZ'), [weights]]*=TTVdatacard/TTVfullsim
data.loc[data[target]==0, [weights]] *= 100000/data.loc[data[target]==0][weights].sum()
data.loc[data[target]==1, [weights]] *= 100000/data.loc[data[target]==1][weights].sum()
print data.columns.values.tolist()

# drop events with NaN weights - for safety
#data.replace(to_replace=np.inf, value=np.NaN, inplace=True)
#data.replace(to_replace=np.inf, value=np.zeros, inplace=True)
#data = data.apply(lambda x: pandas.to_numeric(x,errors='ignore'))
data.dropna(subset=[weights],inplace = True) # data
data.fillna(0)

nS = len(data.loc[data.target.values == 0])
nB = len(data.loc[data.target.values == 1])
print "length of sig, bkg without NaN: ", nS, nB

#################################################################################
### Plot histograms of training variables
nbins=8
colorFast='g'
colorFastT='b'
colorFull='r'
hist_params = {'normed': True, 'histtype': 'bar', 'fill': False , 'lw':5}
#plt.figure(figsize=(60, 60))
if 'evtLevelSUM_TTH' in bdtType : labelBKG = "tt+ttV"
if bdtType=='evtLevelTT_TTH' : labelBKG = "tt"
if bdtType=='evtLevelTTV_TTH' : labelBKG = "ttV"
printmin=True
plotResiduals=False
plotAll=False
BDTvariables=trainVars(plotAll)
make_plots(BDTvariables,nbins,
    data.ix[data.target.values == 0],labelBKG, colorFast,
    data.ix[data.target.values == 1],'Signal', colorFastT,
    channel+"/"+bdtType+"_"+trainvar+"_Variables_BDT_fastsim"+FastsimWP+".pdf",
    printmin,
	plotResiduals
    )

### Plot aditional histograms
if 1<0 : # channel=="1l_2tau" :
	BDTvariables=['mvaOutput_hadTopTaggerWithKinFit',
	"mvaOutput_Hj_tagger", #"memOutput_LR",
	'mvaOutput_Hjj_tagger']
	make_plots(BDTvariables,nbins,
    data.ix[data.target.values == 0],labelBKG, colorFast,
    data.ix[data.target.values == 1],'Signal', colorFastT,
    channel+"/"+bdtType+"_"+trainvar+"_BDTVariables_fastsim"+FastsimWP+".pdf",
    printmin
    )

if doFS and doPlots :
	plotResiduals=False
	make_plots(BDTvariables,nbins,
	    data.ix[data.target.values == 1],              "Fast "+FastsimWP, colorFast,
	    dataTightFS.ix[dataTightFS.target.values == 1],'Full '+FullsimWP, colorFastT,
	    channel+"/"+bdtType+"_"+trainvar+"_Variables_signal_fastsim"+FastsimWP+"_fullsim"+FullsimWP+".pdf",
	    printmin,
		plotResiduals
	    )

	make_plots(BDTvariables,nbins,
	    data.ix[data.target.values == 0],              "Fast "+labelBKG+" "+FastsimWP, colorFast,
	    dataTightFS.ix[dataTightFS.target.values == 0],'Full '+labelBKG+" "+FullsimWP, colorFastT,
	    channel+"/"+bdtType+"_"+trainvar+"_Variables_BKG_fastsim"+FastsimWP+"_fullsim"+FullsimWP+".pdf",
	    printmin,
		plotResiduals
	    )

	make_plots(BDTvariables,nbins,
	    dataTightFS.ix[dataTightFS.target.values == 1],"Fullsim signal", colorFast,
	    dataTightFS.ix[dataTightFS.target.values == 0],"Fullsim "+labelBKG, colorFastT,
	    channel+"/"+bdtType+"_"+trainvar+"_Variables_fullsim"+FullsimWP+".pdf",
	    printmin,
		plotResiduals
	    )

	make_plots(BDTvariables,nbins,
	    dataTight.ix[dataTight.target.values == 1],    "Fast "+FastsimTWP, colorFast,
	    dataTightFS.ix[dataTightFS.target.values == 1],'Full '+FullsimWP, colorFastT,
	    channel+"/"+bdtType+"_"+trainvar+"_Variables_signal_fastsim"+FastsimTWP+"_fullsim"+FullsimWP+".pdf",
	    printmin,
		plotResiduals
	    )

	make_plots(BDTvariables,nbins,
	    dataTight.ix[dataTight.target.values == 0],    "Fast "+labelBKG+" "+FastsimTWP, colorFast,
	    dataTightFS.ix[dataTightFS.target.values == 0],'Full '+labelBKG+" "+FullsimWP, colorFastT,
	    channel+"/"+bdtType+"_"+trainvar+"_Variables_BKG_fastsim"+FastsimTWP+"_fullsim"+FullsimWP+".pdf",
	    printmin,
		plotResiduals
	    )

if doFS2 and doPlots :
	make_plots(BDTvariables,nbins,
	    dataTightFS2.ix[dataTightFS2.target.values == 1], 'Full '+FullsimWP2, colorFast,
	    dataTightFS.ix[dataTightFS.target.values == 1],   'Full '+FullsimWP, colorFastT,
	    channel+"/"+bdtType+"_"+trainvar+"_Variables_signal_fullsim"+FullsimWP2+"_"+FullsimWP+".pdf",
	    printmin,
		plotResiduals
	    )

	make_plots(BDTvariables,nbins,
	    dataTightFS2.ix[dataTightFS2.target.values == 0], 'Full '+labelBKG+" "+FullsimWP2, colorFast,
	    dataTightFS.ix[dataTightFS.target.values == 0],   'Full '+labelBKG+" "+FullsimWP, colorFastT,
	    channel+"/"+bdtType+"_"+trainvar+"_Variables_BKG_fullsim"+FullsimWP2+"_"+FullsimWP+".pdf",
	    printmin,
		plotResiduals
	    )

	make_plots(BDTvariables,nbins,
	    dataTightFS2.ix[dataTightFS2.target.values == 1], 'Full '+FullsimWP2, colorFast,
	    data.ix[data.target.values == 1],                 'Fast '+FastsimWP, colorFastT,
	    channel+"/"+bdtType+"_"+trainvar+"_Variables_signal_fastsim"+FastsimWP+"_fullsim"+FullsimWP2+".pdf",
	    printmin,
		plotResiduals
	    )

	make_plots(BDTvariables,nbins,
	    dataTightFS2.ix[dataTightFS2.target.values == 0], 'Full '+labelBKG+" "+FullsimWP2, colorFast,
	    data.ix[data.target.values == 0],                 'Fast '+labelBKG+" "+FastsimWP, colorFastT,
	    channel+"/"+bdtType+"_"+trainvar+"_Variables_BKG_fastsim"+FastsimWP+"_fullsim"+FullsimWP2+".pdf",
	    printmin,
		plotResiduals
	    )

if doPlots :
	make_plots(BDTvariables,nbins,
	    data.ix[data.target.values == 0],          "Fast "+labelBKG+" "+FastsimWP, colorFast,
	    dataTight.ix[dataTight.target.values == 0],"Fast "+labelBKG+" "+FastsimTWP, colorFastT,
	    channel+"/"+bdtType+"_"+trainvar+"_Variables_BKG_fastsim"+FastsimWP+"_"+FastsimTWP+".pdf",
	    printmin,
		plotResiduals
	    )

	make_plots(BDTvariables,nbins,
	    data.ix[data.target.values == 1],          "Fast "+FastsimWP, colorFast,
	    dataTight.ix[dataTight.target.values == 1],"Fast "+FastsimTWP, colorFastT,
	    channel+"/"+bdtType+"_"+trainvar+"_Variables_signal_fastsim"+FastsimWP+"_"+FastsimTWP+".pdf",
	    printmin,
		plotResiduals
	    )

###################################################################
if channel=="2lss_1tau" : njet="nJet25_Recl"
else : njet="nJet"
if 0>1 :
	totestcorr=['mvaOutput_hadTopTaggerWithKinFit',
	"mvaOutput_Hj_tagger",
	'mvaOutput_Hjj_tagger',] #]
	totestcorrNames=['HTT',
	"Hj_tagger",
	'Hjj_tagger',njet]
	for ii in [1,2] :
		if ii == 1 :
			datad=data.loc[data[target].values == 1]
			label="signal"
		else :
			datad=data.loc[data[target].values == 0]
			label="BKG"
		datacorr = datad[totestcorr] #.loc[:,trainVars(False)] #dataHToNobbCSV[[trainVars(True)]]
		correlations = datacorr.corr()
		fig = plt.figure(figsize=(5, 5))
		ax = fig.add_subplot(111)
		cax = ax.matshow(correlations, vmin=-1, vmax=1)
		ticks = np.arange(0,len(totestcorr),1)
		plt.rc('axes', labelsize=8)
		ax.set_xticks(ticks)
		ax.set_yticks(ticks)

		ax.set_xticklabels(totestcorrNames,rotation=-90)
		ax.set_yticklabels(totestcorrNames)
		fig.colorbar(cax)
		fig.tight_layout()
		#plt.subplots_adjust(left=0.9, right=0.9, top=0.9, bottom=0.1)
		plt.savefig("{}/{}_{}_{}_corrBDTs_{}.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False))),label))
		ax.clear()
#########################################################################################
traindataset, valdataset  = train_test_split(data[trainVars(False)+["target","totalWeight"]], test_size=0.2, random_state=7)
## to GridSearchCV the test_size should not be smaller than 0.4 == it is used for cross validation!
## to final BDT fit test_size can go down to 0.1 without sign of overtraining
#############################################################################################
## Training parameters
if options.HypOpt==True :
	# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
	param_grid = {
    			'n_estimators': [200,500,800, 1000,2500],
    			'min_child_weight': [1,100],
    			'max_depth': [1,2,3,4],
    			'learning_rate': [0.01,0.02,0.03]
				}
	scoring = "roc_auc"
	early_stopping_rounds = 200 # Will train until validation_0-auc hasn't improved in 100 rounds.
	cv=3
	cls = xgb.XGBClassifier()
	fit_params = { "eval_set" : [(valdataset[trainVars(False)].values,valdataset[target])],
                           "eval_metric" : "auc",
                           "early_stopping_rounds" : early_stopping_rounds,
						   'sample_weight': valdataset[weights].values }
	gs = GridSearchCV(cls, param_grid, scoring, fit_params, cv = cv, verbose = 0)
	gs.fit(traindataset[trainVars(False)].values,
	traindataset.target.astype(np.bool)
	)
	for i, param in enumerate(gs.cv_results_["params"]):
		print("params : {} \n    cv auc = {}  +- {} ".format(param,gs.cv_results_["mean_test_score"][i],gs.cv_results_["std_test_score"][i]))
	print("best parameters",gs.best_params_)
	print("best score",gs.best_score_)
	#print("best iteration",gs.best_iteration_)
	#print("best ntree limit",gs.best_ntree_limit_)
	file = open("{}/{}_{}_{}_GSCV.log".format(channel,bdtType,trainvar,str(len(trainVars(False)))),"w")
	file.write(
				str(trainVars(False))+"\n"+
				"best parameters"+str(gs.best_params_) + "\n"+
				"best score"+str(gs.best_score_)+ "\n"
				#"best iteration"+str(gs.best_iteration_)+ "\n"+
				#"best ntree limit"+str(gs.best_ntree_limit_)
				)
	for i, param in enumerate(gs.cv_results_["params"]):
		file.write("params : {} \n    cv auc = {}  +- {} {}".format(param,gs.cv_results_["mean_test_score"][i],gs.cv_results_["std_test_score"][i]," \n"))
	file.close()

cls = xgb.XGBClassifier(
			n_estimators = options.ntrees,
			max_depth = options.treeDeph,
			min_child_weight = options.mcw, # min_samples_leaf
			learning_rate = options.lr,
			#max_features = 'sqrt',
			#min_samples_leaf = 100
			#objective='binary:logistic', #booster='gbtree',
			#gamma=0, #min_child_weight=1,
			#max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, #random_state=0
			)
cls.fit(
	traindataset[trainVars(False)].values,
	traindataset.target.astype(np.bool),
	sample_weight=(traindataset[weights].astype(np.float64))
	# more diagnosis, in case
	#eval_set=[(traindataset[trainVars(False)].values,  traindataset.target.astype(np.bool),traindataset[weights].astype(np.float64)),
	#(valdataset[trainVars(False)].values,  valdataset.target.astype(np.bool), valdataset[weights].astype(np.float64))] ,
	#verbose=True,eval_metric="auc"
	)
print trainVars(False)
print traindataset[trainVars(False)].columns.values.tolist()
print ("XGBoost trained")
proba = cls.predict_proba(traindataset[trainVars(False)].values )
fpr, tpr, thresholds = roc_curve(traindataset[target], proba[:,1],
	sample_weight=(traindataset[weights].astype(np.float64)) )
train_auc = auc(fpr, tpr, reorder = True)
print("XGBoost train set auc - {}".format(train_auc))
proba = cls.predict_proba(valdataset[trainVars(False)].values )
fprt, tprt, thresholds = roc_curve(valdataset[target], proba[:,1], sample_weight=(valdataset[weights].astype(np.float64))  )
test_auct = auc(fprt, tprt, reorder = True)
print("XGBoost test set auc - {}".format(test_auct))
#print list(traindataset[trainVars(False)])
#print list(dataTight[trainVars(False)])
#print list(dataTightFS[trainVars(False)])
#print dataTight[target]
proba = cls.predict_proba(dataTight[trainVars(False)].values )
fprtight, tprtight, thresholds = roc_curve(dataTight[target], proba[:,1], sample_weight=(dataTight[weights].astype(np.float64))  )
test_auctight = auc(fprtight, tprtight, reorder = True)
print("XGBoost test set auc - tight lep ID - {}".format(test_auctight))
if doFS :
	proba = cls.predict_proba(dataTightFS[trainVars(False)].values)
	fprtightF, tprtightF, thresholds = roc_curve(dataTightFS[target], proba[:,1], sample_weight=(dataTightFS[weights].astype(np.float64)) )
	test_auctightF = auc(fprtightF, tprtightF, reorder = True)
	print("XGBoost test set auc - fullsim all - {}".format(test_auctightF))
	if "evtLevelSUM_TTH" in bdtType :
		tightTT=dataTightFS.ix[(dataTightFS.proces.values=='TTZ') | (dataTightFS.proces.values=='TTW') | (dataTightFS.proces.values=='TT') | (dataTightFS.proces.values=='signal')]
	if bdtType=="evtLevelTT_TTH" :
		tightTT=dataTightFS.ix[(dataTightFS.proces.values=='TT') | (dataTightFS.proces.values=='signal')]
	if bdtType=="evtLevelTTV_TTH" :
		tightTT=dataTightFS.ix[(dataTightFS.proces.values=='TTZ') | (dataTightFS.proces.values=='TTW') | (dataTightFS.proces.values=='signal')]
	proba = cls.predict_proba(tightTT[trainVars(False)].values)
	fprtightFI, tprtightFI, thresholds = roc_curve(tightTT[target].values, proba[:,1], sample_weight=(tightTT[weights].astype(np.float64)))
	test_auctightFI = auc(fprtightFI, tprtightFI, reorder = True)
	print("XGBoost test set auc - fullsim individual - {}".format(test_auctightFI))

pklpath=channel+"/"+channel+"_XGB_"+trainvar+"_"+bdtType+"_"+str(len(trainVars(False)))+"Var"
print ("Done  ",pklpath,hyppar)
if options.doXML==True :
	print ("Date: ", time.asctime( time.localtime(time.time()) ))
	pickle.dump(cls, open(pklpath+".pkl", 'wb'))
	file = open(pklpath+"_pkl.log","w")
	file.write(str(trainVars(False))+"\n")
	file.close()
	print ("saved ",pklpath+".pkl")
	print ("variables are: ",pklpath+"_pkl.log")
	# save the model in file 'xgb.model.dump'
	#model = cls.booster().get_dump(fmap='', with_stats=False) #.get_dump() #pickle.dumps(cls)
	#xmlfile=channel+"/"+channel+"_XGB_"+trainvar+"_"+bdtType+".xml"
##################################################
fig, ax = plt.subplots(figsize=(6, 6))
## ROC curve
#ax.plot(fprf, tprf, lw=1, label='GB train (area = %0.3f)'%(train_aucf))
#ax.plot(fprtf, tprtf, lw=1, label='GB test (area = %0.3f)'%(test_auctf))
ax.plot(fpr, tpr, lw=1, label='XGB train (area = %0.3f)'%(train_auc))
ax.plot(fprt, tprt, lw=1, label='XGB test (area = %0.3f)'%(test_auct))
#ax.plot(fprc, tprc, lw=1, label='CB train (area = %0.3f)'%(train_aucc))
#ax.plot(fprtight, tprtight, lw=1, label='XGB test - tight ID (area = %0.3f)'%(test_auctight))
if doFS : ax.plot(fprtightFI, tprtightFI, lw=1, label='XGB test - Fullsim (area = %0.3f)'%(test_auctightFI))
#ax.plot(fprtightF, tprtightF, lw=1, label='XGB test - Fullsim All (area = %0.3f)'%(test_auctightF))
ax.set_ylim([0.0,1.0])
ax.set_xlim([0.0,1.0])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc="lower right")
ax.grid()
fig.savefig("{}/{}_{}_{}_{}_roc.png".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar))
fig.savefig("{}/{}_{}_{}_{}_roc.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar))
###########################################################################
## feature importance plot
fig, ax = plt.subplots()
f_score_dict =cls.booster().get_fscore()
f_score_dict = {trainVars(False)[int(k[1:])] : v for k,v in f_score_dict.items()}
feat_imp = pandas.Series(f_score_dict).sort_values(ascending=True)
feat_imp.plot(kind='barh', title='Feature Importances')
fig.tight_layout()
fig.savefig("{}/{}_{}_{}_{}_XGB_importance.png".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar))
fig.savefig("{}/{}_{}_{}_{}_XGB_importance.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar))
###########################################################################
#print (list(valdataset))
hist_params = {'normed': True, 'bins': 10 , 'histtype':'step'}
plt.clf()
y_pred = cls.predict_proba(valdataset.ix[valdataset.target.values == 0, trainVars(False)].values)[:, 1] #
y_predS = cls.predict_proba(valdataset.ix[valdataset.target.values == 1, trainVars(False)].values)[:, 1] #
plt.figure('XGB',figsize=(6, 6))
values, bins, _ = plt.hist(y_pred , label="TT (XGB)", **hist_params)
values, bins, _ = plt.hist(y_predS , label="signal", **hist_params )
#plt.xscale('log')
#plt.yscale('log')
plt.legend(loc='best')
plt.savefig(channel+'/'+bdtType+'_'+trainvar+'_'+str(len(trainVars(False)))+'_'+hyppar+'_XGBclassifier.pdf')
###########################################################################
"""
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
########################################################################
# plot correlation matrix
if options.HypOpt==False :
	for ii in [1,2] :
		if ii == 1 :
			datad=traindataset.loc[traindataset[target].values == 1]
			label="signal"
		else :
			datad=traindataset.loc[traindataset[target].values == 0]
			label="BKG"
		datacorr = datad[trainVars(False)] #.loc[:,trainVars(False)] #dataHToNobbCSV[[trainVars(True)]]
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
		#plt.subplots_adjust(left=0.9, right=0.9, top=0.9, bottom=0.1)
		plt.savefig("{}/{}_{}_{}_corr_{}.png".format(channel,bdtType,trainvar,str(len(trainVars(False))),label))
		plt.savefig("{}/{}_{}_{}_corr_{}.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False))),label))
		ax.clear()
	###################################################################
	if 0>1 :
		for ii in [1,2] :
			if ii == 1 :
				datad=dataTightFS.loc[dataTightFS[target].values == 1]
				label="signal"
			else :
				datad=dataTightFS.loc[dataTightFS[target].values == 0]
				label="BKG"
			datacorr = datad[trainVars(False)] #.loc[:,trainVars(False)] #dataHToNobbCSV[[trainVars(True)]]
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
			#plt.subplots_adjust(left=0.9, right=0.9, top=0.9, bottom=0.1)
			plt.savefig("{}/{}_{}_{}_corr_{}_FS.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False))),label))
			ax.clear()
process = psutil.Process(os.getpid())
print(process.memory_info().rss)
