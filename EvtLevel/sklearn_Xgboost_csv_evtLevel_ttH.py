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
parser.add_option("--oldNtuple", action="store_true", dest="oldNtuple", help="use Matthias", default=False)
parser.add_option("--ntrees ", type="int", dest="ntrees", help="hyp", default=2000)
parser.add_option("--treeDeph", type="int", dest="treeDeph", help="hyp", default=2)
parser.add_option("--lr", type="float", dest="lr", help="hyp", default=0.01)
parser.add_option("--mcw", type="int", dest="mcw", help="hyp", default=1)
(options, args) = parser.parse_args()
#""" bdtType=="evtLevelTTV_TTH"

#channel="2lss_1tau"
channel=options.channel #"1l_2tau"
if channel=='1l_2tau':
	channelInTree='1l_2tau_OS_Tight'
	inputPath='/hdfs/local/acaan/ttHAnalysis/2016/1l_2tau_2018Jan26_forBDT_tightLmediumT/histograms/1l_2tau/forBDTtraining_OS/' #  - tight lepton, loose tau || 1l_2tau_2018Jan24_forBDT_tightLlooseT || 1l_2tau_2018Jan24_forBDT_tightLmediumT || 1l_2tau_2018Jan24_forBDT_tightLisolooseT
	channelInTreeTight='1l_2tau_OS_Tight'
	inputPathTight='/hdfs/local/acaan/ttHAnalysis/2016/1l_2tau_2018Jan26_forBDT_tightLtightT/histograms/1l_2tau/forBDTtraining_OS/'
	inputPathTightFS='/hdfs/local/acaan/ttHAnalysis/2016/2018Jan28_BDT_toTrees_FS_looseT/histograms/1l_2tau/Tight_OS/'
	# 2018Jan28_BDT_toTrees_FS_looseT | 1l_2tau_2018Jan23_VHbb_tree | 2018Jan28_BDT_toTrees_FS
	# 2018Jan26_BDT_fromVHbb_toTrees_fullsimData
	criteria=[]
	testtruth="bWj1Wj2_isGenMatchedWithKinFit"
	FullsimWP="Medium"
	FastsimWP="Medium"

if channel=='2lss_1tau':
	channelInTree='2lss_1tau_lepSS_sumOS_Loose'
	inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Dec30-VHbb-wMEM-LooseLepMedTau/histograms/2lss_1tau/forBDTtraining_SS_OS/'
	criteria=['lep1_isTight', 'lep2_isTight','tau_isTight',"failsTightChargeCut"]
	testtruth="bWj1Wj2_isGenMatchedWithKinFit"
	channelInTreeTight='2lss_1tau_lepSS_sumOS_Tight'
	inputPathTight='/hdfs/local/acaan/ttHAnalysis/2016/2017Dec08-BDT-noMEM-tighLep/histograms/2lss_1tau/forBDTtraining_SS_OS/'
	inputPathTightFS='/hdfs/local/acaan/ttHAnalysis/2016/2018Jan_BDT_fromVHbb_toTrees_fullsimData/histograms/2lss_1tau/Tight_SS_OS'

bdtType=options.bdtType #"evtLevelTT_TTH"
trainvar=options.variables
hyppar=str(options.variables)+"_ntrees_"+str(options.ntrees)+"_deph_"+str(options.treeDeph)+"_mcw_"+str(options.mcw)+"_lr_0o0"+str(int(options.lr*100))

import shutil,subprocess
proc=subprocess.Popen(['mkdir '+options.channel],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()

def trainVars(all):

        if trainvar=="allVar" and channel=="2lss_1tau"  :return [
		'HadTop_eta', 'HadTop_pt', 'MT_met_lep1', 'avg_dr_jet',
		#'bWj1Wj2_isGenMatched', 'bWj1Wj2_isGenMatchedWithKinFit',
		'dr_lep1_tau', 'dr_lep2_tau', 'dr_leps', #'evtWeight',
		'fitHTptoHTpt', 'fittedHadTop_eta', 'fittedHadTop_pt', #'genTopPt', 'genWeight', 'hadtruth',
		'htmiss', 'lep1_conePt', 'lep1_eta', #'lep1_frWeight',
		#'lep1_genLepPt',
		'lep1_pt', 'lep2_pt',
		'lep2_conePt', 'lep2_eta', #'lep2_frWeight', 'lep2_genLepPt',
		'lep1_tth_mva','lep2_tth_mva',
		#'log_memOutput_tt', 'log_memOutput_ttH', 'log_memOutput_ttZ', 'log_memOutput_ttZ_Zll',
		#'lumiScale',
		'mT_lep1', 'mT_lep2', 'mTauTauVis1', 'mTauTauVis2', 'max_lep_eta',
		'mbb', 'ptbb', 'mbb_loose', 'ptbb_loose',
		#'memOutput_LR', 'memOutput_errorFlag', 'memOutput_isValid', 'memOutput_ttZ_LR', 'memOutput_ttZ_Zll_LR', 'memOutput_tt_LR',
		'mindr_lep1_jet',
		'mindr_lep2_jet',
		'mindr_tau_jet',
		#'mvaOutput_2lss_ttbar', 'mvaOutput_Hj_tagger', 'mvaOutput_Hjj_tagger',
		#'mvaOutput_hadTopTagger', 'mvaOutput_hadTopTaggerWithKinFit',
		'nJet25_Recl', 'ncombo', 'ptmiss', 'tau_eta', #'tau_frWeight', 'tau_genTauPt',
		'tau_mva', 'tau_pt', 'unfittedHadTop_eta', 'unfittedHadTop_pt', #'lep1_isTight', 'lep2_isTight', 'tau_isTight'
		'nBJetLoose', 'nBJetMedium', 'nJet', 'nLep'
		]

        if trainvar=="noHTT" and channel=="2lss_1tau" and bdtType=="evtLevelTT_TTH" and all==False :
			return [
			'avg_dr_jet', 'dr_lep1_tau', 'dr_lep2_tau', 'dr_leps',
			'lep1_conePt', 'lep2_conePt', 'mT_lep1', 'mT_lep2',
			'mTauTauVis1', 'mTauTauVis2',
			'max_lep_eta', 'mbb', 'mindr_lep1_jet',
			'mindr_lep2_jet', 'mindr_tau_jet', 'nJet25_Recl', 'ptmiss', 'tau_pt'
			]

        if trainvar=="HTT" and channel=="2lss_1tau" and bdtType=="evtLevelTT_TTH" and all==False :
			return [
			'dr_lep1_tau', 'dr_lep2_tau', 'dr_leps',
			'mT_lep1', 'mT_lep2', 'mTauTauVis1', 'mTauTauVis2', 'mindr_lep1_jet',
			'mindr_lep2_jet', 'mindr_tau_jet', 'ptmiss', 'tau_pt',
			'mvaOutput_hadTopTaggerWithKinFit', 'mvaOutput_Hj_tagger', 'unfittedHadTop_pt',
			'nJet25_Recl', 'avg_dr_jet'
			]

        if trainvar=="HTTMEM" and channel=="2lss_1tau" and bdtType=="evtLevelTT_TTH" and all==False :
			return [
			'dr_lep1_tau',"memOutput_LR", 'dr_lep2_tau', 'dr_leps',
			'mT_lep1', 'mT_lep2', 'mTauTauVis1', 'mTauTauVis2', 'mindr_lep1_jet',
			'mindr_lep2_jet', 'mindr_tau_jet', 'ptmiss', 'tau_pt',
			'mvaOutput_hadTopTaggerWithKinFit', 'mvaOutput_Hj_tagger', 'unfittedHadTop_pt',
			'nJet25_Recl', 'avg_dr_jet'
			]

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
		######
		#"nJet",
		#"mindr_lep1_jet",
		#"avg_dr_jet",
		#"max_lep_eta",
		#"lep2_conePt",
		#"dr_leps",
		#"tau_pt",
		#"dr_lep1_tau"
		]

        if trainvar=="oldVar"  and channel=="2lss_1tau"  and bdtType=="evtLevelTTV_TTH" and all==False :return [
		"max_lep_eta",
		"MT_met_lep1",
		"nJet25_Recl",
		"mindr_lep1_jet",
		"mindr_lep2_jet",
		"lep1_conePt",
		"lep2_conePt"
		####
		#"mindr_lep1_jet",
		#"mindr_lep2_jet",
		#"avg_dr_jet",
		#'max_lep_eta',
		#"lep1_conePt",
		#"lep2_conePt",
		#"mT_lep1",
		#"dr_leps",
		#"mTauTauVis1",
		#"mTauTauVis2"
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
			'avg_dr_jet', 'dr_lep1_tau', 'dr_lep2_tau', 'dr_leps',
			'lep1_conePt', 'lep2_conePt', 'mT_lep1', 'mT_lep2',
			'mTauTauVis1', 'mTauTauVis2', 'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet',
			'ptmiss', 'tau_pt',
			'mvaOutput_hadTopTaggerWithKinFit', 'mvaOutput_Hj_tagger'
			]

        if trainvar=="HTTMEM" and channel=="2lss_1tau" and bdtType=="evtLevelTTV_TTH" and all==False :
			return [
			'avg_dr_jet', 'dr_lep1_tau', 'dr_lep2_tau', 'dr_leps',
			'lep1_conePt', 'lep2_conePt', 'mT_lep1', 'mT_lep2',
			'mTauTauVis1', 'mTauTauVis2', 'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet',
			'ptmiss', 'tau_pt',
			'mvaOutput_hadTopTaggerWithKinFit', 'mvaOutput_Hj_tagger',"memOutput_LR"
			]

        if trainvar=="noHTT" and channel=="2lss_1tau" and bdtType=="evtLevelTTV_TTH" and all==False :
			return [
			'avg_dr_jet', 'dr_lep1_tau', 'dr_lep2_tau', 'dr_leps',
			'lep1_conePt', 'lep2_conePt',
			'mT_lep1', 'mT_lep2', 'mTauTauVis1', 'mTauTauVis2',
			'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet', 'ptmiss', 'tau_pt'
			]

	if channel=="1l_2tau" and all==True :return [
		#"lep_pt",
		"lep_conePt", #
		#"lep_eta",
		#"lep_tth_mva",
		"mindr_lep_jet", "mindr_tau1_jet", "mindr_tau2_jet",
		"avg_dr_jet", "ptmiss",
		"htmiss", "mT_lep",
		#"tau1_mva", "tau2_mva",
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
		#"lumiScale",
		"mvaOutput_hadTopTagger",
		#'mvaOutput_Hj_tagger',# "mvaOutput_Hjj_tagger",
		#"mvaOutput_hadTopTaggerWithKinFit",
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
		#'htmiss',
		'ptmiss',
		'lep_conePt',
		'mT_lep',
		'mTauTauVis',
		'mindr_lep_jet',
		'mindr_tau1_jet',
		#'mindr_tau2_jet',
		#'nJet',
		#'dr_lep_tau_os',
		'dr_lep_tau_ss',
		#"dr_lep_tau_lead",
		"dr_lep_tau_sublead",
		"costS_tau",
		#"dr_HadTop_tau_OS",
		#"dr_HadTop_tau_SS",
		#"tau1_eta",
		#"tau2_eta"
		#"mT_lepHadTop",
		#"mT_lepHadTopH",
		#'nBJetLoose',
		"tau1_pt",
		"tau2_pt"
		]

	if trainvar=="HTTMVAonlyWithKinFit" and channel=="1l_2tau"  and bdtType=="evtLevelTTV_TTH" and all==False :return [
		#"lep_pt",
		"lep_conePt", #"lep_eta", #"lep_tth_mva",
		"mindr_lep_jet", #"mindr_tau1_jet",
		"mindr_tau2_jet",
		"avg_dr_jet", #"ptmiss",
		"mT_lep", #"htmiss", #"tau1_mva", "tau2_mva",
		#"tau1_pt",
		"tau2_pt",
		#"tau1_eta", "tau2_eta",
		"dr_taus", #"dr_lep_tau_os",
		"dr_lep_tau_ss", #"dr_lep_tau_lead", #"dr_lep_tau_sublead",
		#"costS_tau",
		"mTauTauVis",
		#"lumiScale", "genWeight", "evtWeight",
		#"mT_lepHadTop" ,
		#"mT_lepHadTopH",
		#"HadTop_pt", #"HadTop_eta",
		#"dr_lep_HadTop",
		#"dr_HadTop_tau_OS", #"dr_HadTop_tau_SS",
		"dr_HadTop_tau_lead", #"dr_HadTop_tau_sublead",
		#"dr_HadTop_tautau",
		#"dr_HadTop_lepton",
		"mass_HadTop_lepton", #"costS_HadTop_tautau",
		"mvaOutput_hadTopTaggerWithKinFit" #"mvaOutput_hadTopTagger",
		]

	if trainvar=="HTTMVAonlyWithKinFitLepID" and channel=="1l_2tau"  and bdtType=="evtLevelTTV_TTH" and all==False :return [
		#"lep_pt",
		"lep_conePt", #"lep_eta",
		"lep_tth_mva",
		"mindr_lep_jet", #"mindr_tau1_jet",
		"mindr_tau2_jet",
		"avg_dr_jet", #"ptmiss",
		"mT_lep", #"htmiss", #
		"tau1_mva", "tau2_mva",
		#"tau1_pt",
		"tau2_pt",
		#"tau1_eta", "tau2_eta",
		"dr_taus", #"dr_lep_tau_os",
		"dr_lep_tau_ss", #"dr_lep_tau_lead", #"dr_lep_tau_sublead",
		#"costS_tau",
		"mTauTauVis",
		#"lumiScale", "genWeight", "evtWeight",
		#"mT_lepHadTop" ,
		#"mT_lepHadTopH",
		#"HadTop_pt", #"HadTop_eta",
		#"dr_lep_HadTop",
		#"dr_HadTop_tau_OS", #"dr_HadTop_tau_SS",
		"dr_HadTop_tau_lead", #"dr_HadTop_tau_sublead",
		#"dr_HadTop_tautau",
		#"dr_HadTop_lepton",
		#"mass_HadTop_lepton", #"costS_HadTop_tautau",
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
		#'htmiss',
		'ptmiss',
		#'lep_conePt',
		'mT_lep',
		"nJet",
		'mTauTauVis',
		'mindr_lep_jet',
		'mindr_tau1_jet',
		'mindr_tau2_jet',
		#'nJet',
		#'dr_lep_tau_os',
		#'dr_lep_tau_ss',
		"dr_lep_tau_lead",
		#"dr_lep_tau_sublead",
		"costS_tau",
		#"dr_HadTop_tau_OS",
		#"dr_HadTop_tau_SS",
		#"mT_lepHadTop",
		#"mT_lepHadTopH",
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
				#"mT_lepHadTop",
				"mT_lepHadTopH"
		]

	if trainvar=="HTTMVAonlyNoKinFitLepID" and channel=="1l_2tau"  and bdtType=="evtLevelTT_TTH" and all==False :return [
				'avg_dr_jet',
				#'dr_taus',
				'htmiss',
				#'ptmiss',
				#'lep_conePt',
				'mT_lep',
				'mTauTauVis',
				'mindr_lep_jet',
				'mindr_tau1_jet',
				'nJet',
				'dr_lep_tau_ss',
				#"dr_lep_tau_lead",
				"costS_tau",
				'mvaOutput_hadTopTaggerWithKinFit',
				#"mT_lepHadTop",
				#"mT_lepHadTopH",
				'lep_tth_mva',
				'tau1_mva',
				'tau2_mva',
				#'HadTop_pt',
				"tau1_pt",
				"tau2_pt",
				#"dr_HadTop_tau_lead"
		]

	if trainvar=="HTTMVAonlyWithKinFit" and channel=="1l_2tau"  and bdtType=="evtLevelTT_TTH"  and all==False :return [
				'avg_dr_jet',
				'dr_taus',
				#'ptmiss',
				'htmiss',
				#'lep_conePt',
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
						#'htmiss',
						'ptmiss',
						'lep_conePt',
						'mT_lep',
						'mTauTauVis',
						'mindr_lep_jet',
						'mindr_tau1_jet',
						#'mindr_tau2_jet',
						#'nJet',
						#'dr_lep_tau_os',
						'dr_lep_tau_ss',
						#"dr_lep_tau_lead",
						"dr_lep_tau_sublead",
						"costS_tau",
						#"dr_HadTop_tau_OS",
						#"dr_HadTop_tau_SS",
						#"mT_lepHadTop",
						#"mT_lepHadTopH",
						#'nBJetLoose',
						"tau1_pt",
						"tau2_pt",
						'mvaOutput_hadTopTaggerWithKinFit',
						#'HadTop_pt',
						#"dr_HadTop_tau_lead",
						#"mvaOutput_Hj_tagger",
						#'mvaOutput_Hjj_tagger',
		]

	if trainvar=="HTT" and channel=="1l_2tau"  and bdtType=="evtLevelTT_TTH" and all==False :return [
						'avg_dr_jet',
						'dr_taus',
						#'htmiss',
						'ptmiss',
						#'lep_conePt',
						'mT_lep',
						"nJet",
						'mTauTauVis',
						'mindr_lep_jet',
						'mindr_tau1_jet',
						'mindr_tau2_jet',
						#'nJet',
						#'dr_lep_tau_os',
						#'dr_lep_tau_ss',
						"dr_lep_tau_lead",
						#"dr_lep_tau_sublead",
						"costS_tau",
						#"dr_HadTop_tau_OS",
						#"dr_HadTop_tau_SS",
						#"mT_lepHadTop",
						#"mT_lepHadTopH",
						'nBJetLoose',
						"tau1_pt",
						"tau2_pt",
						'mvaOutput_hadTopTaggerWithKinFit',
						'HadTop_pt',
						#"dr_HadTop_tau_lead",
						"mvaOutput_Hj_tagger",
						#'mvaOutput_Hjj_tagger',
		]
####################################################################################################
## Load data
data=load_data(inputPath,channelInTree,trainVars(True),[],testtruth,bdtType)
dataTight=load_data(inputPathTight,channelInTreeTight,trainVars(True),[],testtruth,bdtType)
dataTightFS=load_data_fullsim(inputPathTightFS,channelInTreeTight,trainVars(True),[],testtruth,"all")

if channel=="1l_2tau" or channel=="2lss_1tau":
	nSthuth = len(data.ix[(data.target.values == 0) & (data[testtruth].values==1)])
	nBtruth = len(data.ix[(data.target.values == 1) & (data[testtruth].values==1)])
	print "truth:              ", nSthuth, nBtruth
	print ("truth", data.loc[(data[testtruth]==0) & (data[testtruth]==1)]["totalWeight"].sum() , data.loc[(data['target']==1) & (data[testtruth]==1)]["totalWeight"].sum() )
#################################################################################
## Balance datasets
#https://stackoverflow.com/questions/34803670/pandas-conditional-multiplication

print ("norm", data.loc[data['target']==0]["totalWeight"].sum(),data.loc[data['target']==1]["totalWeight"].sum())
#for tar in [0,1] :
data.loc[data['target']==0, ['totalWeight']] *= 100000/data.loc[data['target']==0]["totalWeight"].sum()
data.loc[data['target']==1, ['totalWeight']] *= 100000/data.loc[data['target']==1]["totalWeight"].sum()
print data.columns.values.tolist()

weights="totalWeight"
# drop events with NaN weights - for safety
#data.replace(to_replace=np.inf, value=np.NaN, inplace=True)
#data.replace(to_replace=np.inf, value=np.zeros, inplace=True)
#data = data.apply(lambda x: pandas.to_numeric(x,errors='ignore'))
data.dropna(subset=["totalWeight"],inplace = True) # data
data.fillna(0)

nS = len(data.loc[data.target.values == 0])
nB = len(data.loc[data.target.values == 1])
print "length of sig, bkg without NaN: ", nS, nB

#################################################################################
### Plot histograms of training variables
hist_params = {'normed': True, 'bins': 15, 'alpha': 0.4}
plt.figure(figsize=(50, 50))
if bdtType=='evtLevelTT_TTH' : labelBKG = "tt"
if bdtType=='evtLevelTTV_TTH' : labelBKG = "ttV"
for n, feature in enumerate(trainVars(True)):
    # add sub plot on our figure
	plt.subplot(7, 7, n+1)
    # define range for histograms by cutting 1% of data from both ends
	min_value, max_value = np.percentile(data[feature], [0.0, 99])
	print (min_value, max_value,feature)
	values, bins, _ = plt.hist(abs(data.ix[data.target.values == 0, feature].values) ,
							   weights= abs(data.ix[data.target.values == 0, weights].values.astype(np.float64)) ,
                               range=(max(0.,min_value), max_value),
							   label=labelBKG, **hist_params )
	values, bins, _ = plt.hist(abs(data.ix[data.target.values == 1, feature].values),
							   weights= abs(data.ix[data.target.values == 1, weights].values.astype(np.float64)) ,
                               range=(max(0.,min_value), max_value), label='Signal', **hist_params)
	areaSig = sum(np.diff(bins)*values)
	#print areaBKG, " ",areaBKG2 ," ",areaSig
	plt.ylim(ymin=0.00001)
	if n == 0 : plt.legend(loc='best')
	plt.title(feature)
	#plt.xscale('log')
	#plt.yscale('log')
plt.ylim(ymin=0)
plt.savefig(channel+"/"+bdtType+"_"+trainvar+"_Variables_BDT.pdf")
plt.clf()
#################################################################################

### Plot aditional histograms

if 1>0 : #channel=="1l_2tau" :
	hist_params = {'normed': True, 'alpha': 0.4}
	plt.figure(figsize=(15, 15))
	n=8
	nbins=40
	for n, feature  in enumerate([
	  #"mvaOutput_1l_2tau_ttbar_HTTWithKinFit_MVAonly",
	  #"mvaOutput_1l_2tau_ttbar_HTTWithKinFit",
	  #"mvaOutput_1l_2tau_ttbar",
	  #"mvaOutput_1l_2tau_ttbar_Old",
	  #"mvaOutput_1l_2tau_ttbar_HTTLepID",
	  #"mvaOutput_1l_2tau_ttbar_OldVar",
	  #"mvaOutput_1l_2tau_ttbar_OldVarHTT",
	'mvaOutput_hadTopTaggerWithKinFit',
	"mvaOutput_Hj_tagger", #"memOutput_LR",
	'mvaOutput_Hjj_tagger',
	#"mvaOutput_2lss_1tau_ttbar",
	#"mvaOutput_2lss_1tau_ttV",
	#"memOutput_tt_LR",  "memOutput_ttZ_LR", "memOutput_ttZ_Zll_LR",
	#"oldVar_from20_to_12", "oldVar_from20_to_7"#,
	#'mvaOutput_hadTopTagger',
	#'mvaOutput_2lss_ttbar', 'mvaOutput_Hj_tagger', 'mvaOutput_Hjj_tagger',
	#'lep1_isTight', 'lep2_isTight','tau_isTight',
	#"failsTightChargeCut",
	#'mvaDiscr_2lss', 'mvaOutput_2lss_ttV',
	#'ncombo'
	]):
	    # add sub plot on our figure
		plt.subplot(3, 3, n+1)
	    # define range for histograms by cutting 1% of data from both ends
		#if n==3 : nbins=20
		#elif n==7 or n==8 : nbins=100
		#else : nbins=10
		min_value, max_value = np.percentile(data[feature], [1, 99])
		print (min_value, max_value,feature)
		values, bins, _ = plt.hist(abs(data.ix[data.target.values == 0, feature].values.astype(np.float64)) ,
								   weights= abs(data.ix[data.target.values == 0, weights].values.astype(np.float64)) ,
	                               range=(max(min_value,0.), max_value),
								   label="TT",
								   bins=nbins,
								   **hist_params )
		values, bins, _ = plt.hist(abs(data.ix[data.target.values == 1, feature].values.astype(np.float64)),
								   weights= data.ix[data.target.values == 1, weights].values.astype(np.float64) ,
	                               range=(max(min_value,0.), max_value),
								   label='Signal',
								   bins=nbins,
								   **hist_params)
		areaSig = sum(np.diff(bins)*values)
		plt.ylim(ymin=0.0001)
		#print areaBKG, " ",areaBKG2 ," ",areaSig
		if n == 0 : plt.legend(loc='best')
		plt.title(feature)
	plt.savefig(channel+"/"+bdtType+"_"+trainvar+"_BDTVariables_BDT.pdf")
	plt.clf()
##################################################################

if 1>0 : #channel=="1l_2tau" :
	##############################################################
	hist_params = {'normed': True, 'alpha': 0.4}
	plt.figure(figsize=(40, 40))
	n=8
	nbins=20
	residualsSignal=[]
	for n, feature  in enumerate(trainVars(True)):
		plt.subplot(7, 7, n+1)
		min_value, max_value = np.percentile(data[feature], [1, 99])
		print (min_value, max_value,feature)
		plot1= plt.hist(abs(dataTightFS.ix[dataTightFS.target.values == 1, feature].values.astype(np.float64)),
								   weights= dataTightFS.ix[dataTightFS.target.values == 1, weights].values.astype(np.float64) ,
	                               range=(max(min_value,0.), max_value), #cumulative=True,
								   label='Fullsim '+FullsimWP+' tau',
								   bins=nbins,
								   **hist_params)
		plot2 = plt.hist(abs(data.ix[data.target.values == 1, feature].values.astype(np.float64)) ,
								   weights= abs(data.ix[data.target.values == 1, weights].values.astype(np.float64)) ,
	                               range=(max(min_value,0.), max_value), #cumulative=True,
								   label="Fastsim "+FastsimWP+" tau",
								   bins=nbins,
								   **hist_params )
		residualsSignal=residualsSignal+[(plot1[0]-plot2[0])/(plot1[0])]
		plt.ylim(ymin=0.00001)
		if n == 0 : plt.legend(loc='best')
		plt.title(feature)
		#plt.xscale('log')
		#plt.yscale('log')
	plt.savefig(channel+"/"+bdtType+"_"+trainvar+"_Variables_signal_fastsim_fullsim.pdf")
	plt.clf()
	#print ("residualsSignal",residualsSignal)
	residualsSignal=np.nan_to_num(residualsSignal)
	for n, feature  in enumerate(trainVars(True)):
		(mu, sigma) = norm.fit(residualsSignal[n])
		plt.subplot(7, 7, n+1)
		n, bins, patches = plt.hist(residualsSignal[n], label='Residuals Full('+FullsimWP+')/Fastsim('+FastsimWP+')  Signal')
		# add a 'best fit' line
		y = mlab.normpdf( bins, mu, sigma)
		l = plt.plot(bins, y, 'r--', linewidth=2)
		plt.ylim(ymin=0)
		plt.title(feature+' '+r'mu=%.3f, sig=%.3f$' %(mu, sigma))
		print feature+' '+r'mu=%.3f, sig=%.3f$' %(mu, sigma)
	plt.savefig(channel+"/"+bdtType+"_"+trainvar+"_Variables_Signal_fullsim_residuals.pdf")
	plt.clf()
	##################################################################
	residualsBKG=[]
	for n, feature  in enumerate(trainVars(True)):
		plt.subplot(7, 7, n+1)
		min_value, max_value = np.percentile(data[feature], [1, 99])
		print (min_value, max_value,feature)
		plot1 = plt.hist(abs(dataTightFS.ix[dataTightFS.target.values == 0, feature].values.astype(np.float64)),
								   weights= dataTightFS.ix[dataTightFS.target.values == 0, weights].values.astype(np.float64) ,
	                               range=(max(min_value,0.), max_value), #cumulative=True,
								   label='Fullsim '+FullsimWP+' tau',
								   bins=nbins,
								   **hist_params)
		plot2 = plt.hist(abs(data.ix[data.target.values == 0, feature].values.astype(np.float64)) ,
								   weights= abs(data.ix[data.target.values == 0, weights].values.astype(np.float64)) ,
	                               range=(max(min_value,0.), max_value), #cumulative=True,
								   label="Fastsim ("+FastsimWP+") tau",
								   bins=nbins,
								   **hist_params )
		residualsBKG=residualsBKG+[(plot1[0]-plot2[0])/(plot1[0])]
		plt.ylim(ymin=0.00000000000000000001)
		if n == 0 : plt.legend(loc='best')
		plt.title(feature)
		#plt.xscale('log')
		#plt.yscale('log')
	plt.ylim(ymin=0)
	plt.savefig(channel+"/"+bdtType+"_"+trainvar+"_Variables_BKG_fastsim_fullsim.pdf")
	plt.clf()

	residualsBKG=np.nan_to_num(residualsBKG)
	for n, feature  in enumerate(trainVars(True)):
		(mu, sigma) = norm.fit(residualsBKG[n])
		plt.subplot(7, 7, n+1)
		n, bins, patches = plt.hist(residualsBKG[n], label='Residuals Full('+FullsimWP+')/Fastsim('+FastsimWP+') BKG')
		# add a 'best fit' line
		y = mlab.normpdf( bins, mu, sigma)
		l = plt.plot(bins, y, 'r--', linewidth=2)
		plt.ylim(ymin=0)
		plt.title(feature+' '+r'mu=%.3f, sig=%.3f$' %(mu, sigma))
		print feature+' '+r'mu=%.3f, sig=%.3f$' %(mu, sigma)
	plt.savefig(channel+"/"+bdtType+"_"+trainvar+"_Variables_BKG_fullsim"+FullsimWP+"_residuals.pdf")
	plt.clf()

	##################################################################
	hist_params = {'normed': True, 'alpha': 0.4}
	plt.figure(figsize=(40, 40))
	for n, feature  in enumerate(trainVars(True)):
	    # add sub plot on our figure
		plt.subplot(7, 7, n+1)
		min_value, max_value = np.percentile(data[feature], [1, 99])
		print (min_value, max_value,feature)
		plot1= plt.hist(abs(dataTightFS.ix[dataTightFS.target.values == 0, feature].values.astype(np.float64)),
								   weights= dataTightFS.ix[dataTightFS.target.values == 0, weights].values.astype(np.float64) ,
	                               range=(max(min_value,0.), max_value), #cumulative=True,
								   label='Fastsim Tight tau',
								   bins=nbins,
								   **hist_params)
		plot2= plt.hist(abs(data.ix[data.target.values == 0, feature].values.astype(np.float64)) ,
								   weights= abs(data.ix[data.target.values == 0, weights].values.astype(np.float64)) ,
	                               range=(max(min_value,0.), max_value), #cumulative=True,
								   label="Fastsim "+FastsimWP+" tau",
								   bins=nbins,
								   **hist_params )
		if n == 0 : plt.legend(loc='best')
		plt.ylim(ymin=0.00001)
		plt.title(feature)
		#plt.xscale('log')
		#plt.yscale('log')
	plt.ylim(ymin=0)
	plt.savefig(channel+"/"+bdtType+"_"+trainvar+"_Variables_BKG_fastsim"+FastsimWP+"_tight.pdf")
	plt.clf()
	##################################################################
	hist_params = {'normed': True, 'alpha': 0.4}
	plt.figure(figsize=(40, 40))
	for n, feature  in enumerate(trainVars(True)):
	    # add sub plot on our figure
		plt.subplot(7, 7, n+1)
		min_value, max_value = np.percentile(data[feature], [1, 99])
		print (min_value, max_value,feature)
		plot1= plt.hist(abs(dataTightFS.ix[dataTightFS.target.values == 1, feature].values.astype(np.float64)),
								   weights= dataTightFS.ix[dataTightFS.target.values == 1, weights].values.astype(np.float64) ,
	                               range=(max(min_value,0.), max_value), #cumulative=True,
								   label='Fastsim Tight tau',
								   bins=nbins,
								   **hist_params)
		plot2= plt.hist(abs(data.ix[data.target.values == 1, feature].values.astype(np.float64)) ,
								   weights= abs(data.ix[data.target.values == 1, weights].values.astype(np.float64)) ,
	                               range=(max(min_value,0.), max_value), #cumulative=True,
								   label="Fastsim "+FastsimWP+" tau",
								   bins=nbins,
								   **hist_params )
		if n == 0 : plt.legend(loc='best')
		plt.ylim(ymin=0.00001)
		plt.title(feature)
		#plt.xscale('log')
		#plt.yscale('log')
	plt.ylim(ymin=0)
	plt.savefig(channel+"/"+bdtType+"_"+trainvar+"_Variables_BKG_fastsim"+FastsimWP+"_tight.pdf")
	plt.clf()
	##################################################################
	hist_params = {'normed': True, 'alpha': 0.4}
	plt.figure(figsize=(40, 40))

	for n, feature  in enumerate(trainVars(True)):
		plt.subplot(7, 7, n+1)
		min_value, max_value = np.percentile(data[feature], [1, 99])
		print (min_value, max_value,feature)
		plot1= plt.hist(abs(dataTightFS.ix[dataTightFS.target.values == 1, feature].values.astype(np.float64)),
								   weights= dataTightFS.ix[dataTightFS.target.values == 1, weights].values.astype(np.float64) ,
	                               range=(max(min_value,0.), max_value), #cumulative=True,
								   label='Fullsim signal '+FullsimWP+' tau',
								   bins=nbins,
								   **hist_params)
		plot2= plt.hist(abs(dataTightFS.ix[dataTightFS.target.values == 0, feature].values.astype(np.float64)),
								   weights= dataTightFS.ix[dataTightFS.target.values == 0, weights].values.astype(np.float64) ,
	                               range=(max(min_value,0.), max_value), #cumulative=True,
								   label='Fullsim BKG '+FullsimWP+' tau',
								   bins=nbins,
								   **hist_params)
		plt.ylim(ymin=0.00001)
		#plt.xscale('log')
		#plt.yscale('log')
		#if n == 0 :
		#	print ("BKG",plot1[0],plot2[0],plot2[1])
		#	print ("residuals",plot1[0]-plot2[0],plot2[1])
		if n == 0 : plt.legend(loc='best')
		plt.title(feature)
	plt.savefig(channel+"/"+bdtType+"_"+trainvar+"_Variables_BKG_fullsim"+FullsimWP+".pdf")
	plt.clf()
###################################################################
if channel=="22lss_1tau" : njet="nJet25_Recl"
else : njet="nJet"
totestcorr=['mvaOutput_hadTopTaggerWithKinFit',
"mvaOutput_Hj_tagger",
'mvaOutput_Hjj_tagger',] #]
totestcorrNames=['HTT',
"Hj_tagger",
'Hjj_tagger',njet]
for ii in [1,2] :
	if ii == 1 :
		datad=data.loc[data['target'].values == 1]
		label="signal"
	else :
		datad=data.loc[data['target'].values == 0]
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
	fit_params = { "eval_set" : [(valdataset[trainVars(False)].values,valdataset["target"])],
                           "eval_metric" : "auc",
                           "early_stopping_rounds" : early_stopping_rounds,
						   'sample_weight': valdataset["totalWeight"].values }
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
			learning_rate = options.lr
			#max_features = 'sqrt',
			#min_samples_leaf = 100
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
fpr, tpr, thresholds = roc_curve(traindataset["target"], proba[:,1],
	sample_weight=(traindataset[weights].astype(np.float64)) )
train_auc = auc(fpr, tpr, reorder = True)
print("XGBoost train set auc - {}".format(train_auc))
proba = cls.predict_proba(valdataset[trainVars(False)].values )
fprt, tprt, thresholds = roc_curve(valdataset["target"], proba[:,1], sample_weight=(valdataset[weights].astype(np.float64))  )
test_auct = auc(fprt, tprt, reorder = True)
print("XGBoost test set auc - {}".format(test_auct))
proba = cls.predict_proba(dataTight[trainVars(False)].values )
fprtight, tprtight, thresholds = roc_curve(dataTight["target"], proba[:,1], sample_weight=(dataTight[weights].astype(np.float64))  )
test_auctight = auc(fprtight, tprtight, reorder = True)
print("XGBoost test set auc - tight lep ID - {}".format(test_auctight))
proba = cls.predict_proba(dataTightFS[trainVars(False)].values)
fprtightF, tprtightF, thresholds = roc_curve(dataTightFS["target"], proba[:,1], sample_weight=(dataTightFS[weights].astype(np.float64)) )
test_auctightF = auc(fprtightF, tprtightF, reorder = True)
print("XGBoost test set auc - fullsim all - {}".format(test_auctightF))
if bdtType=="evtLevelTT_TTH" :
	tightTT=dataTightFS.ix[(dataTightFS.proces.values=='TT') | (dataTightFS.proces.values=='signal')]
if bdtType=="evtLevelTTV_TTH" :
	tightTT=dataTightFS.ix[(dataTightFS.proces.values=='TTZ') | (dataTightFS.proces.values=='TTW') | (dataTightFS.proces.values=='signal')]
proba = cls.predict_proba(tightTT[trainVars(False)].values)
fprtightFI, tprtightFI, thresholds = roc_curve(tightTT["target"], proba[:,1], sample_weight=(tightTT[weights].astype(np.float64)))
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
ax.plot(fprtight, tprtight, lw=1, label='XGB test - tight ID (area = %0.3f)'%(test_auctight))
ax.plot(fprtightFI, tprtightFI, lw=1, label='XGB test - Fullsim (area = %0.3f)'%(test_auctightFI))
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
			datad=traindataset.loc[traindataset['target'].values == 1]
			label="signal"
		else :
			datad=traindataset.loc[traindataset['target'].values == 0]
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
	for ii in [1,2] :
		if ii == 1 :
			datad=dataTightFS.loc[dataTightFS['target'].values == 1]
			label="signal"
		else :
			datad=dataTightFS.loc[dataTightFS['target'].values == 0]
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
