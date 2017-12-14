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
	#inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Nov30-BDT-ncombo/histograms/1l_2tau/forBDTtraining_OS/' # loose lepton, relaxed tau
	#inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Dec08-BDT-MEM/histograms/1l_2tau/forBDTtraining_OS/'#   normal lepton , tight tau
	#inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Dec08-BDT-tight/histograms/1l_2tau/forBDTtraining_OS/' # 2017Dec08-BDT-tight lepton and tau tight
	inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Dec-BDT-noMEM-LooseLepMedTau-TagT-fakeR/histograms/1l_2tau/forBDTtraining_OS/'#  - tight lepton, medium tau
	# 2017Dec13-BDT-noMEM-LooseLepMedTau-TagT-fakeR , tight lepton, medium tau and failsTightChargeCut flag

if channel=='2lss_1tau':
	channelInTree='2lss_1tau_lepSS_sumOS_Loose'
	#inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Oct24/histograms/2lss_1tau/forBDTtraining_SS_OS/'
	#inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Dec08-BDT-MEM/histograms/2lss_1tau/forBDTtraining_SS_OS/'
	# 2017Dec08-BDT-MEM -- with HTT correct
	#inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Dec08-BDT-noMEM/histograms/2lss_1tau/forBDTtraining_SS_OS/' # 2017Dec08-BDT-noMEM -- HTT and truth correct
	#inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Dec08-BDT-noMEM-clenaedJets/histograms/2lss_1tau/forBDTtraining_SS_OS/' # 2017Dec08-BDT-noMEM-clenaedJets
	#inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Dec-BDT-noMEM-LooseLepMedTau-cleanjets/histograms/2lss_1tau/forBDTtraining_SS_OS/' # 2017Dec08-BDT-noMEM-looseLepMedTau
	#inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Dec-BDT-noMEM-LooseLepMedTau-TagT-fakeR/histograms/2lss_1tau/forBDTtraining_SS_OS/' # 2017Dec-BDT-noMEM-LooseLepMedTau-TagT-fakeR
	inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Dec13-BDT-noMEM-LooseLepMedTau-TagT-fakeR/histograms/2lss_1tau/forBDTtraining_SS_OS/' # with charge tag, 2017Dec13-BDT-noMEM-LooseLepMedTau-TagT-fakeR
	criteria=['lep1_isTight', 'lep2_isTight','tau_isTight',"failsTightChargeCut"]
	# 2017Dec-BDT-noMEM-LooseLepMedTau-TagT-fakeR
	# 2017Dec-BDT-noMEM-LooseLepMedTau-cleanjets
	# 2017Mar27_dr03mvaLoose
	# ###########################
	#channelInTree='2lss_1tau_lepSS_sumOS_Loose'
	#inputPath='/home/arun/ttHAnalysis/2016/2017Mar27_dr03mvaLoose/histograms/2lss_1tau/forBDTtraining_SS_OS/' # /home/arun/ttHAnalysis/2016/2017Mar27_dr03mvaLoose/
    # ###########################
	#channelInTree='2lss_1tau_lepSS_sumOS_Fakeable_wFakeRateWeights'
	#inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Dec08-BDT-noMEM-fakableLepMedTau/histograms/2lss_1tau/forBDTtraining_SS_OS/' # 2017Dec08-BDT-noMEM-fakableLepMedTau
	#inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Dec08-BDT-noMEM-fakableLepLooseTau/histograms/2lss_1tau/forBDTtraining_SS_OS/' # 2017Dec08-BDT-noMEM-fakableLepMedTau
	# #########################
	#channelInTree='2lss_1tau_lepSS_sumOS_Tight'
	#inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Dec08-BDT-noMEM-tighLep/histograms/2lss_1tau/forBDTtraining_SS_OS/' #  2017Dec08-BDT-noMEM-tighLep

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
			'mT_lep1', 'mT_lep2', 'mTauTauVis2',
			'mTauTauVis1', 'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet',
			'ptmiss', 'max_lep_eta', 'tau_pt',
			'mvaOutput_hadTopTaggerWithKinFit', 'mvaOutput_Hj_tagger',
			'unfittedHadTop_pt', 'mbb', 'avg_dr_jet', 'nJet25_Recl'
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

        if trainvar=="noHTT" and channel=="2lss_1tau" and bdtType=="evtLevelTTV_TTH" and all==False :
			return [
			'avg_dr_jet', 'dr_lep1_tau', 'dr_lep2_tau', 'dr_leps',
			'lep1_conePt', 'lep2_conePt',
			'mT_lep1', 'mT_lep2', 'mTauTauVis1', 'mTauTauVis2',
			'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet', 'ptmiss', 'tau_pt'
			]

	if channel=="1l_2tau" and all==True :return [
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
		"mTauTauVis"#,
		#"lumiScale", "genWeight", "evtWeight",
		#"mT_lepHadTop" #,"mT_lepHadTopH"
		#"HadTop_pt","HadTop_eta","dr_lep_HadTop","dr_HadTop_tau_OS","dr_HadTop_tau_SS",
		#"dr_HadTop_tau_lead", #"dr_HadTop_tau_sublead",
		#"dr_HadTop_tautau",
		#"dr_HadTop_lepton","mass_HadTop_lepton", "costS_HadTop_tautau",
		#"mvaOutput_hadTopTaggerWithKinFit", #"mvaOutput_hadTopTagger",
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

  # ['avg_dr_jet', 'dr_taus', 'ptmiss', 'lep_conePt', 'mT_lep',
  # 'mTauTauVis', 'mindr_lep_jet', 'mindr_tau1_jet', 'nJet', 'dr_lep_tau_ss',
  # 'dr_lep_tau_lead', 'costS_tau']
	if trainvar=="noHTT" and channel=="1l_2tau"  and bdtType=="evtLevelTT_TTH" and all==False :return [
		'avg_dr_jet',
		'dr_taus',
		#'htmiss',
		'ptmiss',
		'lep_conePt',
		#'lep_eta',
		#'lep_pt',
		#'lep_tth_mva',
		'mT_lep',
		'mTauTauVis',
		'mindr_lep_jet',
		'mindr_tau1_jet',
		#'mindr_tau2_jet',
		#'tau1_eta',
		#'tau1_mva',
		#'tau1_pt',
		#'tau2_eta',
		#'tau2_mva',
		#'tau2_pt',
		#'nBJetLoose',
		#'nBJetMedium',
		'nJet',
		#'dr_lep_tau_os',
		'dr_lep_tau_ss',
		"dr_lep_tau_lead",
		#"dr_lep_tau_sublead",
		"costS_tau",
		#"dr_HadTop_tau_OS",
		#"dr_HadTop_tau_SS",
		#"mT_lepHadTop",
		"mT_lepHadTopH",
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

	if trainvar=="HTTWithKinFitKin" and channel=="1l_2tau"  and bdtType=="evtLevelTT_TTH" and all==False :return [
						'avg_dr_jet',
						'dr_taus',
						#'ptmiss',
						'htmiss',
						'mTauTauVis',
						'mindr_lep_jet',
						'mindr_tau1_jet',
						'dr_lep_tau_ss',
						"costS_tau",
						'nBJetLoose',
						"tau1_pt",
						"tau2_pt",
						#'lep_conePt',
						'nJet',
						"dr_lep_tau_lead",
						'mT_lep',
						'mvaOutput_hadTopTaggerWithKinFit',
						'HadTop_pt',
						"dr_HadTop_tau_lead"
		]
####################################################################################################
## Load data
my_cols_list=trainVars(False)+['key','target','file']+criteria #,'tau_frWeight','lep1_frWeight','lep1_frWeight'
testtruth="bWj1Wj2_isGenMatchedWithKinFit"
# if channel=='2lss_1tau' : my_cols_list=my_cols_list+['tau_frWeight','lep1_frWeight','lep2_frWeight']
# those last are only for channels where selection is relaxed (2lss_1tau) === solve later
data = pandas.DataFrame(columns=my_cols_list)
if bdtType=="evtLevelTT_TTH" : keys=['ttHToNonbb','TTTo2L2Nu','TTToSemilepton']
if bdtType=="evtLevelTTV_TTH" : keys=['ttHToNonbb','TTZToLLNuNu','TTWJetsToLNu']
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
					chunk_df["totalWeight"] = chunk_df["evtWeight"]*chunk_df['tau_frWeight']*chunk_df['lep1_frWeight']*chunk_df['lep2_frWeight']
				if channel=="1l_2tau" : chunk_df["totalWeight"] = chunk_df.evtWeight
				###########
				if channel=="2lss_1tau" :
					data=data.append(chunk_df.ix[chunk_df.failsTightChargeCut.values == 0], ignore_index=True)
				else : #
					#if 1>0 :
					data=data.append(chunk_df, ignore_index=True)
		else : print ("file "+list[ii]+"was empty")
		tfile.Close()
	if len(data) == 0 : continue
	nS = len(data.ix[(data.target.values == 0) & (data.key.values==folderName)])
	nB = len(data.ix[(data.target.values == 1) & (data.key.values==folderName)])
	print "length of sig, bkg: ", nS, nB
	if channel=="1l_2tau" or channel=="2lss_1tau" :
		nSthuth = len(data.ix[(data.target.values == 0) & (data.bWj1Wj2_isGenMatched.values==1) & (data.key.values==folderName)])
		nBtruth = len(data.ix[(data.target.values == 1) & (data.bWj1Wj2_isGenMatched.values==1) & (data.key.values==folderName)])
		nSthuthKin = len(data.ix[(data.target.values == 0) & (data.bWj1Wj2_isGenMatchedWithKinFit.values==1) & (data.key.values==folderName)])
		nBtruthKin = len(data.ix[(data.target.values == 1) & (data.bWj1Wj2_isGenMatchedWithKinFit.values==1) & (data.key.values==folderName)])
		nShadthuth = len(data.ix[(data.target.values == 0) & (data.hadtruth.values==1) & (data.key.values==folderName)])
		nBhadtruth = len(data.ix[(data.target.values == 1) & (data.hadtruth.values==1) & (data.key.values==folderName)])
		print "truth:              ", nSthuth, nBtruth
		print "truth Kin:          ", nSthuthKin, nBtruthKin
		print "hadtruth:           ", nShadthuth, nBhadtruth
print (data.columns.values.tolist())
n = len(data)
nS = len(data.ix[data.target.values == 0])
nB = len(data.ix[data.target.values == 1])
print "length of sig, bkg: ", nS, nB
#print ("weigths", data.loc[data['target']==0]["totalWeight"].sum() , data.loc[data['target']==1]["totalWeight"].sum() )

if channel=="1l_2tau" or channel=="2lss_1tau":
	nSthuth = len(data.ix[(data.target.values == 0) & (data.bWj1Wj2_isGenMatched.values==1)])
	nBtruth = len(data.ix[(data.target.values == 1) & (data.bWj1Wj2_isGenMatched.values==1)])
	print "truth:              ", nSthuth, nBtruth
	print ("truth", data.loc[(data['target']==0) & (data['bWj1Wj2_isGenMatched']==1)]["totalWeight"].sum() , data.loc[(data['target']==1) & (data['bWj1Wj2_isGenMatched']==1)]["totalWeight"].sum() )
#################################################################################
## Balance datasets
#https://stackoverflow.com/questions/34803670/pandas-conditional-multiplication

print ("norm", data.loc[data['target']==0]["totalWeight"].sum(),data.loc[data['target']==1]["totalWeight"].sum())
#for tar in [0,1] :
data.loc[data['target']==0, ['totalWeight']] *= 100000/data.loc[data['target']==0]["totalWeight"].sum()
data.loc[data['target']==1, ['totalWeight']] *= 100000/data.loc[data['target']==1]["totalWeight"].sum()

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
hist_params = {'normed': True, 'bins': 40, 'alpha': 0.4}
plt.figure(figsize=(50, 50))
if bdtType=='evtLevelTT_TTH' : labelBKG = "tt"
if bdtType=='evtLevelTTV_TTH' : labelBKG = "ttV"
for n, feature in enumerate(trainVars(False)):
    # add sub plot on our figure
	plt.subplot(8, 8, n+1)
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
	if n == 0 : plt.legend(loc='best')
	plt.title(feature)
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
	'mvaOutput_hadTopTagger',
	#'mvaOutput_2lss_ttbar', 'mvaOutput_Hj_tagger', 'mvaOutput_Hjj_tagger',
	#'lep1_isTight', 'lep2_isTight','tau_isTight',
	#"failsTightChargeCut",
	#'mvaDiscr_2lss', 'mvaOutput_2lss_ttV',
	'ncombo'
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
		#print areaBKG, " ",areaBKG2 ," ",areaSig
		if n == 0 : plt.legend(loc='best')
		plt.title(feature)
	plt.savefig(channel+"/"+bdtType+"_"+trainvar+"_BDTVariables_BDT.pdf")
	plt.clf()
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
proba = cls.predict_proba(traindataset[trainVars(False)].values  )
fpr, tpr, thresholds = roc_curve(traindataset["target"], proba[:,1] )
train_auc = auc(fpr, tpr, reorder = True)
print("XGBoost train set auc - {}".format(train_auc))
proba = cls.predict_proba(valdataset[trainVars(False)].values)
fprt, tprt, thresholds = roc_curve(valdataset["target"], proba[:,1] )
test_auct = auc(fprt, tprt, reorder = True)
print("XGBoost test set auc - {}".format(test_auct))
if channel=="2lss_1tau" : # 'lep1_isTight', 'lep2_isTight','tau_isTight',"failsTightChargeCut"
	datatight=data.loc[ (data.lep1_isTight.values == 1) &
						(data.lep2_isTight.values == 1) &
						(data.tau_isTight.values == 1) &
						(data.failsTightChargeCut.values == 0 ) ]
	proba = cls.predict_proba(datatight[trainVars(False)].values)
	fprtight, tprtight, thresholds = roc_curve(datatight["target"], proba[:,1] )
	test_auctight = auc(fprtight, tprtight, reorder = True)
	print("XGBoost test set auc - tight lep ID - {}".format(test_auctight))
#print cls.params_
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
"""
clc = catboost.CatBoostClassifier(iterations=2000, depth=1, learning_rate=0.01, loss_function='Logloss',gradient_iterations=3,od_pval=0.01, verbose=False)
clc.fit(
	traindataset[trainVars(False)].values,
	traindataset.target.astype(np.bool)
	#sample_weight= np.absolute((traindataset[weights].astype(np.float64))),
	#eval_set=[(traindataset[trainVars(False)].values,  traindataset.target.astype(np.bool),traindataset[weights].astype(np.float64)),
	#(valdataset[trainVars(False)].values,  valdataset.target.astype(np.bool), valdataset[weights].astype(np.float64))]
	)
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
"""
###############
"""
clf = GradientBoostingClassifier(max_depth=3,learning_rate=0.02,n_estimators=500,min_samples_leaf=100) # ,min_samples_leaf=10,min_samples_split=10
clf.fit(traindataset[trainVars(False)].values,
	traindataset.target.astype(np.bool),
	sample_weight=(traindataset[weights].astype(np.float64))
	)
print ("GradientBoosting trained")
proba = clf.predict_proba(traindataset[trainVars(False)].values  )
fprf, tprf, thresholdsf = roc_curve(traindataset["target"], proba[:,1] )
train_aucf = auc(fprf, tprf, reorder = True)
print("GradientBoosting train set auc - {}".format(train_aucf))
proba = clf.predict_proba(valdataset[trainVars(False)].values)
fprtf, tprtf, thresholdsf = roc_curve(valdataset["target"], proba[:,1] )
test_auctf = auc(fprtf, tprtf, reorder = True)
print("GradientBoosting test set auc - {}".format(test_auctf))
"""
##################################################
fig, ax = plt.subplots()
## ROC curve
#ax.plot(fprf, tprf, lw=1, label='GB train (area = %0.3f)'%(train_aucf))
#ax.plot(fprtf, tprtf, lw=1, label='GB test (area = %0.3f)'%(test_auctf))
ax.plot(fpr, tpr, lw=1, label='XGB train (area = %0.3f)'%(train_auc))
ax.plot(fprt, tprt, lw=1, label='XGB test (area = %0.3f)'%(test_auct))
#ax.plot(fprc, tprc, lw=1, label='CB train (area = %0.3f)'%(train_aucc))
ax.plot(fprtight, tprtight, lw=1, label='XGB test - tight ID (area = %0.3f)'%(test_auctight))
ax.set_ylim([0.0,1.0])
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
