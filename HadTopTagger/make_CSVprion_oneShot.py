import sys , time
#import sklearn_to_tmva
import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
import pandas
#from pandas import HDFStore,DataFrame
import math , array

import matplotlib
matplotlib.use('agg')
#matplotlib.use('PS')   # generate postscript output by default
import matplotlib.pyplot as plt
#from matplotlib import cm as cm
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
import glob
from itertools import groupby

from optparse import OptionParser
parser = OptionParser()
parser.add_option("--process ", type="string", dest="process", help="process", default='T')
parser.add_option("--dofiles ", type="int", dest="dofiles", help="dofiles batch", default=0)
(options, args) = parser.parse_args()

dofiles=options.dofiles

"""
python make_CSVprion_oneShot.py --process 'ttHToNonbb' --dofiles > ttHToNonbb_0.log &
python make_CSVprion_oneShot.py --process 'TTToSemilepton' --dofiles > TTToSemilepton_0.log &
python make_CSVprion_oneShot.py --process 'TTWJetsToLNu' --dofiles > TTWJetsToLNu_2.log &
python make_CSVprion_oneShot.py --process 'TTZToLLNuNu' > TTZToLLNuNu_2.log &

,,'TTTo2L2Nu',,
"""

#inputPath = "/hdfs/local/veelken/ttHAnalysis/2016/2017Aug31/histograms/hadTopTagger/"
inputPath="/hdfs/local/acaan/ttHAnalysis/2016/2017Nov11/histograms/hadTopTagger/"
datasetsout = "structured/"
maxSignal=30000
partition=20

def dataAppend(tree0,data,countmatchB,folderName):
				data=data.append({
				'bWj1Wj2_isGenMatched': tree0.bWj1Wj2_isGenMatched,
				#'b_isGenMatched': tree0.b_isGenMatched,
				#'Wj1_isGenMatched': tree0.Wj1_isGenMatched,
				#'Wj2_isGenMatched': tree0.Wj2_isGenMatched,
				#'statusKinFit': tree0.statusKinFit,
				#'qg_b': tree0.qg_b,
				'qg_Wj2': tree0.qg_Wj2,
				'qg_Wj1': tree0.qg_Wj1,
				'pT_bWj1Wj2': tree0.pT_bWj1Wj2,
				#'pT_b': tree0.pT_b,
				#'pT_Wj2': tree0.pT_Wj2,
				'pT_Wj1Wj2': tree0.pT_Wj1Wj2,
				#'pT_Wj1': tree0.pT_Wj1,
				'nllKinFit': tree0.nllKinFit,
				#'max_dR_div_expRjet': tree0.max_dR_div_expRjet,
				'm_bWj2': tree0.m_bWj2,
				'm_bWj1Wj2': tree0.m_bWj1Wj2,
				'm_bWj1': tree0.m_bWj1,
				#'m_Wj1Wj2_div_m_bWj1Wj2': tree0.m_Wj1Wj2_div_m_bWj1Wj2,
				'm_Wj1Wj2': tree0.m_Wj1Wj2,
				#'logPKinFit': tree0.logPKinFit,
				#'logPErrKinFit': tree0.logPErrKinFit,
				#'dR_bWj2': tree0.dR_bWj2,
				#'dR_bWj1': tree0.dR_bWj1,
				#'dR_bW': tree0.dR_bW,
				'dR_Wj1Wj2': tree0.dR_Wj1Wj2,
				'alphaKinFit': tree0.alphaKinFit,
				'CSV_b': tree0.CSV_b,
				'pT_b' : tree0.pT_b,
				#'eta_b' : tree0.eta_b,
				#'phi_b' : tree0.phi_b,
				#'mass_b' : tree0.mass_b,
				'kinFit_pT_b' : tree0.kinFit_pT_b,
				#'kinFit_eta_b' : tree0.kinFit_eta_b,
				#'kinFit_phi_b' : tree0.kinFit_phi_b,
				#'kinFit_mass_b' : tree0.kinFit_mass_b,
				'pT_Wj1' : tree0.pT_Wj1,
				#'eta_Wj1' : tree0.eta_Wj1,
				#'phi_Wj1' : tree0.phi_Wj1,
				#'mass_Wj1' : tree0.mass_Wj1,
				'kinFit_pT_Wj1' : tree0.kinFit_pT_Wj1,
				#'kinFit_eta_Wj1' : tree0.kinFit_eta_Wj1,
				#'kinFit_phi_Wj1' : tree0.kinFit_phi_Wj1,
				#'kinFit_mass_Wj1' : tree0.kinFit_mass_Wj1,
				'pT_Wj2' : tree0.pT_Wj2,
				#'eta_Wj2' : tree0.eta_Wj2,
				#'phi_Wj2' : tree0.phi_Wj2,
				#'mass_Wj2' : tree0.mass_Wj2,
				'kinFit_pT_Wj2' : tree0.kinFit_pT_Wj2,
				#'kinFit_eta_Wj2' : tree0.kinFit_eta_Wj2,
				#'kinFit_phi_Wj2' : tree0.kinFit_phi_Wj2,
				#'kinFit_mass_Wj2' : tree0.kinFit_mass_Wj2,
				#"cosTheta_leadWj_restTop" : tree0.cosTheta_leadWj_restTop,
				#"cosTheta_subleadWj_restTop" : tree0.cosTheta_subleadWj_restTop,
				"cosTheta_leadEWj_restTop" : tree0.cosTheta_leadEWj_restTop,
				"cosTheta_subleadEWj_restTop" : tree0.cosTheta_subleadEWj_restTop,
				#"cosTheta_Kin_leadWj_restTop" : tree0.cosTheta_Kin_leadWj_restTop,
				#"cosTheta_Kin_subleadWj_restTop" : tree0.cosTheta_Kin_subleadWj_restTop,
				"cosThetaW_rest" : tree0.cosThetaW_rest,
				"cosThetaKinW_rest" : tree0.cosThetaKinW_rest,
				#"cosThetaW_lab" : tree0.cosThetaW_lab,
				#"cosThetaKinW_lab" : tree0.cosThetaKinW_lab,
				"cosThetab_rest" : tree0.cosThetab_rest,
				#"cosThetaKinb_rest" : tree0.cosThetaKinb_rest,
				#"cosThetab_lab" : tree0.cosThetab_lab,
				#"cosThetaKinb_lab" : tree0.cosThetaKinb_lab,
				"Dphi_Wj1_Wj2_lab" : tree0.Dphi_Wj1_Wj2_lab,
				#"Dphi_KinWj1_KinWj2_lab" : tree0.Dphi_KinWj1_KinWj2_lab,
				#"Dphi_Wb_rest" : tree0.Dphi_Wb_rest,
				#"Dphi_KinWb_rest" : tree0.Dphi_KinWb_rest,
				"Dphi_Wb_lab" : tree0.Dphi_Wb_lab,
				"Dphi_KinWb_lab" : tree0.Dphi_KinWb_lab,
				"cosThetaWj1_restW" : tree0.cosThetaWj1_restW,
				#"cosThetaKinWj_restW" : tree0.cosThetaKinWj_restW,
				'eventRaw' : countmatchB ,
				'checkEventRaw' : tree0.evt,
				'run' : tree0.run,
				'lumi' : tree0.lumi
				#'process' : folderName
				#'CSV_Wj1': tree0.CSV_Wj1,
				#'CSV_Wj2': tree0.CSV_Wj2,
				}, ignore_index=True)
				return data

#######################################################
## save structured in tree and CSV prior plain tree in annother
## save only 50.000 events in each
#######################################################

print ("Date: ", time.asctime( time.localtime(time.time()) ))

doHist=True
histAllCombo=[]
histOnlyMatchedCombo=[]
histOnlyMatchedCSVCombo=[]
histOnlyMatchedCSVsortCombo=[]

histCSVStatus=[]

keys = ['ttHToNonbb','TTToSemilepton','TTTo2L2Nu','TTZToLLNuNu','TTWJetsToLNu']
####################################################################################################
## Load data

folderName=options.process #'TTToSemilepton'
"""


if dofiles==5 :
	if folderName =='ttHToNonbb' : initfile=11
	if folderName =='TTToSemilepton' : initfile=9
	if folderName =='TTTo2L2Nu' : initfile=26
	if folderName =='TTZToLLNuNu' : initfile=140
	if folderName =='TTWJetsToLNu' : initfile=164
if dofiles==4 :
	if folderName =='ttHToNonbb' : initfile=9
	if folderName =='TTToSemilepton' : initfile=6
	if folderName =='TTTo2L2Nu' : initfile=26
	if folderName =='TTZToLLNuNu' : initfile=112
	if folderName =='TTWJetsToLNu' : initfile=120
if dofiles==3 :
	if folderName =='ttHToNonbb' : initfile=7
	if folderName =='TTToSemilepton' : initfile=4
	if folderName =='TTTo2L2Nu' : initfile=26
	if folderName =='TTZToLLNuNu' :
		initfile=84
	if folderName =='TTWJetsToLNu' :
		initfile=86
if dofiles==2 :
	if folderName =='ttHToNonbb' :
		initfile=4
	if folderName =='TTToSemilepton' :
		initfile=2
	if folderName =='TTTo2L2Nu' :
		initfile=26
	if folderName =='TTZToLLNuNu' :
		initfile=56
	if folderName =='TTWJetsToLNu' :
		initfile=56
if dofiles==1 :
	if folderName =='ttHToNonbb' :
		initfile=4
	if folderName =='TTToSemilepton' :
		initfile=1
	if folderName =='TTTo2L2Nu' :
		initfile=26
	if folderName =='TTZToLLNuNu' :
		initfile=28
		endfile=0
	if folderName =='TTWJetsToLNu' : initfile=26
if dofiles==0 :
	if folderName =='ttHToNonbb' : initfile=0
	if folderName =='TTToSemilepton' : initfile=0
	if folderName =='TTTo2L2Nu' : initfile=26
	if folderName =='TTZToLLNuNu' :
		initfile=0
		endfile=27
	if folderName =='TTWJetsToLNu' : initfile=0

#"""
#if bdtType=="evtLevelTT_TTH" : keys=['ttHToNonbb','TTTo2L2Nu','TTToSemilepton']
#if bdtType=="evtLevelTTV_TTH" : keys=['ttHToNonbb','TTZToLLNuNu','TTWJetsToLNu']
if 1>0:
	outfileCSVsort=datasetsout+'/'+folderName+'_CSVsort_from_'+str(maxSignal)+'sig_'+str(dofiles)+'.csv'
	#outfileCSV=datasetsout+'/'+folderName+'_CSVprior_from_'+str(maxSignal)+'sig_'+str(dofiles)+'.csv'
	outfile=datasetsout+'/'+folderName+'_Structured_from_'+str(maxSignal)+'sig_'+str(dofiles)+'.csv'
	outfileAll=datasetsout+'/'+folderName+'_StructuredAll_from_'+str(maxSignal)+'sig_'+str(dofiles)+'.csv'
	#dataCSVsort = pandas.HDFStore(folderName+'_CSVprior.h5', data_columns=trainVars()+['eventRaw','process'])
	#dataStructured = pandas.HDFStore(folderName+'_Structured.h5') # , data_columns=trainVars()+['eventRaw','process']
	#dataCSVsort = pandas.DataFrame() # columns=trainVars()+['eventRaw'])
	dataStructured = pandas.DataFrame() #columns=trainVars()+['eventRaw'])
	dataCSVsort = pandas.DataFrame()
	dataAll = pandas.DataFrame()
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
	inputTree = 'analyze_hadTopTagger/evtntuple/'+sampleName+'/evtTree'
	if ('TTT' in folderName) or folderName=='ttHToNonbb' :
		# ttHToNonbb_fastsim_p1/ttHToNonbb_fastsim_p1_8.root
		procP1=glob.glob(inputPath+folderName+"_fastsim_p1/"+folderName+"_fastsim_p1_*.root")
		procP2=glob.glob(inputPath+folderName+"_fastsim_p2/"+folderName+"_fastsim_p2_*.root")
		procP3=glob.glob(inputPath+folderName+"_fastsim_p3/"+folderName+"_fastsim_p3_*.root")
		list=procP1+procP2+procP3
	else :
		procP1=glob.glob(inputPath+folderName+"_fastsim/"+folderName+"_fastsim_*.root")
		list=procP1
	print ("Date: ", time.asctime( time.localtime(time.time()) ))
	#dataCSVsort = pandas.DataFrame(columns=trainVars())
	lastevt=0
	countevt=0
	countevtMatch=0
	countevtSignal=0
	countAllEntries=0
	countmatch=0
	countentries=1
	countSignalChunck=0
	print len(list)
	# parse files in a way there will not be duplicates
	initfile= min(int(dofiles*len(list)/partition),len(list)-1)
	endfile= min(int((1+dofiles)*len(list)/partition),len(list)-1)
	print (initfile, endfile)
	for ii in range(initfile, endfile) : # initfile
		if countevtSignal > maxSignal : break
		lastfile= min(list[ii],len(list)-1)
		#print (list[ii],inputTree)
		try: tfile = ROOT.TFile(list[ii])
		except :
			print "Doesn't exist"
			print ('file ', list[ii],' corrupt')
			continue
		#if not if hasattr(tree0, 'Get') : # = tfile.Get(inputTree) :
		#	print ('file ', list[ii],' corrupt')
		try: tree0 = tfile.Get(inputTree)
		except :
			print "Doesn't exist"
			print ('file ', list[ii],' corrupt')
			continue
		try:
			n_entries= tree0.GetEntries()
		except :
			print "Doesn't exist"
			print ('file ', list[ii],' corrupt')
			continue
		else :
			print "Exists"
			n_entries= tree0.GetEntries()
		data = pandas.DataFrame() #
		for ev in range(0,10000000 ) : # n_entries ,  desc="{} ({} evts)".format(key, n_entries0)) :
			if ev % 100000 == 0 : print (folderName,ii,ev,countevtSignal, time.asctime( time.localtime(time.time()) ))
			if countevtSignal > maxSignal : break
			tree0.GetEntry(ev)
			if tree0.evt==lastevt: data=dataAppend(tree0,data,countevtMatch,folderName)
			else :
				if lastevt>0 and len(data)> 0:
					if doHist : histAllCombo.append(len(data))
					dataAll=dataAll.append(data, ignore_index=True)
					data = data.sort_values(by=['CSV_b'], ascending=False)
					if int(data['bWj1Wj2_isGenMatched'].sum()) >0 :
						if doHist :
							histOnlyMatchedCombo.append(len(data))
							#CSVposition=data.loc[data['bWj1Wj2_isGenMatched'].values > 0].index
							#print (CSVposition[0], data['CSV_b'].values[CSVposition[0]])
							CSVnoDuplicates= -np.unique(-data['CSV_b'].values)
							value_index = np.where(CSVnoDuplicates == data['CSV_b'].values[data.loc[data['bWj1Wj2_isGenMatched'].values > 0].index[0]] )
							#print data['CSV_b'].values
							#print CSVnoDuplicates
							#print value_index[0][0]
							histCSVStatus.append(value_index[0][0])
						countevtMatch =countevtMatch+1
						countevtSignal=countevtSignal+1 #int(data['bWj1Wj2_isGenMatched'].sum())
						dataStructured=dataStructured.append(data, ignore_index=True)
						"""
						# make CSV prior
						if len(data.loc[data['CSV_b'].values > 0.244]) >0 :
							datacsv=data.loc[data['CSV_b'].values > 0.244]
							dataCSVsort=dataCSVsort.append(datacsv, ignore_index=True)
							histOnlyMatchedCSVCombo.append(len(datacsv))
						else :
							dataCSVsort=dataCSVsort.append(data, ignore_index=True)
							histOnlyMatchedCSVCombo.append(len(data))
						#if doHist : histOnlyMatchedCSVCombo.append(len(dataCSVsort))
						"""
						# make CSV sort
						mincombo=min(4,len(CSVnoDuplicates)-1)  #.any()
						#ind = np.argpartition(data['CSV_b'].values, -5)[-5:]
						#datasort= data.iloc[ind]
						datasort= data.loc[data['CSV_b'].values > CSVnoDuplicates[mincombo]]
						dataCSVsort=dataCSVsort.append(datasort, ignore_index=True)
						if doHist :
							histOnlyMatchedCSVsortCombo.append(len(datasort))
					data = pandas.DataFrame()
					data=dataAppend(tree0,data,countevtMatch,folderName)
				lastevt=int(tree0.evt)
				countevt=countevt+1
		countAllEntries=countAllEntries+n_entries
		tfile.Close()
		removeN=7*len(dataAll.loc[dataAll['bWj1Wj2_isGenMatched'] == 0] )/10
		drop_indices = np.random.choice(dataAll.loc[dataAll['bWj1Wj2_isGenMatched'] == 0].index, int(removeN), replace=False)
		dataAll= dataAll.drop(drop_indices)
print (dataCSVsort.columns.values.tolist())


dataCSVsort['ncombo']=1
print ("Do combo row dataCSVsort: ", time.asctime( time.localtime(time.time()) ) )
for ii in range(int(dataCSVsort['eventRaw'].min()),int(dataCSVsort['eventRaw'].max())): dataCSVsort.loc[dataCSVsort['eventRaw']==ii,'ncombo'] = len(dataCSVsort.loc[dataCSVsort['eventRaw']==ii])
print ("Did combo row dataCSVsort: ", time.asctime( time.localtime(time.time()) ))


dataStructured['ncombo']=1
print ("Do combo row dataStructured: ", time.asctime( time.localtime(time.time()) ) )
for ii in range(int(dataStructured['eventRaw'].min()),int(dataStructured['eventRaw'].max())): dataStructured.loc[dataStructured['eventRaw']==ii,'ncombo'] = len(dataCSVsort.loc[dataCSVsort['eventRaw']==ii])
print ("Did combo row dataStructured: ", time.asctime( time.localtime(time.time()) ))

dataAll['ncombo']=1
print ("Do combo row dataAll: ", time.asctime( time.localtime(time.time()) ) )
for ii in range(int(dataAll['eventRaw'].min()),int(dataAll['eventRaw'].max())): dataAll.loc[dataAll['eventRaw']==ii,'ncombo'] = len(dataAll.loc[dataAll['eventRaw']==ii])
print ("Did combo row dataAll: ", time.asctime( time.localtime(time.time()) ))

if len(dataCSVsort) > 0 :
	target='bWj1Wj2_isGenMatched'
	print (folderName,"CSVprior bkg, signal, nevents",len(dataCSVsort.ix[dataCSVsort[target].values == 0]),len(dataCSVsort.ix[dataCSVsort[target].values > 0]),countevtMatch)
	print (folderName,"Tripletprior bkg, signal, nevents",len(dataStructured.ix[dataStructured[target].values == 0]),len(dataStructured.ix[dataStructured[target].values > 0]),countevtMatch)
	print (folderName,"dataAll bkg, signal, nevents",len(dataAll.ix[dataAll[target].values == 0]),len(dataAll.ix[dataAll[target].values > 0]),countevtMatch)

	print ("stopped: ",lastfile)
	c_size = 1000
	#dataCSVsort.to_csv(outfileCSV, chunksize=c_size)
	dataCSVsort.to_csv(outfileCSVsort, chunksize=c_size)
	dataStructured.to_csv(outfile, chunksize=c_size)
	dataAll.to_csv(outfileAll, chunksize=c_size)

	file = open(datasetsout+'/'+folderName+'_stats_from_'+str(maxSignal)+'sig.txt',"w")
	file.write("Process: "+str(folderName))
	file.write("Entries looped: "+str(countAllEntries))
	file.write("Inspected nev: "+str(countevt))
	file.write("Matched nev: "+str(countevtMatch))
	file.write("CSVsort: bkg, signal"+str(len(dataCSVsort.ix[dataCSVsort[target].values == 0]))+" "+str(len(dataCSVsort.ix[dataCSVsort[target].values > 0])))
	file.write("Tripletprior: bkg, signal"+str(len(dataStructured.ix[dataStructured[target].values == 0]))+" "+str(len(dataStructured.ix[dataStructured[target].values > 0])))
	file.write("No prior (all data): bkg, signal"+str(len(dataAll.ix[dataAll[target].values == 0]))+" "+str(len(dataAll.ix[dataAll[target].values > 0])))
	file.write(str(dataCSVsort.columns.values.tolist()))
	file.close()
else : print (lastfile)

#################################################################################
### Plot histograms

hist_params = {'normed': True, 'bins': 18, 'alpha': 0.4}
plt.figure(figsize=(20, 20))
Vars=dataAll.columns.values.tolist()
maxVar=[None] * len(Vars) # these will enter in the xml writting
minVar=[None] * len(Vars)
for n, feature in enumerate(Vars):
    # add sub plot on our figure
	plt.subplot(6,6, n+1)
    # define range for histograms by cutting 1% of data from both ends
	if ("cosTheta" in feature) or ("Dphi" in feature) : dataAll[feature] = dataAll[feature].abs()
	min_value, max_value = np.percentile(dataAll[feature], [1, 99])
	# this is to pass to xml reader
	maxVar[n]=max_value
	minVar[n]=min_value
	if 'qg_' in feature :
		min_value=0.0
		max_value=1.0
	print (min_value, max_value,feature)
	values, bins, _ = plt.hist(dataAll.ix[dataAll[target].values == 0, feature].values ,
                               range=(min_value, max_value),
							   label="BKG", **hist_params )
	values, bins, _ = plt.hist(dataAll.ix[dataAll[target].values == 1, feature].values,
                               range=(min_value, max_value), label='Signal', **hist_params)
	areaSig = sum(np.diff(bins)*values)
	#print areaBKG, " ",areaBKG2 ," ",areaSig
	if n == 0 : plt.legend(loc='best')
	plt.title(feature)
plt.savefig("structured/"+folderName+"_Variables_all.pdf")
plt.savefig("structured/"+folderName+"_Variables_all.png")
plt.clf()
##############################################################################
Vars=dataCSVsort.columns.values.tolist()
maxVar=[None] * len(Vars) # these will enter in the xml writting
minVar=[None] * len(Vars)
for n, feature in enumerate(Vars):
    # add sub plot on our figure
	plt.subplot(6,6, n+1)
    # define range for histograms by cutting 1% of data from both ends
	if ("cosTheta" in feature) or ("Dphi" in feature) : dataCSVsort[feature] = dataCSVsort[feature].abs()
	min_value, max_value = np.percentile(dataCSVsort[feature], [1, 99])
	# this is to pass to xml reader
	maxVar[n]=max_value
	minVar[n]=min_value
	if 'qg_' in feature :
		min_value=0.0
		max_value=1.0
	print (min_value, max_value,feature)
	values, bins, _ = plt.hist(dataCSVsort.ix[dataCSVsort[target].values == 0, feature].values ,
                               range=(min_value, max_value),
							   label="BKG", **hist_params )
	values, bins, _ = plt.hist(dataCSVsort.ix[dataCSVsort[target].values == 1, feature].values,
                               range=(min_value, max_value), label='Signal', **hist_params)
	areaSig = sum(np.diff(bins)*values)
	#print areaBKG, " ",areaBKG2 ," ",areaSig
	if n == 0 : plt.legend(loc='best')
	plt.title(feature)
plt.savefig("structured/"+folderName+"_Variables_CSVsort.pdf")
plt.savefig("structured/"+folderName+"_Variables_CSVsort.png")
plt.clf()

plt.figure(figsize=(12, 12))
if doHist :
	bin_size = 50;
	min_value, max_value = np.percentile(histAllCombo, [1, 99])
	N = (max_value-min_value)/bin_size; Nplus1 = N + 1

	print (min_value, max_value)


	#ax.set_ylim(1, 8000)
	bin_list = np.linspace(min_value, max_value, Nplus1)
	values, bins, _ = plt.hist(histAllCombo, bin_list , histtype='step', label='all combo, len='+str(int(countAllEntries)), color= 'r', edgecolor='r', lw=3)
	areaSig = sum(np.diff(bins)*values)
	values, bins, _ = plt.hist(histOnlyMatchedCombo, bin_list ,  histtype='step',
			label='only matched, len='+str(int(len(dataStructured)))+' (sig='+str(int(str(len(dataStructured.ix[dataStructured[target].values > 0]))))+')', color= 'g', edgecolor='g', lw=3)
	areaSigCombo = sum(np.diff(bins)*values)

	values2, bins2, _ = plt.hist(histOnlyMatchedCSVCombo, bin_list , histtype='step',
			label='with CSV prior, len='+str(int(len(dataCSVsort)))+' (sig='+str(int(str(len(dataCSVsort.ix[dataCSVsort[target].values > 0]))))+')', color= 'b', edgecolor='b', lw=3)
	areaSigComboCSV = sum(np.diff(bins2)*values2)
	values3, bins3, _ = plt.hist(histOnlyMatchedCSVsortCombo, bin_list , histtype='step',
			label='with CSV sort (4), len='+str(int(len(dataCSVsort)))+' (sig='+str(int(str(len(dataCSVsort.ix[dataCSVsort[target].values > 0]))))+')', color= 'y', edgecolor='y', lw=3)
	areaSigComboCSVsort = sum(np.diff(bins3)*values3)
	plt.xlim(min_value, max_value)
	plt.ylim(1, 8000)
	plt.yscale('log', nonposy='clip')
	#plt.hist(histAllCombo, bin_list , normed=1,  histtype='bar', label='all combo', color= 'r', edgecolor='r', lw=3)
	print (areaSig,areaSigCombo,areaSigComboCSV,areaSigComboCSVsort)
	plt.legend(loc='upper right')
	plt.title(" from "+str(maxSignal)+" evt" )
	#plt.xlabel("Mhh ()")
	plt.ylabel("n combo")
	plt.savefig("structured/numberOfCombos.pdf")
	#
	#"""
	#min_value, max_value = np.percentile(histCSVStatus, [1, 99])
	#print (
	plt.clf()
	min_value=0
	max_value=10
	bin_size=1
	N = (max_value-min_value)/bin_size; Nplus1 = N + 1
	bin_list = np.linspace(min_value, max_value, Nplus1)
	plt.yscale('linear', nonposy='clip')
	plt.xlim(0, 10)
	plt.ylim(0.0, 0.5)
	values, bins, _ = plt.hist(histCSVStatus, bin_list , normed=True ,  histtype= 'step', color= 'g', edgecolor='r', lw=3)
	plt.legend(loc='upper right')
	#plt.title(" In  kl =="+str(kl)+", kt =="+str(kt)+", c2 =="+str(c2)+", cg =="+str(cg)+", c2g ==" +str(c2g) )
	#plt.xlabel("Mhh (GeV)")
	plt.title(" from "+str(maxSignal)+" evt" )
	plt.xlabel('position of CSV')
	plt.ylabel('percentage')
	plt.savefig("structured/CSVposition.pdf")
	#"""


"""
removeN=5*len(dataCSVsort.loc[dataCSVsort[target] == 0] )/10
drop_indices = np.random.choice(dataCSVsort.loc[dataCSVsort[target] == 0].index, int(removeN), replace=False)
dataCSVsortBKG = dataCSVsort.loc[dataCSVsort[target] == 0].drop(drop_indices)
dataCSVsort = dataCSVsortBKG.append(dataCSVsort.loc[dataCSVsort[target].values > 0])
#the dataStructuredis just to later test

print (folderName,": CSVprior*1/10, total, bkg, signal, nevents",len(dataCSVsort) ,len(dataCSVsort.ix[dataCSVsort[target].values == 0]),len(dataCSVsort.ix[dataCSVsort[target].values > 0]),countevtMatch)
"""
