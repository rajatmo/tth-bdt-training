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

#import pylcgdict
#pylcgdict.loadDictionary('SealROOTDict')
#g=pylcgdict.Namespace("")

datasets = "/eos/cms/store/user/acarvalh/tth_toHepTopTagger/"
datasetsout = "/afs/cern.ch/work/a/acarvalh/codeCMSHHH4b/CMSSW_8_0_25/src/xgboost_test/"
#store = pandas.HDFStore(datasets+'/data_ttHToNonbb_CSV0244_p1.h5')



print ("Date: ", time.asctime( time.localtime(time.time()) ))
c_size = 1000
keys = ['ttHToNonbb','TTToSemilepton','TTZToLLNuNu']
counter=0
if 1> 0 : # key in keys : 
	key='ttHToNonbb'0'
	for typeF in [1,2,3] :
        
        #tfile2 = ROOT.TFile.Open(datasets+"dumb.root","RECREATE")
        #tree = ROOT.TTree()
    
        typeF=1
        # get entries
	if key=='ttHToNonbb' or key=='TTToSemilepton' : 
		f_name="structured_histograms_harvested_stage1_hadTopTagger_"+str(key)+"_fastsim_p"+str(typeF) #
		f_nameout="structured_histograms_harvested_stage1_hadTopTagger_"+str(key)+"_fastsim_p"+str(typeF)+"_less" #		
		if key=='ttHToNonbb' : t_name="analyze_hadTopTagger/evtntuple/signal/evtTree"
		elif key=='TTToSemilepton' : t_name="analyze_hadTopTagger/evtntuple/TT/evtTree" 
		
	elif key=='TTZToLLNuNu' and typeF==1 : 
		f_name=datasets+"structured_histograms_harvested_stage1_hadTopTagger_"+str(key)
		t_name="analyze_hadTopTagger/evtntuple/TTZ/evtTree"
	else :
		print ("regexp failed") 
		#break
	print(f_name)
	tfile = ROOT.TFile(datasets+f_name+".root")
	tree = tfile.Get("TCVARS")


	print ("Date: ", time.asctime( time.localtime(time.time()) ))
	n_entrie=tree.GetEntries()

	print ("Date: ", time.asctime( time.localtime(time.time()) ))
        #if typeF==1 : tree = tree1 else : tree = tree1 
        #tree.Draw('evt')
        n_entries = tree.GetEntries()
        print (n_entries,signal)
        # read schema from root and create table
        empty_arr = tree2array(tree0, stop = 0)
        empty_df = pandas.DataFrame(empty_arr)
        #store.put(key, empty_df, format='table', chunksize=c_size)
        #fileF=1
        #lastevt=0
        for start in trange(0, n_entries, c_size,  desc="{} ({} evts)".format(key, n_entries)) : #  
		#tree.GetEvent( start )
                #if tree.evt < lastevt : 
                #    fileF+=1 
                #    lastevt=0
                #else : lastevt+=1
		#if (tree.CSV_b>=0.244 or (tree.CSV_b<0.244 and tree.CSV_Wj1<0.244 and tree.CSV_Wj2<0.244)) : pass
		stop = start + c_size
		stop = stop if (stop < n_entries) else n_entries
		chunk_arr = tree2array(tree,  start=start, stop = stop)
		chunk_df = pandas.DataFrame(chunk_arr)
		chunk_df.index = pandas.Series(chunk_df.index) + start 
                chunk_df.to_csv(datasets+f_name+"_OnlyHadTopMatched_b-prior.csv",index=False,header=False,mode='a',chunksize=c_size)
		#store.append(key, chunk_df, chunksize=c_size)
        #counter+=n_entries0
        #if typeF==1 : counter=0
        tfile2.Close()
        print ("written "+datasets+f_name+"_OnlyHadTopMatched_b-prior.csv")
	# """
#store.close()

