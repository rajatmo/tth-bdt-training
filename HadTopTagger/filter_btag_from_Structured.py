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

datasets = "/hdfs/local/acaan/HadTopTagger/2017Aug31/"
datasetsout = "/hdfs/local/acaan/HadTopTagger/2017Aug31/"
#store = pandas.HDFStore(datasets+'/data_ttHToNonbb_CSV0244_p1.h5')

evaluateBDT=True
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
##################################################################
### pickle 
if evaluateBDT==True :
	# HadTopTagger/HadTopTagger_XGB_allVar_lessBKG_CSV_screening.pkl
	loaded_model = pickle.load(open('HadTopTagger_baseline_1000trees_allBKG//HadTopTagger_XGB_allVar_allBKG_CSV_screening.pkl', 'rb'))
	target="bWj1Wj2_isGenMatched"

print ("Date: ", time.asctime( time.localtime(time.time()) ))
c_size = 1000
keys = ['ttHToNonbb','TTToSemilepton','TTZToLLNuNu']
counter=0
if 1> 0 : # key in keys : 
        key='ttHToNonbb'
        #tfile2 = ROOT.TFile.Open(datasets+"dumb.root","RECREATE")
        #tree = ROOT.TTree()
        #for typeF in [1,2,3] :
        typeF=3
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
	tree0 = tfile.Get("TCVARS")
	
	if evaluateBDT==True : filename="dumb.root"
	else : "structured/"+f_nameout+".root"
	tfile2 = ROOT.TFile.Open(filename,"RECREATE")
	tree=ROOT.TTree("TCVARSbfilter","TCVARSbfilter") 
        # """
        print (tree0.GetEntries(),typeF)
        #if typeF == 1 :


        #evt = array.array( 'i', [ 0 ] )
        #ncomb = array.array( 'i', [ 0 ] )
        #haveMatch = array.array( 'i', [ 0 ] )
        CSV_b = array.array( 'f', [ 0 ] ) 
        bWj1Wj2_isGenMatched = array.array( 'i', [ 0 ] )
        #"""
        b_isGenMatched = array.array( 'i', [ 0 ] )
        Wj1_isGenMatched = array.array( 'i', [ 0 ] )
        Wj2_isGenMatched = array.array( 'i', [ 0 ] )
        #statusKinFit = array.array( 'i', [ 0 ] )
        #qg_b = array.array( 'f', [ 0 ] )
        qg_Wj2 = array.array( 'f', [ 0 ] )
        qg_Wj1 = array.array( 'f', [ 0 ] )
        pT_bWj1Wj2 = array.array( 'f', [ 0 ] )
        pT_b = array.array( 'f', [ 0 ] )
        pT_Wj2 = array.array( 'f', [ 0 ] )
        pT_Wj1Wj2 = array.array( 'f', [ 0 ] )
        pT_Wj1 = array.array( 'f', [ 0 ] )
        nllKinFit = array.array( 'f', [ 0 ] )
        #max_dR_div_expRjet = array.array( 'f', [ 0 ] )
        #m_bWj2 = array.array( 'f', [ 0 ] )
        m_bWj1Wj2 = array.array( 'f', [ 0 ] )
        #m_bWj1 = array.array( 'f', [ 0 ] )
        #m_Wj1Wj2_div_m_bWj1Wj2 = array.array( 'f', [ 0 ] )
        m_Wj1Wj2 = array.array( 'f', [ 0 ] )
        #logPKinFit = array.array( 'f', [ 0 ] )
        #logPErrKinFit = array.array( 'f', [ 0 ] )
        #dR_bWj2 = array.array( 'f', [ 0 ] )
        #dR_bWj1 = array.array( 'f', [ 0 ] )
        dR_bW = array.array( 'f', [ 0 ] )
        dR_Wj1Wj2 = array.array( 'f', [ 0 ] )
        alphaKinFit = array.array( 'f', [ 0 ] )
        #CSV_Wj1 = array.array( 'f', [ 0 ] )
        #CSV_Wj2 = array.array( 'f', [ 0 ] )
        #"""

        #tree.Branch('evt', evt, 'evt/I')
        #tree.Branch('ncomb', ncomb, 'ncomb/I')
        #tree.Branch('haveMatch', haveMatch, 'haveMatch/I')
        tree.Branch('CSV_b', CSV_b, 'CSV_b/F') 
        tree.Branch('bWj1Wj2_isGenMatched', bWj1Wj2_isGenMatched, 'bWj1Wj2_isGenMatched/I') 
        #"""
        tree.Branch('b_isGenMatched', b_isGenMatched, 'b_isGenMatched/I') 
        tree.Branch('Wj1_isGenMatched', Wj1_isGenMatched, 'Wj2_isGenMatched/I') 
        tree.Branch('Wj2_isGenMatched', Wj1_isGenMatched, 'Wj1_isGenMatched/I') 
        #tree.Branch('statusKinFit', statusKinFit) 
        #tree.Branch('qg_b', qg_b) 
        tree.Branch('qg_Wj2', qg_Wj2, 'qg_Wj2/F') 
        tree.Branch('qg_Wj1', qg_Wj1, 'qg_Wj1/F') 
        tree.Branch('pT_bWj1Wj2', pT_bWj1Wj2, 'pT_bWj1Wj2/F') 
        tree.Branch('pT_b', pT_b, 'pT_b/F') 
        tree.Branch('pT_Wj2', pT_Wj2, 'pT_Wj2/F') 
        tree.Branch('pT_Wj1Wj2', pT_Wj1Wj2, 'pT_Wj1Wj2/F') 
        tree.Branch('pT_Wj1', pT_Wj1, 'pT_Wj1/F') 
        tree.Branch('nllKinFit', nllKinFit, 'nllKinFit/F') 
        #tree.Branch('max_dR_div_expRjet', max_dR_div_expRjet) 
        #tree.Branch('m_bWj2', m_bWj2) 
        tree.Branch('m_bWj1Wj2', m_bWj1Wj2, 'm_bWj1Wj2/F') 
        #tree.Branch('m_bWj1', m_bWj1) 
        #tree.Branch('m_Wj1Wj2_div_m_bWj1Wj2', m_Wj1Wj2_div_m_bWj1Wj2) 
        tree.Branch('m_Wj1Wj2', m_Wj1Wj2, 'm_Wj1Wj2/F') 
        #tree.Branch('logPKinFit', logPKinFit) 
        #tree.Branch('logPErrKinFit', logPErrKinFit) 
        #tree.Branch('dR_bWj2', dR_bWj2) 
        #tree.Branch('dR_bWj1', dR_bWj1) 
        tree.Branch('dR_bW', dR_bW, 'dR_bW/F') 
        tree.Branch('dR_Wj1Wj2', dR_Wj1Wj2, 'dR_Wj1Wj2/F') 
        tree.Branch('alphaKinFit', alphaKinFit, 'alphaKinFit/F') 
        #tree.Branch('CSV_Wj1', CSV_Wj1)
        #tree.Branch('CSV_Wj2', CSV_Wj2)
        #"""
    	#tree = ROOT.TTree()
	print ("Date: ", time.asctime( time.localtime(time.time()) ))
	if evaluateBDT==True : n_entries0=100000 
	else : n_entries0=tree0.GetEntries()
        signal=0
        bkg=0
	counttrue=0
	for ev in trange(0,n_entries0): #, desc="{} ({})".format(key, n_entries0)) :
            tree0.GetEntry(ev) 
            if tree0.haveMatch >0 : # or (tree.CSV_b<0.244 and tree.CSV_Wj1<0.244 and tree.CSV_Wj2<0.244)
				bindices= []
				for ii in range (0,len(tree0.bWj1Wj2_isGenMatched)) : 
					if tree0.CSV_b[ii] >0.244 : bindices.append(ii)
				if len(bindices) < 1 : 
					for ii in range (0,len(tree0.bWj1Wj2_isGenMatched)) :  bindices.append(ii)
				if evaluateBDT==True : 
					 	data = pandas.DataFrame()
						#data = data.assign(CSV_b=pandas.Series(tree0.CSV_b).values)
						data['CSV_b'] = pandas.Series(np.take(np.array(tree0.CSV_b), bindices))
						data['alphaKinFit'] = pandas.Series(np.take(np.array(tree0.alphaKinFit), bindices) )
						data['dR_Wj1Wj2'] = pandas.Series(np.take(np.array(tree0.dR_Wj1Wj2), bindices) )
						data['dR_bW'] = pandas.Series(np.take(np.array(tree0.dR_bW), bindices) )
						data['m_Wj1Wj2'] = pandas.Series(np.take(np.array(tree0.m_Wj1Wj2), bindices) )
						data['m_bWj1Wj2'] = pandas.Series(np.take(np.array(tree0.m_bWj1Wj2), bindices) )
						data['nllKinFit'] = pandas.Series(np.take(np.array(tree0.nllKinFit), bindices) )
						data['pT_Wj1'] = pandas.Series(np.take(np.array(tree0.pT_Wj1), bindices) )
						data['pT_Wj1Wj2'] = pandas.Series(np.take(np.array(tree0.pT_Wj1Wj2), bindices) )
						data['pT_Wj2'] = pandas.Series(np.take(np.array(tree0.pT_Wj2), bindices) )
						data['pT_b'] = pandas.Series(np.take(np.array(tree0.pT_b), bindices) )
						data['pT_bWj1Wj2'] = pandas.Series(np.take(np.array(tree0.pT_bWj1Wj2), bindices) )
						data['qg_Wj1'] = pandas.Series(np.take(np.array(tree0.qg_Wj1), bindices) )
						data['qg_Wj2'] = pandas.Series(np.take(np.array(tree0.qg_Wj2), bindices))
						data['bWj1Wj2_isGenMatched'] = pandas.Series(np.take(np.array(tree0.bWj1Wj2_isGenMatched), bindices) )
						proba = loaded_model.predict_proba(data[trainVars()].values)
						if data['bWj1Wj2_isGenMatched'][np.argmax(proba[:,1])] >0 : counttrue= counttrue+1
				else :
					for ii in range(0,len(bindices)) :# range(0,len(tree0.bWj1Wj2_isGenMatched)) :
					 bWj1Wj2_isGenMatched[0] = int(tree0.bWj1Wj2_isGenMatched[bindices[ii]])
					 if tree0.haveMatch == 0 : bkg=bkg+1
					 if bkg < 10 : pass
					 else : bkg=0
                     #print (ii,tree0.bWj1Wj2_isGenMatched[bindices[ii]])
					 signal=signal+int(tree0.haveMatch)
					 b_isGenMatched[0] = int(tree0.b_isGenMatched[bindices[ii]])
					 Wj1_isGenMatched[0] = int(tree0.Wj1_isGenMatched[bindices[ii]])
					 Wj2_isGenMatched[0] = int(tree0.Wj2_isGenMatched[bindices[ii]])
                     #statusKinFit[0] = tree0.statusKinFit
                     #qg_b[0] = tree0.qg_b
					 qg_Wj2[0] = tree0.qg_Wj2[bindices[ii]]
					 qg_Wj1[0] = tree0.qg_Wj1[bindices[ii]]
					 pT_bWj1Wj2[0] = tree0.pT_bWj1Wj2[bindices[ii]]
					 pT_b[0] = tree0.pT_b[bindices[ii]]
					 pT_Wj2[0] = tree0.pT_Wj2[bindices[ii]]
					 pT_Wj1Wj2[0] = tree0.pT_Wj1Wj2[bindices[ii]]
					 pT_Wj1[0] = tree0.pT_Wj1[bindices[ii]]
					 nllKinFit[0] = tree0.nllKinFit[bindices[ii]]
                     #max_dR_div_expRjet[0] = tree0.max_dR_div_expRjet
                     #m_bWj2[0] = tree0.m_bWj2
					 m_bWj1Wj2[0] = tree0.m_bWj1Wj2[bindices[ii]]
                     #m_bWj1[0] = tree0.m_bWj1
                     #m_Wj1Wj2_div_m_bWj1Wj2[0] = tree0.m_Wj1Wj2_div_m_bWj1Wj2
					 m_Wj1Wj2[0] = tree0.m_Wj1Wj2[bindices[ii]]
                     #logPKinFit[0] = tree0.logPKinFit
                     #logPErrKinFit[0] = tree0.logPErrKinFit
                     #dR_bWj2[0] = tree0.dR_bWj2
                     #dR_bWj1[0] = tree0.dR_bWj1
					 dR_bW[0] = tree0.dR_bW[bindices[ii]]
					 dR_Wj1Wj2[0] = tree0.dR_Wj1Wj2[bindices[ii]]
					 alphaKinFit[0] = tree0.alphaKinFit[bindices[ii]]
					 CSV_b[0] = tree0.CSV_b[bindices[ii]]
                     #CSV_Wj1[0] = tree0.CSV_Wj1
                     #CSV_Wj2[0] = tree0.CSV_Wj2
					 tree.Fill()

	tfile.Close()
	tree.Write()
	tfile2.Close()
	print (counttrue,n_entries0, float(counttrue)/float(n_entries0))
	"""
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
