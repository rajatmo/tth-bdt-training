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

datasets = " /hdfs/local/veelken/ttHAnalysis/2016/2017Aug31/histograms/hadTopTagger/"
datasetsout = "structured/"

print ("Date: ", time.asctime( time.localtime(time.time()) ))
c_size = 1000
keys = ['ttHToNonbb','TTToSemilepton','TTZToLLNuNu','TTWJetsToLNu']
counter=0

dotree=1


if 1> 0 : # key in keys : TTWJetsToLNu histograms_harvested_stage1_hadTopTagger_TTWJetsToLNu_fastsim.root
        key='TTToSemilepton'
        #for typeF in [1,2,3] :
        typeF=1
        # get entries
        print ("Date: ", time.asctime( time.localtime(time.time()) ))
	if key=='ttHToNonbb' or key=='TTToSemilepton' : 
		f_name="histograms_harvested_stage1_hadTopTagger_"+str(key)+"_fastsim_p"+str(typeF)+".root" #  
		if key=='ttHToNonbb' : t_name="analyze_hadTopTagger/evtntuple/signal/evtTree"
		else: t_name="analyze_hadTopTagger/evtntuple/TT/evtTree" 
	elif key=='TTZToLLNuNu' : 
		f_name="histograms_harvested_stage1_hadTopTagger_TTZToLLNuNu_fastsim.root"
		t_name="analyze_hadTopTagger/evtntuple/TTZ/evtTree"
	elif key=='TTWJetsToLNu' : 
		f_name="histograms_harvested_stage1_hadTopTagger_TTWJetsToLNu_fastsim.root"
		t_name="analyze_hadTopTagger/evtntuple/TTW/evtTree"
	else :
		print ("regexp failed") 
		pass
	print(f_name)
        #
        tfile2 = ROOT.TFile.Open(datasetsout+"structured_"+f_name,"RECREATE")
        tree=ROOT.TTree('TCVARS', 'TCVARS' )
		
	tfile = ROOT.TFile(datasets+f_name)
	tree0 = tfile.Get(t_name)
        #if dotree==1 :

        #CSVb = ROOT.std.vector( float )() # array.array( 'f', 4000*[ -1000.0 ] )
        bWj1Wj2_isGenMatched = ROOT.std.vector( int )()
        b_isGenMatched = ROOT.std.vector( int )()
        Wj1_isGenMatched = ROOT.std.vector( int )()
        Wj2_isGenMatched = ROOT.std.vector( int )()
        #statusKinFit = ROOT.std.vector( float )()
        #qg_b = ROOT.std.vector( float )()
        qg_Wj2 = ROOT.std.vector( float )()
        qg_Wj1 = ROOT.std.vector( float )()
        pT_bWj1Wj2 = ROOT.std.vector( float )()
        pT_b = ROOT.std.vector( float )()
        pT_Wj2 = ROOT.std.vector( float )()
        pT_Wj1Wj2 = ROOT.std.vector( float )()
        pT_Wj1 = ROOT.std.vector( float )()
        nllKinFit = ROOT.std.vector( float )()
        #max_dR_div_expRjet = ROOT.std.vector( float )()
        #m_bWj2 = ROOT.std.vector( float )()
        m_bWj1Wj2 = ROOT.std.vector( float )()
        #m_bWj1 = ROOT.std.vector( float )()
        #m_Wj1Wj2_div_m_bWj1Wj2 = ROOT.std.vector( float )()
        m_Wj1Wj2 = ROOT.std.vector( float )()
        #logPKinFit = ROOT.std.vector( float )()
        #logPErrKinFit = ROOT.std.vector( float )()
        #dR_bWj2 = ROOT.std.vector( float )()
        #dR_bWj1 = ROOT.std.vector( float )()
        dR_bW = ROOT.std.vector( float )()
        dR_Wj1Wj2 = ROOT.std.vector( float )()
        alphaKinFit = ROOT.std.vector( float )()
        CSV_b = ROOT.std.vector( float )()
        #CSV_Wj1 = ROOT.std.vector( float )()
        #CSV_Wj2 = ROOT.std.vector( float )()
        #
        #tree.Branch('CSVb', CSVb) 
        tree.Branch('bWj1Wj2_isGenMatched', bWj1Wj2_isGenMatched) 
        tree.Branch('b_isGenMatched', b_isGenMatched) 
        tree.Branch('Wj1_isGenMatched', Wj1_isGenMatched) 
        tree.Branch('Wj2_isGenMatched', Wj1_isGenMatched) 
        #tree.Branch('statusKinFit', statusKinFit) 
        #tree.Branch('qg_b', qg_b) 
        tree.Branch('qg_Wj2', qg_Wj2) 
        tree.Branch('qg_Wj1', qg_Wj1) 
        tree.Branch('pT_bWj1Wj2', pT_bWj1Wj2) 
        tree.Branch('pT_b', pT_b) 
        tree.Branch('pT_Wj2', pT_Wj2) 
        tree.Branch('pT_Wj1Wj2', pT_Wj1Wj2) 
        tree.Branch('pT_Wj1', pT_Wj1) 
        tree.Branch('nllKinFit', nllKinFit) 
        #tree.Branch('max_dR_div_expRjet', max_dR_div_expRjet) 
        #tree.Branch('m_bWj2', m_bWj2) 
        tree.Branch('m_bWj1Wj2', m_bWj1Wj2) 
        #tree.Branch('m_bWj1', m_bWj1) 
        #tree.Branch('m_Wj1Wj2_div_m_bWj1Wj2', m_Wj1Wj2_div_m_bWj1Wj2) 
        tree.Branch('m_Wj1Wj2', m_Wj1Wj2) 
        #tree.Branch('logPKinFit', logPKinFit) 
        #tree.Branch('logPErrKinFit', logPErrKinFit) 
        #tree.Branch('dR_bWj2', dR_bWj2) 
        #tree.Branch('dR_bWj1', dR_bWj1) 
        tree.Branch('dR_bW', dR_bW) 
        tree.Branch('dR_Wj1Wj2', dR_Wj1Wj2) 
        tree.Branch('alphaKinFit', alphaKinFit) 
        tree.Branch('CSV_b', CSV_b) 
        #tree.Branch('CSV_Wj1', CSV_Wj1)
        #tree.Branch('CSV_Wj2', CSV_Wj2)


        evt = array.array( 'i', [ 0 ] )
        ncomb = array.array( 'i', [ 0 ] )
        haveMatch = array.array( 'i', [ 0 ] )
        tree.Branch('evt', evt, 'evt/I')
        tree.Branch('ncomb', ncomb, 'ncomb/I')
        tree.Branch('haveMatch', haveMatch, 'haveMatch/I')
        print ("Date: ", time.asctime( time.localtime(time.time()) ))
        print ("Dimensions",int(tree0.GetMaximum('evt')),\
                            int(tree0.GetMaximum('run')),int(tree0.GetMaximum('lumi')))
        #if typeF==1 : 
        #elif typeF==2 : 
        allcombo=np.empty((int(tree0.GetMaximum('evt')+1),\
                                        int(tree0.GetMaximum('run')+1),int(tree0.GetMaximum('lumi')+1)))
        n_entries0= tree0.GetEntries()
        tree0.GetEntry(0)
        print (tree0.evt)

        bWj1Wj2_isGenMatched.push_back(tree0.bWj1Wj2_isGenMatched)
        b_isGenMatched.push_back(tree0.b_isGenMatched)
        Wj1_isGenMatched.push_back(tree0.Wj1_isGenMatched)
        Wj2_isGenMatched.push_back(tree0.Wj2_isGenMatched)
        #statusKinFit.push_back(tree0.statusKinFit)
        #qg_b.push_back(tree0.qg_b)
        qg_Wj2.push_back(tree0.qg_Wj2)
        qg_Wj1.push_back(tree0.qg_Wj1)
        pT_bWj1Wj2.push_back(tree0.pT_bWj1Wj2)
        pT_b.push_back(tree0.pT_b)
        pT_Wj2.push_back(tree0.pT_Wj2)
        pT_Wj1Wj2.push_back(tree0.pT_Wj1Wj2)
        pT_Wj1.push_back(tree0.pT_Wj1)
        nllKinFit.push_back(tree0.nllKinFit)
        #max_dR_div_expRjet.push_back(tree0.max_dR_div_expRjet)
        #m_bWj2.push_back(tree0.m_bWj2)
        m_bWj1Wj2.push_back(tree0.m_bWj1Wj2)
        #m_bWj1.push_back(tree0.m_bWj1)
        #m_Wj1Wj2_div_m_bWj1Wj2.push_back(tree0.m_Wj1Wj2_div_m_bWj1Wj2)
        m_Wj1Wj2.push_back(tree0.m_Wj1Wj2)
        #logPKinFit.push_back(tree0.logPKinFit)
        #logPErrKinFit.push_back(tree0.logPErrKinFit)
        #dR_bWj2.push_back(tree0.dR_bWj2)
        #dR_bWj1.push_back(tree0.dR_bWj1)
        dR_bW.push_back(tree0.dR_bW)
        dR_Wj1Wj2.push_back(tree0.dR_Wj1Wj2)
        alphaKinFit.push_back(tree0.alphaKinFit)
        CSV_b.push_back(tree0.CSV_b)
        #CSV_Wj1.push_back(tree0.CSV_Wj1)
        #CSV_Wj2.push_back(tree0.CSV_Wj2)
        #
        lastevt=tree0.evt
        countevt=0
        countcomb=1
        countmatch=tree0.b_isGenMatched
        countentries=1
        for ev in trange(1,n_entries0,  desc="{} ({} evts)".format(key, n_entries0)) :
		tree0.GetEntry(ev)
                if 1>0 : 
                  #if tree0.CSV_b > 0.5 : # and tree0.alphaKinFit > 0.5 and tree0.CSV_b < 1.8  :
                  #    
                  #countcombo=0
                  if dotree==1 :
                   if tree0.evt==lastevt:
                     if tree0.bWj1Wj2_isGenMatched >0 : countmatch=countmatch+1
                     bWj1Wj2_isGenMatched.push_back(tree0.bWj1Wj2_isGenMatched)
        	     b_isGenMatched.push_back(tree0.b_isGenMatched)
                     Wj1_isGenMatched.push_back(tree0.Wj1_isGenMatched)
                     Wj2_isGenMatched.push_back(tree0.Wj2_isGenMatched)
                     #statusKinFit.push_back(tree0.statusKinFit)
                     #qg_b.push_back(tree0.qg_b)
                     qg_Wj2.push_back(tree0.qg_Wj2)
                     qg_Wj1.push_back(tree0.qg_Wj1)
                     pT_bWj1Wj2.push_back(tree0.pT_bWj1Wj2)
                     pT_b.push_back(tree0.pT_b)
                     pT_Wj2.push_back(tree0.pT_Wj2)
                     pT_Wj1Wj2.push_back(tree0.pT_Wj1Wj2)
                     pT_Wj1.push_back(tree0.pT_Wj1)
                     nllKinFit.push_back(tree0.nllKinFit)
                     #max_dR_div_expRjet.push_back(tree0.max_dR_div_expRjet)
                     #m_bWj2.push_back(tree0.m_bWj2)
                     m_bWj1Wj2.push_back(tree0.m_bWj1Wj2)
                     #m_bWj1.push_back(tree0.m_bWj1)
                     #m_Wj1Wj2_div_m_bWj1Wj2.push_back(tree0.m_Wj1Wj2_div_m_bWj1Wj2)
                     m_Wj1Wj2.push_back(tree0.m_Wj1Wj2)
                     #logPKinFit.push_back(tree0.logPKinFit)
                     #logPErrKinFit.push_back(tree0.logPErrKinFit)
                     #dR_bWj2.push_back(tree0.dR_bWj2)
                     #dR_bWj1.push_back(tree0.dR_bWj1)
                     dR_bW.push_back(tree0.dR_bW)
                     dR_Wj1Wj2.push_back(tree0.dR_Wj1Wj2)
                     alphaKinFit.push_back(tree0.alphaKinFit)
                     CSV_b.push_back(tree0.CSV_b)
                     #CSV_Wj1.push_back(tree0.CSV_Wj1)
                     #CSV_Wj2.push_back(tree0.CSV_Wj2)
                     countcomb=countcomb+1
                   else : 
                     evt[0]=countevt
                     ncomb[0]=CSV_b.size()
                     haveMatch[0]=countmatch
                     if countmatch>0 : # countmatch==1: #bWj1Wj2_isGenMatched.compress(bWj1Wj2_isGenMatched>0).size()>0: 
                        tree.Fill()
                        allcombo[int(tree0.evt)][int(tree0.run)][int(tree0.lumi)]=allcombo[int(tree0.evt)][int(tree0.run)][int(tree0.lumi)]+1
                        countentries=countentries+ncomb[0]
                     countmatch=0
                     #CSVb.clear()
                     bWj1Wj2_isGenMatched.clear()
        	     b_isGenMatched.clear()
                     Wj1_isGenMatched.clear()
                     Wj2_isGenMatched.clear()
                     #statusKinFit.clear()
                     #qg_b.clear()
                     qg_Wj2.clear()
                     qg_Wj1.clear()
                     pT_bWj1Wj2.clear()
                     pT_b.clear()
                     pT_Wj2.clear()
                     pT_Wj1Wj2.clear()
                     pT_Wj1.clear()
                     nllKinFit.clear()
                     #max_dR_div_expRjet.push_back(tree0.max_dR_div_expRjet)
                     #m_bWj2.push_back(tree0.m_bWj2)
                     m_bWj1Wj2.clear()
                     #m_bWj1.push_back(tree0.m_bWj1)
                     #m_Wj1Wj2_div_m_bWj1Wj2.push_back(tree0.m_Wj1Wj2_div_m_bWj1Wj2)
                     m_Wj1Wj2.clear()
                     #logPKinFit.push_back(tree0.logPKinFit)
                     #logPErrKinFit.push_back(tree0.logPErrKinFit)
                     #dR_bWj2.push_back(tree0.dR_bWj2)
                     #dR_bWj1.push_back(tree0.dR_bWj1)
                     dR_bW.clear()
                     dR_Wj1Wj2.clear()
                     alphaKinFit.clear()
                     CSV_b.clear()
                     #CSV_Wj1.push_back(tree0.CSV_Wj1)
                     #CSV_Wj2.push_back(tree0.CSV_Wj2)
                     if tree0.bWj1Wj2_isGenMatched >0 : countmatch=countmatch+1
                     bWj1Wj2_isGenMatched.push_back(tree0.bWj1Wj2_isGenMatched)
        	     b_isGenMatched.push_back(tree0.b_isGenMatched)
                     Wj1_isGenMatched.push_back(tree0.Wj1_isGenMatched)
                     Wj2_isGenMatched.push_back(tree0.Wj2_isGenMatched)
                     #statusKinFit.push_back(tree0.statusKinFit)
                     #qg_b.push_back(tree0.qg_b)
                     qg_Wj2.push_back(tree0.qg_Wj2)
                     qg_Wj1.push_back(tree0.qg_Wj1)
                     pT_bWj1Wj2.push_back(tree0.pT_bWj1Wj2)
                     pT_b.push_back(tree0.pT_b)
                     pT_Wj2.push_back(tree0.pT_Wj2)
                     pT_Wj1Wj2.push_back(tree0.pT_Wj1Wj2)
                     pT_Wj1.push_back(tree0.pT_Wj1)
                     nllKinFit.push_back(tree0.nllKinFit)
                     #max_dR_div_expRjet.push_back(tree0.max_dR_div_expRjet)
                     #m_bWj2.push_back(tree0.m_bWj2)
                     m_bWj1Wj2.push_back(tree0.m_bWj1Wj2)
                     #m_bWj1.push_back(tree0.m_bWj1)
                     #m_Wj1Wj2_div_m_bWj1Wj2.push_back(tree0.m_Wj1Wj2_div_m_bWj1Wj2)
                     m_Wj1Wj2.push_back(tree0.m_Wj1Wj2)
                     #logPKinFit.push_back(tree0.logPKinFit)
                     #logPErrKinFit.push_back(tree0.logPErrKinFit)
                     #dR_bWj2.push_back(tree0.dR_bWj2)
                     #dR_bWj1.push_back(tree0.dR_bWj1)
                     dR_bW.push_back(tree0.dR_bW)
                     dR_Wj1Wj2.push_back(tree0.dR_Wj1Wj2)
                     alphaKinFit.push_back(tree0.alphaKinFit)
                     CSV_b.push_back(tree0.CSV_b)
                     #CSV_Wj1.push_back(tree0.CSV_Wj1)
                     #CSV_Wj2.push_back(tree0.CSV_Wj2)
                     countcomb=1
                     countevt=countevt+1
                     lastevt=tree0.evt
        tfile.Close()
        print ("Date: ", time.asctime( time.localtime(time.time()) ))
        print (countentries) # 248148
        #if dotree==1 :
        tree.Write()
        tfile2.Close()
allflat=allcombo.flatten()
allflatEvt=allcombo.compress(allflat>0)
print (int(len(allflatEvt)))
#####################################################33

hist1D = ROOT.TH1F('Name', 'Triplets/event (only matched)',int(len(allflatEvt))-1, 1, int(len(allflatEvt)))
canv= ROOT.TCanvas()
for entry in trange(0,int(len(allflatEvt)),  desc=" ({} evts)".format( n_entries0)): 
    #print (allflat[entry])
    hist1D.SetBinContent(entry,allflatEvt[entry])        
hist1D.Draw()
canv.SaveAs('comboPerEvent_brute_onlyMatch.pdf','pdf')
canv.SaveAs('comboPerEvent_brute_onlyMatch.png','png')



