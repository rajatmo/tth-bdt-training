import itertools as it
import numpy as np
from root_numpy import root2array, stretch
from numpy.lib.recfunctions import append_fields
from itertools import product
from ROOT.Math import PtEtaPhiEVector,VectorUtil

def load_data(inputPath,channelInTree,variables,criteria,testtruth,bdtType) :
    print variables
    my_cols_list=variables+['key','target','file']+criteria #,'tau_frWeight','lep1_frWeight','lep1_frWeight' trainVars(False)
    # if channel=='2lss_1tau' : my_cols_list=my_cols_list+['tau_frWeight','lep1_frWeight','lep2_frWeight']
    # those last are only for channels where selection is relaxed (2lss_1tau) === solve later
    data = pandas.DataFrame(columns=my_cols_list)
    if bdtType=="evtLevelTT_TTH" : keys=['ttHToNonbb','TTTo2L2Nu','TTToSemilepton']
    if bdtType=="evtLevelTTV_TTH" : keys=['ttHToNonbb','TTZToLLNuNu','TTWJetsToLNu']
    if bdtType=="all" : keys=['ttHToNonbb','TTZToLLNuNu','TTWJetsToLNu','TTTo2L2Nu','TTToSemilepton']
    if bdtType=="arun" : keys=['ttHToNonbb','TTZToLLNuNu','TTWJetsToLNu','TTTo2L2Nu','TTToSemilepton']
    for folderName in keys :
    	print (folderName, channelInTree)
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
        if bdtType!="arun" :
        	if ('TTT' in folderName) or folderName=='ttHToNonbb' :
        		procP1=glob.glob(inputPath+"/"+folderName+"_fastsim_p1/"+folderName+"_fastsim_p1_forBDTtraining*OS_central_*.root")
        		procP2=glob.glob(inputPath+"/"+folderName+"_fastsim_p2/"+folderName+"_fastsim_p2_forBDTtraining*OS_central_*.root")
        		procP3=glob.glob(inputPath+"/"+folderName+"_fastsim_p3/"+folderName+"_fastsim_p3_forBDTtraining*OS_central_*.root")
        		list=procP1+procP2+procP3
        	else :
        		procP1=glob.glob(inputPath+"/"+folderName+"_fastsim/"+folderName+"_fastsim_forBDTtraining*OS_central_*.root")
        		list=procP1
        else : list=["arun_xml_2lss_1tau/ntuple_2lss_1tau_SS_OS_all.root"]
    	#print ("Date: ", time.asctime( time.localtime(time.time()) ))
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
    				#if ii ==0 : print (chunk_df.columns.values.tolist())
    				chunk_df['key']=folderName
    				chunk_df['target']=target
    				#chunk_df['file']=list[ii].split("_")[10]
    				if channel=="2lss_1tau" and bdtType!="arun" :
    					chunk_df["totalWeight"] = chunk_df["evtWeight"]*chunk_df['tau_frWeight']*chunk_df['lep1_frWeight']*chunk_df['lep2_frWeight']
    				if channel=="1l_2tau" : chunk_df["totalWeight"] = chunk_df.evtWeight
    				###########
    				if channel=="2lss_1tau"  and len(criteria)>0:
    					data=data.append(chunk_df.ix[chunk_df.failsTightChargeCut.values == 0], ignore_index=True)
    				else : #
    					#if 1>0 :
    					data=data.append(chunk_df, ignore_index=True)
    		else : print ("file "+list[ii]+"was empty")
    		tfile.Close()
    	if len(data) == 0 : continue
    	nS = len(data.ix[(data.target.values == 0) & (data.key.values==folderName)])
    	nB = len(data.ix[(data.target.values == 1) & (data.key.values==folderName)])
    	print folderName,"length of sig, bkg: ", nS, nB
    	if (channel=="1l_2tau" or channel=="2lss_1tau") and bdtType!="arun" :
    		nSthuth = len(data.ix[(data.target.values == 0) & (data.bWj1Wj2_isGenMatched.values==1) & (data.key.values==folderName)])
    		nBtruth = len(data.ix[(data.target.values == 1) & (data.bWj1Wj2_isGenMatched.values==1) & (data.key.values==folderName)])
    		nSthuthKin = len(data.ix[(data.target.values == 0) & (data.bWj1Wj2_isGenMatchedWithKinFit.values==1) & (data.key.values==folderName)])
    		nBtruthKin = len(data.ix[(data.target.values == 1) & (data.bWj1Wj2_isGenMatchedWithKinFit.values==1) & (data.key.values==folderName)])
    		nShadthuth = len(data.ix[(data.target.values == 0) & (data.hadtruth.values==1) & (data.key.values==folderName)])
    		nBhadtruth = len(data.ix[(data.target.values == 1) & (data.hadtruth.values==1) & (data.key.values==folderName)])
    		print "truth:              ", nSthuth, nBtruth
    		print "truth Kin:          ", nSthuthKin, nBtruthKin
    		print "hadtruth:           ", nShadthuth, nBhadtruth
    if folderName=='ttHToNonbb' : print (data.columns.values.tolist())
    n = len(data)
    nS = len(data.ix[data.target.values == 0])
    nB = len(data.ix[data.target.values == 1])
    print channelInTree," length of sig, bkg: ", nS, nB
    #print ("weigths", data.loc[data['target']==0]["totalWeight"].sum() , data.loc[data['target']==1]["totalWeight"].sum() )
    return data


def load_data_fullsim(inputPath,channelInTree,variables,criteria,testtruth,bdtType) :
    print variables
    my_cols_list=variables+['key','target','file']+criteria #,'tau_frWeight','lep1_frWeight','lep1_frWeight' trainVars(False)
    # if channel=='2lss_1tau' : my_cols_list=my_cols_list+['tau_frWeight','lep1_frWeight','lep2_frWeight']
    # those last are only for channels where selection is relaxed (2lss_1tau) === solve later
    sampleNames=['signal','TT','TTW','TTZ',"EWK","Rares"]
    samplesTT=['TTJets_DiLept',
    'TTJets_DiLept_ext1',
    'TTJets_SingleLeptFromT',
    'TTJets_SingleLeptFromT_ext1',
    'TTJets_SingleLeptFromTbar',
    'TTJets_SingleLeptFromTbar_ext1',
    'ST_tW_antitop_5f_inclusiveDecays',
    'ST_tW_top_5f_inclusiveDecays',
    'ST_s-channel_4f_leptonDecays',
    'ST_t-channel_antitop_4f_inclusiveDecays',
    'ST_t-channel_top_4f_inclusiveDecays']
    samplesTTW=['TTWJetsToLNu_ext1','TTWJetsToLNu_ext2']
    samplesTTZ=['TTZToLL_M10_ext2', 'TTZToLL_M10_ext1', 'TTZToLL_M-1to10']
    samplesEWK=['DYJetsToLL_M-10to50',
    'DYJetsToLL_M-50_ext1',
    'DYJetsToLL_M-50_ext2',
    'WJetsToLNu',
    'WWTo2L2Nu',
    'ZZTo4L',
    'WZTo3LNu']
    samplesRares=['WGToLNuG_ext2',
    'TGJets',
    'TGJets_ext1',
    'TTTT',
    'TTWW',
    'WZZ',
    'ZGTo2LG',
    'WGToLNuG_ext1',
    'WWTo2L2Nu_DoubleScattering',
    'WWW_4F',
    'ZZZ',
    'TTGJets'
    'TTGJets_ext1',
    'tZq_ll_4f',
    'WpWpJJ_EWK-QCD']
    dataloc = pandas.DataFrame(columns=my_cols_list)
    if bdtType=="evtLevelTT_TTH" : keys=['ttHToNonbb','TTTo2L2Nu','TTToSemilepton']
    if bdtType=="evtLevelTTV_TTH" : keys=['ttHToNonbb','TTZToLLNuNu','TTWJetsToLNu']
    if bdtType=="all" : keys=['ttHToNonbb','TTZToLLNuNu','TTWJetsToLNu','TTTo2L2Nu','TTToSemilepton']
    if bdtType=="arun" : keys=['ttHToNonbb','TTZToLLNuNu','TTWJetsToLNu','TTTo2L2Nu','TTToSemilepton']
    for sampleName in sampleNames :
    	print (sampleName, channelInTree)
    	if sampleName=='TT' : #'TTT' in folderName :
    		folderNames=samplesTT
    		target=0
    	if sampleName=='signal' :
    		folderNames=['ttHJetToNonbb_M125_amcatnlo']
    		target=1
    	if sampleName=='TTW':
    		folderNames=samplesTTW
    		target=0
        if sampleName=='TTZ' :
    		folderNames=samplesTTZ
    		target=0
    	if sampleName=='EWK':
    		folderNames=samplesEWK
    		target=0
        if sampleName=='Rares' :
    		folderNames=samplesRares
    		target=0
    	inputTree = channelInTree+'/sel/evtntuple/'+sampleName+'/evtTree'
        #print (folderNames)
        list=[]
        for folderName in folderNames :
            # TGJets_forBDTtraining_lepSS_sumOS_central_1.root
            procP1=glob.glob(inputPath+"/"+folderName+"/"+folderName+"_forBDTtraining*OS_central_*.root")
            list= list+procP1
            #if sampleName=='TT' : print (folderName)
    	#print (list)
    	for ii in range(0, len(list)) : #
    		#if sampleName=='TT' : print (list[ii])
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
    				#if ii ==0 : print (chunk_df.columns.values.tolist())
    				chunk_df['key']=folderName
    				chunk_df["target"]=target
    				chunk_df['proces']=sampleName
    				chunk_df["totalWeight"] = chunk_df.evtWeight
    				###########
    				if channel=="2lss_1tau"  and len(criteria)>0:
    					data=data.append(chunk_df.ix[chunk_df.failsTightChargeCut.values == 0], ignore_index=True)
    				else : #
    					#if 1>0 :
    					dataloc=dataloc.append(chunk_df, ignore_index=True)
    		else : print ("file "+list[ii]+"was empty")
    		tfile.Close()
    	if len(dataloc) == 0 : continue
    	nS = len(dataloc.ix[(dataloc.target.values == 0) & (dataloc.proces.values==sampleName)])
    	nB = len(dataloc.ix[(dataloc.target.values == 1) & (dataloc.proces.values==sampleName)])
    	print sampleName,"length of sig, bkg: ", nS, nB
    	if (channel=="1l_2tau" or channel=="2lss_1tau") and bdtType!="arun" :
    		nSthuth = len(dataloc.ix[(dataloc.target.values == 0) & (dataloc.bWj1Wj2_isGenMatched.values==1) & (dataloc.proces.values==sampleName)])
    		nBtruth = len(dataloc.ix[(dataloc.target.values == 1) & (dataloc.bWj1Wj2_isGenMatched.values==1) & (dataloc.proces.values==sampleName)])
    		nSthuthKin = len(dataloc.ix[(dataloc.target.values == 0) & (dataloc.bWj1Wj2_isGenMatchedWithKinFit.values==1) & (dataloc.proces.values==sampleName)])
    		nBtruthKin = len(dataloc.ix[(dataloc.target.values == 1) & (dataloc.bWj1Wj2_isGenMatchedWithKinFit.values==1) & (dataloc.proces.values==sampleName)])
    		nShadthuth = len(dataloc.ix[(dataloc.target.values == 0) & (dataloc.hadtruth.values==1) & (dataloc.proces.values==sampleName)])
    		nBhadtruth = len(dataloc.ix[(dataloc.target.values == 1) & (dataloc.hadtruth.values==1) & (dataloc.proces.values==sampleName)])
    		print "truth:              ", nSthuth, nBtruth
    		print "truth Kin:          ", nSthuthKin, nBtruthKin
    		print "hadtruth:           ", nShadthuth, nBhadtruth
    if 'ttHToNonbb' in folderName : print (dataloc.columns.values.tolist())
    n = len(dataloc)
    nS = len(dataloc.ix[dataloc.target.values == 0])
    nB = len(dataloc.ix[dataloc.target.values == 1])
    print sampleName," length of sig, bkg: ", nS, nB
    #print ("weigths", data.loc[data['target']==0]["totalWeight"].sum() , data.loc[data['target']==1]["totalWeight"].sum() )
    return dataloc


def load_data_xml (dataIn) :
    data=dataIn
    data["oldTrainTMVA_tt"]=-2.
    data["oldTrainTMVA_ttV"]=-2.
    for ii,ss in data.iterrows():
    	#if ii > 20 : break
    	clsTTreader = TMVA.Reader("Silent")
    	clsTTVreader = TMVA.Reader("Silent")
    	for feature in trainVarsTT(BDTvar) :
    		var= feature
    		if feature == "TMath::Max(TMath::Abs(lep1_eta),TMath::Abs(lep2_eta))" : var=  "max_lep_eta"
    		varVal=array.array('f',[ss[var]])
    		clsTTreader.AddVariable(str(feature),  varVal);
    	for feature in trainVarsTTV(BDTvar) :
    		var= feature
    		if feature == "TMath::Max(TMath::Abs(lep1_eta),TMath::Abs(lep2_eta))" : var=  "max_lep_eta"
    		varVal=array.array('f',[ss[var]])
    		clsTTVreader.AddVariable(str(feature),  varVal);
    	clsTTreader.BookMVA("BDT", "arun_xml_2lss_1tau/2lss_1tau_ttbar_BDTG.weights.xml")
    	clsTTVreader.BookMVA("BDT", "arun_xml_2lss_1tau/2lss_1tau_ttV_BDTG.weights.xml")
    	#for ii,ss in data.iterrows():
    	if ii % 1000 == 0 : print ("result TMVA ",ii, clsTTreader.EvaluateMVA("BDT"),clsTTVreader.EvaluateMVA("BDT"),data.loc[data.index[ii],trainVarsTT("oldTrainN")].values)
    	data.loc[data.index[ii],"oldTrainTMVA_tt"]=clsTTreader.EvaluateMVA("BDT")
    	data.loc[data.index[ii],"oldTrainTMVA_ttV"]=clsTTVreader.EvaluateMVA("BDT")
    data.to_csv('arun_xml_2lss_1tau/arun_xml_2lss_1tau_FromAnalysis.csv')
    return data

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
