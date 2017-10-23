import glob
import sys , time
#import sklearn_to_tmva
import sklearn
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
import pandas
#from pandas import HDFStore,DataFrame
import math
import matplotlib
matplotlib.use('agg')
#matplotlib.use('PS')   # generate postscript output by default
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import numpy as np
import pickle
from sklearn.externals import joblib
import root_numpy
from root_numpy import root2array, rec2array, array2root, tree2array
import ROOT
from tqdm import trange
#####################################
# to write this many files
from contextlib import contextmanager
@contextmanager
def open_file(path, mode):
     file_to=open(path,mode)
     yield file_to
     file_to.close()
#####################################
channel='1l_2tau'
if channel=='1l_2tau':channelInTree='1l_2tau_OS_Tight'
inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Oct17/histograms/1l_2tau/forBDTtraining_OS/'

keys=['ttHToNonbb','TTTo2L2Nu','TTToSemilepton']
for folderName in keys :
	print (folderName)
	if 'TT' in folderName : sampleName='TT'
	if folderName=='ttHToNonbb' : sampleName='signal'
	inputTree = channelInTree+'/sel/evtntuple/'+sampleName+'/evtTree'	
	outfile = inputPath+channel+'_'+folderName+'_21Oct2017.csv' #%sampleName
	#
	procP1=glob.glob(inputPath+"/"+folderName+"_fastsim_p1/"+folderName+"_fastsim_p1_forBDTtraining_OS_central_*.root")
	procP2=glob.glob(inputPath+"/"+folderName+"_fastsim_p2/"+folderName+"_fastsim_p2_forBDTtraining_OS_central_*.root")
	procP3=glob.glob(inputPath+"/"+folderName+"_fastsim_p3/"+folderName+"_fastsim_p3_forBDTtraining_OS_central_*.root")
	#print (procP1)
	list=procP1+procP2+procP3
	print ("Date: ", time.asctime( time.localtime(time.time()) ))
	#store = pandas.HDFStore(outfile)
	c_size = 1000
	# read schema from root and create table
	tfile = ROOT.TFile(list[0])
	tree = tfile.Get(inputTree)
	empty_arr = tree2array(tree, stop = 0)
	empty_df = pandas.DataFrame(empty_arr)
	tfile.Close()
	#store.put(folderName, empty_df, format='table')
	for ii in trange(0, len(list)) : #
		tfile = ROOT.TFile(list[ii])
		tree = tfile.Get(inputTree)
		#print (list[ii])
		if tree is not None :
			chunk_arr = tree2array(tree) #,  start=start, stop = stop)
			chunk_df = pandas.DataFrame(chunk_arr) #
			empty_df.append(chunk_df)
			#store.append(folderName, chunk_df, chunksize=tree.GetEntries())
			#with open_file(outfile,'r') as infile: chunk_df.to_csv(outfile,index=False,header=False,mode='a', chunksize=tree.GetEntries()) 
		tfile.Close()
	print ("Date: ", time.asctime( time.localtime(time.time()) ))
	empty_df.to_csv(outfile, chunksize=c_size)
	#store.close()
	print ("written "+outfile)

"""
list=procP3+procP2+procP1
print (len(list))
thefile = open('ttHToNonbb_to_csv.txt', 'w')
for item in list:
  thefile.write("%s\n" % item)
"""