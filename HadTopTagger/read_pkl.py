from time import time,ctime
import sys,os 
"""
import shutil,subprocess
proc=subprocess.Popen(['cd /home/acaan/CMSSW_9_4_0_pre1/src/ ; cmsenv '],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()

# /cvmfs/cms.cern.ch/slc6_amd64_gcc530/external/py2-pandas/0.17.1-giojec/lib/python2.7/site-packages/pandas-0.17.1-py2.7-linux-x86_64.egg
	
os.environ['PYTHONUSERBASE'] = '/home/acaan/CMSSW_8_1_0/src/cmssw_extras'
os.environ['PATH'] = '/home/acaan/CMSSW_8_1_0/src/cmssw_extras/bin:'+os.environ['PATH'] 
os.environ['PYTHONPATH'] = '/home/acaan/CMSSW_8_1_0/src/cmssw_extras/lib/python2.7/site-packages:'+os.environ['PYTHONPATH'] 
"""

"""
os.environ['PYTHONUSERBASE'] = '/cvmfs/cms.cern.ch/slc6_amd64_gcc530/external/py2-scikit-learn/0.17.1-ikhhed'
os.environ['PATH'] = '/home/acaan/CMSSW_8_1_0/src/cmssw_extras/bin:'+os.environ['PATH'] 
os.environ['PYTHONPATH'] = '/cvmfs/cms.cern.ch/slc6_amd64_gcc530/external/py2-scikit-learn/0.17.1-ikhhed/lib/python2.7/site-packages:'+os.environ['PYTHONPATH'] 
"""

#sys.path.append('/cvmfs/cms.cern.ch/slc6_amd64_gcc530/external/py2-scikit-learn/0.17.1-ikhhed/lib/python2.7/site-packages')
import sklearn
from sklearn.externals import joblib
print('The scikit-learn version is {}.'.format(sklearn.__version__))
import pandas
print('The pandas version is {}.'.format(pandas.__version__))
import cPickle as pickle
print('The pickle version is {}.'.format(pickle.__version__))
import numpy as np 
print('The numpy version is {}.'.format(np.__version__))


#sys.path.insert(0, "/home/acaan/classifiers/xgboost/python-package")
sys.path.insert(0, '/cvmfs/cms.cern.ch/slc6_amd64_gcc530/external/py2-pippkgs_depscipy/3.0-njopjo7/lib/python2.7/site-packages')
import xgboost as xgb 
print('The xgb version is {}.'.format(xgb.__version__))

import subprocess 
from sklearn.externals import joblib 
from itertools import izip 

vec2=['CSV_b','qg_Wj2' ,'qg_Wj1','m_bWj1Wj2' ,'pT_bWj1Wj2','m_Wj1Wj2','nllKinFit','alphaKinFit','pT_b','pT_b_o_kinFit_pT_b',
		'pT_Wj1_o_kinFit_pT_Wj1','pT_Wj2','pT_Wj2_o_kinFit_pT_Wj2','cosThetaW_rest','cosThetaWj1_restW']

				
def mul():
	print 'Today is',ctime(time()), 'All python libraries we need loaded good	HTT'
	#new_dict = {'CSV_b': 0.14102917909622192, 'qg_Wj2': 0.8344255685806274, 'pT_Wj1Wj2': 55.387821197509766, 'nllKinFit': 1.5610963106155396, 'pT_bWj1Wj2': 46.315467834472656, 'pT_Wj2': 43.89348602294922, 'm_Wj1Wj2': 77.6514892578125}
	new_dict = {'CSV_b' : 0.232256,
		'qg_Wj2' : 0.118736,
		'qg_Wj1' : 0.993289,
		'm_bWj1Wj2' : 131.347946,
		'pT_bWj1Wj2' : 112.474075,
		'm_Wj1Wj2' : 50.330856,
		'pT_Wj2' : 37.480865}
	print (new_dict)
	data = pandas.DataFrame() 
	data=data.append(new_dict, ignore_index=True) 
	print (data.columns.values.tolist())
	print len(data)
	result=-20
	bdtfile='HadTopTagger_sklearnV0o17o1_HypOpt/TTToSemilepton_HadTopTagger_sklearnV0o17o1_HypOpt_XGB_ntrees_6_deph_1_lr_0o01_CSV_sort.pkl'
	bdtfilejoblib='HadTopTagger_sklearnV0o17o1_HypOpt/TTToSemilepton_HadTopTagger_sklearnV0o17o1_HypOpt_XGB_ntrees_6_deph_1_lr_0o01_CSV_sort_joblib.pkl'
	f = None  
	try: 
		f = open('/home/acaan/CMSSW_9_4_0_pre1/src/tth-bdt-training-test/HadTopTagger/'+bdtfile,'rb')
		fjl = open('/home/acaan/CMSSW_9_4_0_pre1/src/tth-bdt-training-test/HadTopTagger/'+bdtfilejoblib,'rb')
	except IOError as e: 
		print('Couldnt open or write to file (%s).' % e) 
	else:  
			print ('file opened') 
			try: 
				pkldata = pickle.load(f)
				#jldata = joblib.load(fjl)
			except : 
				print('Oops!',sys.exc_info()[0],'occured.') 
			else: 
				model = pkldata.booster().get_dump(with_stats=False) 
				print model
				
				#modeljl = jldata.booster().get_dump(with_stats=False) 
				#print modeljl
				
				print ('pkl loaded') 
				proba = pkldata.predict_proba(data[data.columns.values.tolist()].values  )
				
				result = proba[:,1][0]
				print ('predict BDT to one event',result)  
				f.close()  
	return result                                

if __name__ == "__main__":
    # execute only if run as a script
	mul()
