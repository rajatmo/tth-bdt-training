Repo to run event-level BDT

We have here the following functionalities/options
===================================================================================
sklearn_Xgboost_csv_evtLevel_ttH.py

By default we use 50% of the dataset to train/test = the same split is used for the three classifiers / run
this training fraction is also point of optimization at some point

--channel 

  The ones whose variables implemented now are:
  
   - 1l_2tau
   
   - 2lss_1tau
   
  It will create a local folder and store the report*/xml

--optimization True/False

  --> if optimization==False it does not write the xml file

--hypOpt

  Runs hyp. optimiyation with GridSearchCV in XGBoost 
  
  (of course you need to tune which ones to run on in the code, look for GridSearchCV)
  
  Does not output any report than print the result of the ROC AUC in the screen.

  The trick to the BDT result be more stable (result on train closer to the one on test) is:

  - n_tree (estimators) is high (>1000)
  
  - depth of each tree is smalle (1 to 3)
  
  - learning rate is small (0.01 should be the minimum)

  The opimiyation target is ROC AUC, if you want you can change it to loss.
  
  --> If you are doing that, mind the overtraining rate monitoring the run!

--variables

  Set of variables to use -- it shall be put by hand in the code, in the fuction trainVars(all) 
  
                             all==True -- all variables that should be loaded (including weights) -- it is used only once  
                             
                             all==False -- only variables of training (not including weights)
                             
  For the 2lss_2tau for example I defined 3 sets of variables/each to confront at limit level

  trainvar="allVar" -- all variables that are avaible to training (including lepton IDs, this is here just out of curiosity) 
  
  trainvar="oldVar" -- a minimal set of variables (excluding lepton IDs and lep pt's)
 
  trainvar="notForbidenVar" -- a maximal set of variables (excluding lepton IDs and lep pt's) 
  
  trainvar="notForbidenVarNoMEM" -- the same as above, but excluding as well MeM variables

====================================================================================

Disclaimer: this code require are the ntuples created with the option "forBDTtraining" of
 	tthAnalyzeRun_CHANNEL.py

https://github.com/HEP-KBFI/tth-htt/tree/master/test

It reads and converts to python panda in the fly, it can take up to 5 min to load the data. 
I tested that doing this with the intermediary step of creating a .csv ot hdf5 file would require aproximatelly the same loading time/run.

======================================================================================

xgboost2tmva.py --> transcribe the output of XGBoost to xml file, compatible with KBFI analysis code

                    If your number of trees is high this will take some minutes (of giva a bus error-), 
                    
                    I suggest to have optimization==False if you are still tunning variables/parameters
                    
                    (I am not sure it is working to CatBoost algorithm) 

To make the output xml readable also by humans you should do in command line

xmllint --format FILE.xml

At some point I will incorporate that to the sklearn_Xgboost_csv_evtLevel_ttH.py

=====================================================================================

* in the report three algorithms are confronted/set of variables

Gradient Boost = GB

XGBoost = XGB

CatBoost = CB 

The GB is there for historical reasons and the CB by scientific curiosity.

The xml file is done only for XGB

====================================================================================

If you are implementing a new channel mind to:

 - Implement the variables on trainVars() function

 - check the drawn of variables in the repport

