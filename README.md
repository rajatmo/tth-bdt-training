# tth-bdt-training

### Check out this package with the :
cd $CMSSW_BASE/src/
git clone https://github.com/HEP-KBFI/tth-bdt-training.git $CMSSW_BASE/src/tthAnalysis/bdtTraining


Auxiliary code and config files for BDT training, used by the ttH with H->tautau analysis

Do cmsenv in the release you are going to work (eg CMSSW_9_4_0_pre1, it does not really matter as we only use ROOT of it)

### there is no instalation necessary to use the scripts if you do have already sklearn and xgboost (that is the case of CMSSW_9X release). If you use 8X the scripts for training and reading will also work.

=================================================
### If you do need a local instalation of the ML packages starting from python2.7 (e.g. you do not want to use any cmssw stuff) the bellow can be used

Install conda to a home directory -- the recipe in the link bellow will install python2.7 althoguether
https://conda.io/docs/user-guide/install/linux.html (with prefix)
https://www.anaconda.com/download/ 

Install pip: https://pip.pypa.io/en/stable/installing/ (using --user option)
Do:

pip install scikit-learn --user

pip install xgboost --user

pip install catboost --user

pip uninstall numpy (yes, unistall, otherwise will conflict with ROOT that came with CMSSW)

In the beggining of each session ALWAYS run setup enviroment (after the cmsenv)

export PYTHONUSERBASE=/home/acaan/python_local/

export PATH=/home/acaan/python_local/bin:$PATH

export PYTHONPATH=/home/acaan/python_local/lib/python2.7/site-packages:$PYTHONPATH

=============================================================================================

