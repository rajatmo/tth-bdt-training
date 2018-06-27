import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils
# run as 

process = cms.PSet()
#process.dumpPython()

sampleName = 'signal'
#sampleName = 'TT'
#sampleName = 'TTW'
#sampleName = 'TTZ'
#sampleName = 'EWK'

inputTree = '1l_2tau_OS_Tight/sel/evtntuple/%s/evtTree' %sampleName
inputPath='/hdfs/local/acaan/ttHAnalysis/2016/2017Oct17/histograms/1l_2tau/forBDTtraining_OS/'
outfile = inputPath+'1l_2tau_OS_Tight_%s_21Oct2017.csv' %sampleName

"""
procP1=glob.glob(inputPath+"/"+folderName+"_fastsim_p1/"+folderName+"_fastsim_p1_forBDTtraining_OS_central_*.root")
procP2=glob.glob(inputPath+"/"+folderName+"_fastsim_p2/"+folderName+"_fastsim_p2_forBDTtraining_OS_central_*.root")
procP3=glob.glob(inputPath+"/"+folderName+"_fastsim_p3/"+folderName+"_fastsim_p3_forBDTtraining_OS_central_*.root")
print (procP1)
"""

if sampleName=='signal': folderName='ttHToNonbb' #'ttHToNonbb_fastsim_p1_forBDTtraining_OS_central_9.root'
mylist = FileUtils.loadListFromFile (folderName+'_to_csv.txt')

process.fwliteInput = cms.PSet(
	#fileNames = cms.vstring(),
	#fileNames.extend(procP1),
	#fileNames.extend(procP2),
	#fileNames.extend(procP3),
	fileNames = cms.untracked.vstring( *mylist),
    ##maxEvents = cms.int32(100000),
	maxEvents = cms.int32(-1),
	outputEvery = cms.uint32(10000)
)

process.fwliteOutput = cms.PSet(
    fileName = cms.string(outfile)
)

process.write_csv = cms.PSet(    

    treeName = cms.string(inputTree),
    #preselection = cms.string("memOutput_errorFlag==0"),
    branches_to_write = cms.PSet(
        # list of branches in input Ntuple that will be written to CSV output file
        avg_dr_jet=cms.string('avg_dr_jet/F'),
        dr_lep_fittedHadTop=cms.string('dr_lep_fittedHadTop/F'),
        dr_lep_tau_os=cms.string('dr_lep_tau_os/F'),
        dr_lep_tau_ss=cms.string('dr_lep_tau_ss/F'),
        dr_taus=cms.string('dr_taus/F'),
        evtWeight=cms.string('evtWeight/F'),
        fittedHadTop_eta=cms.string('fittedHadTop_eta/F'),
        fittedHadTop_pt=cms.string('fittedHadTop_pt/F'),
        genWeight=cms.string('genWeight/F'),
        htmiss=cms.string('htmiss/F'),
        lep_conePt=cms.string('lep_conePt/F'),
        lep_eta=cms.string('lep_eta/F'),
        lep_pt=cms.string('lep_pt/F'),
        lep_tth_mva=cms.string('lep_tth_mva/F'),
        lumiScale=cms.string('lumiScale/F'),
        mT_lep=cms.string('mT_lep/F'),
        mTauTauVis=cms.string('mTauTauVis/F'),
        mindr_lep_jet=cms.string('mindr_lep_jet/F'),
        mindr_tau1_jet=cms.string('mindr_tau1_jet/F'),
        mindr_tau2_jet=cms.string('mindr_tau2_jet/F'),
        mvaOutput_hadTopTagger=cms.string('mvaOutput_hadTopTagger/F'),
        ptmiss=cms.string('ptmiss/F'),
        tau1_eta=cms.string('tau1_eta/F'),
        tau1_mva=cms.string('tau1_mva/F'),
        tau1_pt=cms.string('tau1_pt/F'),
        tau2_eta=cms.string('tau2_eta/F'),
        tau2_mva=cms.string('tau2_mva/F'),
        tau2_pt=cms.string('tau2_pt/F'),
        nBJetLoose=cms.string('nBJetLoose/I'),
        nBJetMedium=cms.string('nBJetMedium/I'),
        nJet=cms.string('nJet/I'),
        run=cms.string('run/i'),
        lumi=cms.string('lumi/i'),
        evt=cms.string('evt/l'),   
        )
)
