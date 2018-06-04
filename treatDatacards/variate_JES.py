from array import array
from ROOT import *
from math import sqrt, sin, cos, tan, exp
#from root_numpy import root2array, stretch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

#Set plotting style
gStyle.SetOptStat(0)
gStyle.SetOptFit(0)

mom="/home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun02/JES/"
variations=["mvaNtuple_ttHJetToNonbb_signal_1l2tau.root",  "mvaNtuple_ttHJetToNonbb_signal_1l2tau_jesdown.root",  "mvaNtuple_ttHJetToNonbb_signal_1l2tau_jesup.root"]
label = ["central", "JES down", "JES up"]
color = [8,9,6]

hist1D =  TH1F('Name', 'BDT discriminator; Events',10, 0, 1)
hist1D_1d =  TH1F('Name1', 'BDT discriminator; Events',10, 0, 1)
hist1D_1u =  TH1F('Name2', 'BDT discriminator; Events',10, 0, 1)

histos = [hist1D,hist1D_1d, hist1D_1u]
canv= TCanvas( 'c2', '@ Analysis level ',  400, 400 )
xl1=.1
yl1=0.6
xl2=xl1+.25
yl2=yl1+.3;
leg2 = TLegend(xl1,yl1,xl2,yl2);
for nn, variation in enumerate(variations) :
    #if nn > 0 : continue
    file=TFile(mom+variation)
    tree = file.Get("mva")
    #hist1D =  TH1F('Name', 'BDT discriminator; Events',10, 0, 1)
    for entry in tree:
        mvaOutput_1l_2tau_HTT_SUM_VT = entry.mvaOutput_1l_2tau_HTT_SUM_VT
        event_weight = entry.event_weight
        histos[nn].Fill(mvaOutput_1l_2tau_HTT_SUM_VT,event_weight)
    #hist1D.Write()
    histos[nn].SetLineColor(color[nn])
    histos[nn].SetTitle("  ")
    histos[nn].SetLineWidth(3)
    histos[nn].GetXaxis().SetTitle("BDT output")
    histos[nn].GetYaxis().SetTitle("Events")
    leg2.AddEntry(histos[nn],label[nn],"l")
    if nn == 0 : histos[nn].Draw()
    else : histos[nn].Draw("same")
    histos[nn].Draw("hist,same")
leg2.Draw()
canv.SaveAs('varioations_JES_1l_2tau_BDT.pdf','pdf')
    #arrayBranches =  tree.GetListOfBranches()
    #for branch in arrayBranches :
    #    print branch.GetName()
