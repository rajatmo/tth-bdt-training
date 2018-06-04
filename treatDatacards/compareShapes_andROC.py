#!/usr/bin/env python

#Load modules here
from array import array
from ROOT import *
from math import sqrt, sin, cos, tan, exp

#Set plotting style
gStyle.SetOptStat(0)
gStyle.SetOptFit(0)

SavePlots = True
verbose = True
Nverbose = 10
veryverbose = False

# Calculate ROC curve from two histograms:
def CalcROC(histSig, histBkg) :

    # Loop over histogram bins and calculate vectors of efficiencies:
    # ---------------------------------------------------------------
    effSig = array( "f" )
    effBkg = array( "f" )

    # Check that the two histograms have the same number of bins and same range:
    if (histSig.GetNbinsX() == histBkg.GetNbinsX()) :
        if ( abs(histSig.GetXaxis().GetXmax() - histBkg.GetXaxis().GetXmax()) < 0.0001 and
             abs(histSig.GetXaxis().GetXmin() - histBkg.GetXaxis().GetXmin()) < 0.0001) :
            Nbins = histSig.GetNbinsX()

            # Get integral (including underflow and overflow):
            integralSig = 0.0
            integralBkg = 0.0
            for ibin in range(Nbins+2) :
                integralSig += histSig.GetBinContent(ibin)
                integralBkg += histBkg.GetBinContent(ibin)

            # Integrate each bin, and add result to ROC curve (contained in effSig and effBkg):
            effSig.append(0.0)
            effBkg.append(0.0)
            sumSig = 0.0
            sumBkg = 0.0
            for ibin in range (Nbins+1, 0, -1) :
                sumSig += histSig.GetBinContent(ibin)
                sumBkg += histBkg.GetBinContent(ibin)
                effSig.append(sumSig/integralSig)
                effBkg.append(sumBkg/integralBkg)
                if (veryverbose) :
                    print "  bin %3d:   effSig = %5.3f   effBkg = %5.3f"%(ibin, effSig[-1], effBkg[-1])

            # Make ROC curve in a TGraph of the two arrays, and return this:
            graphROC = TGraph(len(effSig), effSig, effBkg)
            return graphROC

        else :
            print "ERROR: Signal and Background histograms have different ranges!"
            return None

    else :
        print "ERROR: Signal and Background histograms have different binning!"
        return None

    return None

channels = [
    "3l_1tau",
    "2lss_1tau_plainKin",
    "2lss_1tau_HTT",
    "2l_2tau",
    "1l_2tau_plainKin",
    "1l_2tau_HTT",
]

labels = [
    "3l 1#tau_{h}",
    "2l ss 1#tau_{h}",
    "2l ss 1#tau_{h}",
    "2l 2#tau_{h}",
    "1l 2#tau_{h}",
    "1l 2#tau_{h}"
]

repo2017 = [
    "/hdfs/local/karl/ttHAnalysis/2017/2018Jun04_take2_3l1tau_WP090/histograms/3l_1tau/histograms_harvested_stage2_3l_1tau_Tight_OS.root",
    "/hdfs/local/karl/ttHAnalysis/2017/2018Jun04_take2_2lss1tau_WP090/histograms/2lss_1tau/histograms_harvested_stage2_2lss_1tau_Tight_lepSS_sumOS.root",
    "/hdfs/local/karl/ttHAnalysis/2017/2018Jun04_take2_2lss1tau_WP090/histograms/2lss_1tau/histograms_harvested_stage2_2lss_1tau_Tight_lepSS_sumOS.root",
    "/hdfs/local/karl/ttHAnalysis/2017/2018Jun04_take2_2l2tau_WP090/histograms/2l_2tau/histograms_harvested_stage2_2l_2tau_disabled_disabled_Tight_OS.root",
    "/hdfs/local/karl/ttHAnalysis/2017/2018Jun04_take2_1l2tau_WP090/histograms/1l_2tau/histograms_harvested_stage2_1l_2tau_Tight_OS.root",
    "/hdfs/local/karl/ttHAnalysis/2017/2018Jun04_take2_1l2tau_WP090/histograms/1l_2tau/histograms_harvested_stage2_1l_2tau_Tight_OS.root"
]

repo2017_WP075 = [
    "/hdfs/local/karl/ttHAnalysis/2017/2018Jun01_3l1tau_WP075/histograms/3l_1tau/histograms_harvested_stage2_3l_1tau_Tight_OS.root",
    "/hdfs/local/karl/ttHAnalysis/2017/2018Jun01_2lss1tau/histograms/2lss_1tau/histograms_harvested_stage2_2lss_1tau_Tight_lepSS_sumOS.root",
    "/hdfs/local/karl/ttHAnalysis/2017/2018Jun01_2lss1tau/histograms/2lss_1tau/histograms_harvested_stage2_2lss_1tau_Tight_lepSS_sumOS.root",
    "/hdfs/local/karl/ttHAnalysis/2017/2018Jun01_2l2tau_WP075/histograms/2l_2tau/histograms_harvested_stage2_2l_2tau_disabled_disabled_Tight_OS.root",
    "/hdfs/local/karl/ttHAnalysis/2017/2018Jun01_1l2tau_WP075/histograms/1l_2tau/histograms_harvested_stage2_1l_2tau_Tight_OS.root",
    "/hdfs/local/karl/ttHAnalysis/2017/2018Jun01_1l2tau_WP075/histograms/1l_2tau/histograms_harvested_stage2_1l_2tau_Tight_OS.root"
]

histostitle = [
    "plainKin_SUM_M",
    "plainKin_SUM_M",
    "HTT_SUM_M",
    "plainKin_SUM_VT",
    "plainKin_SUM_VT",
    "HTT_SUM_VT"
    ]

histos2017 = [
    "mvaOutput_plainKin_SUM_M_noRebin",
    "mvaOutput_2lss_1tau_plainKin_1B_M",
    "mvaOutput_2lss_1tau_HTT_SUM_M_noRebin",
    "mvaOutput_plainKin_SUM_VT",
    "mvaOutput_plainKin_SUM_VT_noRebin",
    "mvaOutput_HTT_SUM_VT_noRebin"
    ]

repo2016 = [
    "/hdfs/local/acaan/ttHAnalysis/2016/3l_1tau_2018Mar12_VHbb_TLepMTau_shape/histograms/3l_1tau/histograms_harvested_stage2_3l_1tau_Tight_OS.root",
    "/hdfs/local/acaan/ttHAnalysis/2016/2lss_1tau_2018Feb28_VHbb_TLepMTau_shape/histograms/2lss_1tau/histograms_harvested_stage2_2lss_1tau_Tight_lepSS_sumOS.root",
    "/hdfs/local/acaan/ttHAnalysis/2016/2lss_1tau_2018Feb28_VHbb_TLepMTau_shape/histograms/2lss_1tau/histograms_harvested_stage2_2lss_1tau_Tight_lepSS_sumOS.root",
    "/hdfs/local/acaan/ttHAnalysis/2016/2l_2tau_2018Feb20_VHbb_TLepVTTau/histograms/2l_2tau/histograms_harvested_stage2_2l_2tau_disabled_disabled_Tight_OS.root",
    "/hdfs/local/acaan/ttHAnalysis/2016/1l_2tau_2018Mar02_VHbb_TLepVTTau_shape/histograms/1l_2tau/histograms_harvested_stage2_1l_2tau_Tight_OS.root",
    "/hdfs/local/acaan/ttHAnalysis/2016/1l_2tau_2018Mar02_VHbb_TLepVTTau_shape/histograms/1l_2tau/histograms_harvested_stage2_1l_2tau_Tight_OS.root"
    ]

histos2016 = [
    "mvaOutput_noHTT_SUM_M",
    "mvaOutput_2lss_noHTT_1B_M",
    "mvaOutput_2lss_HTT_1B_M",
    "mvaOutput_noHTT_SUM_VT",
    "mvaOutput_sum_noHTT_VT",
    "mvaOutput_sum_HTT_VT"
    ]

repoHistos = [
    "3l_1tau_OS_lepTight_tauTight/sel/",
    "2lss_1tau_lepSS_sumOS_Tight/sel/",
    "2lss_1tau_lepSS_sumOS_Tight/sel/",
    "2l_2tau_sumOS_Tight/sel/",
    "1l_2tau_OS_Tight/sel/",
    "1l_2tau_OS_Tight/sel/"
    ]

drawFakes = [
    True, True, True, True, True, True
]

drawTTV = [
    True, True, True, True, True, True
]

rebinBy = [
    25,
    1*10,
    1*10,
    25,
    10,
    10
]

rebinBy_2016 = [
    25,
    6*2*10,
    6*2*10,
    25,
    10,
    10
]

setmax = [
    1.2,
    0.9,
    0.9,
    1.2,
    0.42,
    0.42
]

############################################################
doRoc=False

if doRoc :
    for nn, channel in enumerate(channels) :
        if not nn in [1,2]  : continue

        file=TFile(repo2017[nn])
        signal_mvaOutput = file.Get(repoHistos[nn]+'evt/signal/'+histos2017[nn])
        TTW_mvaOutput = file.Get(repoHistos[nn]+'evt/TTW/'+histos2017[nn])
        TTZ_mvaOutput = file.Get(repoHistos[nn]+'evt/TTZ/'+histos2017[nn])
        EWK_mvaOutput = file.Get(repoHistos[nn]+'evt/EWK/'+histos2017[nn])
        Rares_mvaOutput = file.Get(repoHistos[nn]+'evt/Rares/'+histos2017[nn])
        fakes_data_mvaOutput = file.Get(repoHistos[nn]+'evt/fakes_data/'+histos2017[nn])
        print repo2017[nn]
        print repoHistos[nn]+'evt/TTW/'+histos2017[nn]
        AllBKG = TTW_mvaOutput + TTZ_mvaOutput + EWK_mvaOutput + Rares_mvaOutput #+ fakes_data_mvaOutput

        file3=TFile(repo2017_WP075[nn])
        signal_mvaOutput_WP075 = file3.Get(repoHistos[nn]+'evt/signal/'+histos2017[nn])
        TTW_mvaOutput_WP075 = file3.Get(repoHistos[nn]+'evt/TTW/'+histos2017[nn])
        TTZ_mvaOutput_WP075 = file3.Get(repoHistos[nn]+'evt/TTZ/'+histos2017[nn])
        EWK_mvaOutput_WP075 = file3.Get(repoHistos[nn]+'evt/EWK/'+histos2017[nn])
        Rares_mvaOutput_WP075 = file3.Get(repoHistos[nn]+'evt/Rares/'+histos2017[nn])
        fakes_data_mvaOutput_WP075 = file3.Get(repoHistos[nn]+'evt/fakes_data/'+histos2017[nn])
        AllBKG_WP075 = TTW_mvaOutput_WP075 + TTZ_mvaOutput_WP075 + EWK_mvaOutput_WP075 + Rares_mvaOutput_WP075 #+ fakes_data_mvaOutput_WP075

        file2=TFile(repo2016[nn])
        signal_mvaOutput_2016 = file2.Get(repoHistos[nn]+'evt/signal/'+histos2016[nn])
        TTW_mvaOutput_2016 = file2.Get(repoHistos[nn]+'evt/TTW/'+histos2016[nn])
        TTZ_mvaOutput_2016 = file2.Get(repoHistos[nn]+'evt/TTZ/'+histos2016[nn])
        EWK_mvaOutput_2016 = file2.Get(repoHistos[nn]+'evt/EWK/'+histos2016[nn])
        Rares_mvaOutput_2016 = file2.Get(repoHistos[nn]+'evt/Rares/'+histos2016[nn])
        fakes_data_mvaOutput_2016 = file2.Get(repoHistos[nn]+'evt/fakes_data/'+histos2016[nn])
        AllBKG_2016 = TTW_mvaOutput_2016 + TTZ_mvaOutput_2016 + EWK_mvaOutput_2016 + Rares_mvaOutput_2016 #+ fakes_data_mvaOutput_2016

        print ("nbins: ", fakes_data_mvaOutput.GetXaxis().GetNbins() , fakes_data_mvaOutput_2016.GetXaxis().GetNbins())

        c2 = TCanvas( 'c2', '@ Analysis level ',  400, 400 )
        xl1=.5
        yl1=0.1
        xl2=xl1+.4
        yl2=yl1+.3;
        leg2 = TLegend(xl1,yl1,xl2,yl2);
        #leg2.SetHeader(channel+" (ttH X MC-only BKG + fakes)")
        roc_2017=CalcROC(AllBKG,signal_mvaOutput)
        roc_2017.SetLineColor(8)
        roc_2017.SetTitle("  ")
        roc_2017.SetLineWidth(3)
        roc_2017.GetXaxis().SetTitle("False positive rate")
        roc_2017.GetYaxis().SetTitle("True positive rate")
        #f1 = TF1("f",roc_2017,0.,1.)
        #integral = f1.Integral(0.,1.);
        leg2.AddEntry(roc_2017,"2017 lepMVA 0.9","l")
        roc_2017.GetXaxis().SetRangeUser(0.,1.)
        roc_2017.GetYaxis().SetRangeUser(0.,1.)
        roc_2017.Draw()

        """
        roc_2017_WP075=CalcROC(AllBKG_WP075, signal_mvaOutput_WP075)
        roc_2017_WP075.SetLineColor(9)
        roc_2017_WP075.SetLineWidth(3)
        leg2.AddEntry(roc_2017_WP075,"2017 lepMVA 0.75","l")
        roc_2017_WP075.Draw("same")
        """

        roc_2016=CalcROC(AllBKG_2016, signal_mvaOutput_2016)
        roc_2016.SetLineColor(kMagenta)
        roc_2016.SetLineWidth(3)
        leg2.AddEntry(roc_2016,"2016","l")
        roc_2016.Draw("same")

        leg2.Draw("same")
        t = TLatex()
        #TLatex t(.1,.9,channels[nn]);
        t.SetNDC(kTRUE);
        #t.SetTextFont(32)
        t.SetTextColor(1)
        t.SetTextSize(0.04)
        t.SetTextAlign(12)
        t.DrawLatex( .15,.85,labels[nn] )
        t.DrawLatex( .15,.8,"ttH X MC-only BKG + fakes" )
        #t.DrawLatex( .15,.75,"" )
        c2.SaveAs('roc_'+channel+'_TTV.pdf')

        c3 = TCanvas( 'c3', '@ Analysis level ',  400, 400 )
        leg3 = TLegend(xl1,yl1,xl2,yl2);
        #leg3.SetHeader(channel+" (ttH X MC-only BKG + fakes)")
        roc_fakes_2017=CalcROC(fakes_data_mvaOutput,signal_mvaOutput)
        roc_fakes_2017.SetLineColor(8)
        roc_fakes_2017.SetTitle("  ")
        roc_fakes_2017.SetLineWidth(3)
        roc_fakes_2017.GetXaxis().SetTitle("False positive rate")
        roc_fakes_2017.GetYaxis().SetTitle("True positive rate")
        roc_fakes_2017.GetXaxis().SetRangeUser(0.,1.)
        roc_fakes_2017.GetYaxis().SetRangeUser(0.,1.)
        leg3.AddEntry(roc_2017,"2017 lepMVA 0.9","l")
        roc_fakes_2017.Draw()

        """
        roc_fakes_2017_WP075=CalcROC(fakes_data_mvaOutput_WP075, signal_mvaOutput_WP075)
        roc_fakes_2017_WP075.SetLineColor(9)
        roc_fakes_2017_WP075.SetLineWidth(3)
        leg3.AddEntry(roc_2017_WP075,"2017 lepMVA 0.75","l")
        roc_fakes_2017_WP075.Draw("same")
        """

        roc_fakes_2016=CalcROC(fakes_data_mvaOutput_2016, signal_mvaOutput_2016)
        roc_fakes_2016.SetLineColor(kMagenta)
        roc_fakes_2016.SetLineWidth(3)
        leg3.AddEntry(roc_2016,"2016","l")
        roc_fakes_2016.Draw("same")

        leg3.Draw("same")
        t = TLatex()
        #TLatex t(.1,.9,channels[nn]);
        t.SetNDC(kTRUE);
        #t.SetTextFont(32)
        t.SetTextColor(1)
        t.SetTextSize(0.04)
        t.SetTextAlign(12)
        t.DrawLatex( .15,.85,labels[nn] )
        t.DrawLatex( .15,.8,"ttH X MC-only BKG + fakes" )
        #t.DrawLatex( .15,.75,"" )
        c3.SaveAs('roc_'+channel+'_fakes.pdf')

        xl1=.4
        yl1=0.65
        xl2=xl1+.5
        yl2=yl1+.25;
        leg = TLegend(xl1,yl1,xl2,yl2);

        #"""
        signal_mvaOutput.Scale(1./signal_mvaOutput.Integral())
        #signal_mvaOutput_WP075.Scale(1./signal_mvaOutput_WP075.Integral())
        signal_mvaOutput_2016.Scale(1./signal_mvaOutput_2016.Integral())
        #AllBKG_WP075.Scale(1./AllBKG_WP075.Integral())
        AllBKG.Scale(1./AllBKG.Integral())
        AllBKG_2016.Scale(1./AllBKG_2016.Integral())
        fakes_data_mvaOutput.Scale(1./fakes_data_mvaOutput.Integral())
        #fakes_data_mvaOutput_WP075.Scale(1./fakes_data_mvaOutput_WP075.Integral())
        fakes_data_mvaOutput_2016.Scale(1./fakes_data_mvaOutput_2016.Integral())
        #"""

        #leg.SetHeader(channel)
        c1 = TCanvas( 'c1', '@ Analysis level ',  400, 400 )
        signal_mvaOutput.SetLineColor(8)
        signal_mvaOutput.SetLineWidth(3)
        signal_mvaOutput.Rebin(rebinBy[nn])
        signal_mvaOutput.SetTitle("  ")
        signal_mvaOutput.SetLineStyle(10)
        signal_mvaOutput.GetXaxis().SetTitle(histostitle[nn])
        leg.AddEntry(signal_mvaOutput,"ttH - 2017 lepMVA 0.9","l")
        signal_mvaOutput.Draw()

        """
        signal_mvaOutput_WP075.SetLineColor(8)
        signal_mvaOutput_WP075.SetLineWidth(3)
        signal_mvaOutput_WP075.SetLineStyle(2)
        signal_mvaOutput_WP075.Rebin(rebinBy[nn])
        signal_mvaOutput_WP075.SetTitle("CMS simulation                                           ")
        signal_mvaOutput_WP075.GetXaxis().SetTitle(histostitle[nn])
        leg.AddEntry(signal_mvaOutput_WP075,"ttH - 2017 lepMVA 0.75","l")
        signal_mvaOutput_WP075.Draw("same")
        """

        signal_mvaOutput_2016.SetLineColor(8)
        signal_mvaOutput_2016.SetLineWidth(3)
        signal_mvaOutput_2016.Rebin(rebinBy_2016[nn])
        leg.AddEntry(signal_mvaOutput_2016,"ttH - 2016","l")
        #signal_mvaOutput_2016.GetXaxis().SetRangeUser(0.,1.)
        signal_mvaOutput_2016.Draw("same")

        if drawTTV[nn] :
            AllBKG.SetLineColor(kMagenta)
            AllBKG.SetLineWidth(3)
            AllBKG.Rebin(rebinBy[nn])
            AllBKG.SetLineStyle(10)
            leg.AddEntry(AllBKG,"MC-only BKG - 2017 lepMVA 0.9","l")
            AllBKG.Draw("same")

            """
            AllBKG_WP075.SetLineColor(kMagenta)
            AllBKG_WP075.SetLineWidth(3)
            AllBKG_WP075.SetLineStyle(2)
            AllBKG_WP075.Rebin(rebinBy[nn])
            leg.AddEntry(AllBKG_WP075,"MC-only BKG - 2017 lepMVA 0.75","l")
            AllBKG_WP075.Draw("same")
            """

            AllBKG_2016.SetLineColor(kMagenta)
            AllBKG_2016.SetLineWidth(3)
            AllBKG_2016.Rebin(rebinBy_2016[nn])
            leg.AddEntry(AllBKG_2016,"MC-only BKG - 2016","l")
            AllBKG_2016.Draw("same")

        if drawFakes[nn] :
            fakes_data_mvaOutput.SetLineColor(9)
            fakes_data_mvaOutput.SetLineWidth(3)
            fakes_data_mvaOutput.Rebin(rebinBy[nn])
            fakes_data_mvaOutput.SetLineStyle(10)
            leg.AddEntry(fakes_data_mvaOutput,"fakes  - 2017 lepMVA 0.9","l")
            fakes_data_mvaOutput.Draw("same")

            """
            fakes_data_mvaOutput_WP075.SetLineColor(9)
            fakes_data_mvaOutput_WP075.SetLineWidth(3)
            fakes_data_mvaOutput_WP075.SetLineStyle(2)
            fakes_data_mvaOutput_WP075.Rebin(rebinBy[nn])
            leg.AddEntry(fakes_data_mvaOutput_WP075,"fakes - 2017 lepMVA 0.75","l")
            fakes_data_mvaOutput_WP075.Draw("same")
            """

            fakes_data_mvaOutput_2016.SetLineColor(9)
            fakes_data_mvaOutput_2016.SetLineWidth(3)
            fakes_data_mvaOutput_2016.Rebin(rebinBy_2016[nn])
            leg.AddEntry(fakes_data_mvaOutput_2016,"fakes - 2016","l")
            fakes_data_mvaOutput_2016.Draw("same")

        signal_mvaOutput.Draw("hist,same")
        #signal_mvaOutput_WP075.Draw("hist,same")
        signal_mvaOutput_2016.Draw("hist,same")
        if drawFakes[nn] :
            fakes_data_mvaOutput.Draw("hist,same")
            #fakes_data_mvaOutput_WP075.Draw("hist,same")
            fakes_data_mvaOutput_2016.Draw("hist,same")
        if drawTTV[nn] :
            AllBKG.Draw("hist,same")
            #AllBKG_WP075.Draw("hist,same")
            AllBKG_2016.Draw("hist,same")

        signal_mvaOutput.GetXaxis().SetRangeUser(0.,1.)
        signal_mvaOutput.SetMaximum(setmax[nn])
        #if not (nn < 3) : c1.SetLogy(1)
        t = TLatex()
        #TLatex t(.1,.9,channels[nn]);
        t.SetNDC(kTRUE);
        #t.SetTextFont(32)
        t.SetTextColor(1)
        t.SetTextSize(0.04)
        t.SetTextAlign(12)
        t.DrawLatex( .15,.85,labels[nn] )
        #t.DrawLatex( .15,.8,"(ttH X MC-only BKG + fakes)" )
        #t.DrawLatex( .15,.75,"" )
        leg.Draw("same")

        t = TLatex()
        #TLatex t(.1,.9,channels[nn]);
        t.SetNDC(kTRUE);
        #t.SetTextFont(32)
        t.SetTextColor(1)
        t.SetTextSize(0.04)
        t.SetTextAlign(12)
        t.DrawLatex( .15,.85,labels[nn] )
        #t.DrawLatex( .15,.8,"2017 conditions" )
        #t.DrawLatex( .15,.75,"LepMVA 0.9" )

        c1.SaveAs('shapeComp_'+channel+'.pdf')


        file.Close()
        file2.Close()

###################################
##
####################################

mvaVariables = [
    "mvaInputs_3l",
    "mvaInputs_2lss",
    "mvaInputs_2lss",
    "mvaInputs_2lss", # 2l2t
    "mvaInputs_HTT_sum",
    "mvaInputs_HTT_sum"
]

BDTcomponents=["signal", "TTZ", "TTW", "EWK", "Rares", "fakes_data","tH", "TT"]
BDTVar=[
   ["lep1_conePt", "lep2_conePt", "mindr_lep1_jet", "max_lep_eta", "mindr_tau_jet",
    "ptmiss", "tau_pt", "dr_leps", "mTauTauVis1", "mTauTauVis2", "mbb_loose", "nJet"],
   ["avg_dr_jet", "dr_lep1_tau", "dr_lep2_tau", "dr_leps", "lep2_conePt",
    "mT_lep1", "mT_lep2", "mTauTauVis2", "max_lep_eta",
    "mbb", "mindr_lep1_jet", "mindr_lep2_jet", "mindr_tau_jet",
    "nJet", "ptmiss", "tau_pt",
    "HTT", "HadTop_pt"],
   ["avg_dr_jet", "dr_lep1_tau", "dr_lep2_tau", "dr_leps", "lep2_conePt",
    "mT_lep1", "mT_lep2", "mTauTauVis2", "max_lep_eta",
    "mbb_loose",
    "mindr_lep1_jet", "mindr_lep2_jet", "mindr_tau_jet",
    "nJet", "ptmiss", "tau_pt",
    "HTT", "HadTop_pt"],
   ["mTauTauVis", "cosThetaS_hadTau", "tau1_pt", "tau2_pt",
    "lep2_conePt", "mindr_lep1_jet", "mT_lep1", "mindr_tau_jet",
    "avg_dr_jet", "avr_dr_lep_tau", "dr_taus", "is_OS", "nBJetLoose"],
   ["avg_dr_jet", "dr_taus", "ptmiss", "lep_conePt", "mT_lep", "mTauTauVis", "mindr_lep_jet",
    "mindr_tau1_jet", "mindr_tau2_jet", "dr_lep_tau_ss", "dr_lep_tau_lead",
    "costS_tau", "nBJetLoose", "tau1_pt",
    "tau2_pt", "HTT", "HadTop_pt"],
    ["avg_dr_jet", "dr_taus", "ptmiss", "lep_conePt", "mT_lep", "mTauTauVis", "mindr_lep_jet",
    "mindr_tau1_jet", "mindr_tau2_jet", "dr_lep_tau_ss", "dr_lep_tau_lead",
    "costS_tau", "nBJetLoose", "tau1_pt",
    "tau2_pt", "HTT", "HadTop_pt"]
]

if not doRoc :
    for nn, channel in enumerate(channels) :
        if not nn == 4 : continue
        file=TFile(repo2017[nn])
        #BDTcomponents=["signal", "TTZ", "TTW", "EWK", "Rares", "fakes_data","tH", "TT"]
        if mvaVariables[nn] != "" :
            for ii, var in enumerate(BDTVar[nn]) :
                #if not var in ["HTT", "HadTop_pt"] : continue
                #if ii > 0 : continue
                print repo2017[nn]
                print repoHistos[nn]+mvaVariables[nn]+'/signal/'+var
                signal_mvaOutput = file.Get(repoHistos[nn]+mvaVariables[nn]+'/signal/'+var)
                TTW_mvaOutput = file.Get(repoHistos[nn]+mvaVariables[nn]+'/TTW/'+var)
                TTZ_mvaOutput = file.Get(repoHistos[nn]+mvaVariables[nn]+'/TTZ/'+var)
                EWK_mvaOutput = file.Get(repoHistos[nn]+mvaVariables[nn]+'/EWK/'+var)
                Rares_mvaOutput = file.Get(repoHistos[nn]+mvaVariables[nn]+'/Rares/'+var)
                AllBKG = TTW_mvaOutput + TTZ_mvaOutput + EWK_mvaOutput + Rares_mvaOutput
                fakes_data_mvaOutput = file.Get(repoHistos[nn]+mvaVariables[nn]+'/fakes_data/'+var)
                TT_mvaOutput = file.Get(repoHistos[nn]+mvaVariables[nn]+'/TT/'+var)
                xl1=.6
                yl1=0.65
                xl2=xl1+.3
                yl2=yl1+.25;
                leg = TLegend(xl1,yl1,xl2,yl2);
                c1 = TCanvas( 'c1', '@ Analysis level ',  400, 400 )
                signal_mvaOutput.Scale(1./signal_mvaOutput.Integral())
                signal_mvaOutput.SetLineColor(8)
                signal_mvaOutput.SetLineWidth(3)
                if var == "HTT" : signal_mvaOutput.Rebin(5)
                elif var not in ["nBJetLoose","nJet", "mTauTauVis","mTauTauVis1","mTauTauVis2", "is_OS", "HadTop_pt"] : signal_mvaOutput.Rebin(2)
                signal_mvaOutput.SetTitle("  ")
                #signal_mvaOutput.SetLineStyle(10)
                signal_mvaOutput.GetXaxis().SetTitle(var)
                leg.AddEntry(signal_mvaOutput,"ttH ","l")
                signal_mvaOutput.Draw()
                signal_mvaOutput.Draw("hist,same")

                AllBKG.Scale(1./AllBKG.Integral())
                AllBKG.SetLineColor(kMagenta)
                AllBKG.SetLineWidth(3)
                if var == "HTT" : AllBKG.Rebin(5)
                elif var not in ["nBJetLoose","nJet", "mTauTauVis","mTauTauVis1","mTauTauVis2", "is_OS", "HadTop_pt"] : AllBKG.Rebin(2)
                #AllBKG.SetLineStyle(10)
                leg.AddEntry(AllBKG,"MC-only BKG","l")
                AllBKG.Draw("same")
                AllBKG.Draw("hist,same")

                if not channels[nn] in ["3l_1tau","2l_2tau"]:
                    fakes_data_mvaOutput.Scale(1./fakes_data_mvaOutput.Integral())
                    fakes_data_mvaOutput.SetLineColor(9)
                    fakes_data_mvaOutput.SetLineWidth(3)
                    if var == "HTT" : fakes_data_mvaOutput.Rebin(5)
                    elif var not in ["nBJetLoose","nJet", "mTauTauVis","mTauTauVis1","mTauTauVis2", "is_OS", "HadTop_pt"] : fakes_data_mvaOutput.Rebin(2)

                    #fakes_data_mvaOutput.SetLineStyle(10)
                    leg.AddEntry(fakes_data_mvaOutput,"fakes","l")
                    fakes_data_mvaOutput.Draw("same")
                    fakes_data_mvaOutput.Draw("hist,same")

                if var in ["avg_dr_jet","avr_dr_lep_tau", "mTauTauVis","mT_lep1", "HadTop_pt", "mindr_tau_jet"] : signal_mvaOutput.SetMaximum(0.40)
                elif var in ["cosThetaS_hadTau"] : signal_mvaOutput.SetMaximum(0.30)
                elif var in ["tau1_pt", "tau_pt", "lep2_conePt", "HTT"] : signal_mvaOutput.SetMaximum(0.50)
                elif var in ["tau2_pt"] :
                    signal_mvaOutput.SetMaximum(1.1)
                    signal_mvaOutput.GetXaxis().SetRangeUser(10.,120.)
                elif var in ["max_lep_eta"] : signal_mvaOutput.GetXaxis().SetRangeUser(0.,2.5)
                elif var in ["costS_tau"] : signal_mvaOutput.SetMaximum(0.1)
                elif var in ["nBJetLoose","nJet","mbb_loose", "is_OS"] : signal_mvaOutput.SetMaximum(1.0)
                else : signal_mvaOutput.SetMaximum(0.25)

                t = TLatex()
                #TLatex t(.1,.9,channels[nn]);
                t.SetNDC(kTRUE);
                #t.SetTextFont(32)
                t.SetTextColor(1)
                t.SetTextSize(0.04)
                t.SetTextAlign(12)
                t.DrawLatex( .15,.85,labels[nn] )
                t.DrawLatex( .15,.8,"2017 conditions" )
                t.DrawLatex( .15,.75,"LepMVA 0.9" )

                leg.Draw("same")
                c1.SaveAs(channels[nn]+'/shapeComp_'+var+'.pdf')
                c1.Clear()

                if var == "HTT" :
                    xl1=.4
                    yl1=0.15
                    xl2=xl1+.5
                    yl2=yl1+.25;
                    leg3 = TLegend(xl1,yl1,xl2,yl2);
                    roc_HTT_2016=CalcROC(fakes_data_mvaOutput, signal_mvaOutput)
                    roc_HTT_2016.SetTitle("  ")
                    roc_HTT_2016.SetLineColor(kMagenta)
                    roc_HTT_2016.SetLineWidth(3)
                    leg3.AddEntry(roc_HTT_2016,"against fakes","l")
                    roc_HTT_2016.Draw()
                    roc_HTT_MC_2016=CalcROC(AllBKG, signal_mvaOutput)
                    roc_HTT_MC_2016.SetLineColor(8)
                    roc_HTT_MC_2016.SetLineWidth(3)
                    leg3.AddEntry(roc_HTT_MC_2016,"against MC-only","l")
                    roc_HTT_MC_2016.Draw("same")
                    leg3.Draw("same")
                    roc_HTT_2016.GetXaxis().SetRangeUser(0.,1.)
                    roc_HTT_2016.GetYaxis().SetRangeUser(0.,1.)
                    roc_HTT_2016.GetXaxis().SetTitle("False positive rate")
                    roc_HTT_2016.GetYaxis().SetTitle("True positive rate")
                    t.DrawLatex( .15,.85,labels[nn]+" response of HTT discr." )
                    t.DrawLatex( .15,.8,"2017 conditions" )
                    t.DrawLatex( .15,.75,"LepMVA 0.9" )
                    c1.SaveAs(channels[nn]+'/rocComp_'+var+'.pdf')
                c1.Close()

"""
WriteDatacards_2l_2tau --input_file=/home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun02/datacards_2l2tau_41p53invfb_Binned_2018jun02.root --output_file=/home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun02/ttH_2l2tau_41p53invfb_Binned_2018jun02.root --add_shape_sys=false

combine -M Asymptotic -m 125 -t -1 /home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun02/ttH_2l2tau_41p53invfb_Binned_2018jun02.txt


mkdir GoF; cd GoF
combineTool.py  -M T2W -i /home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun02/ttH_2l2tau_41p53invfb_Binned_2018jun02.txt
combine -M MaxLikelihoodFit -d /home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun02/ttH_2l2tau_41p53invfb_Binned_2018jun02.txt -t -1
python $CMSSW_BASE/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py -a mlfit.root -g plots.root
combineTool.py -M GoodnessOfFit --algorithm saturated -d ../ttH_noHTT_noHTT_SUM_M_nbin_5_quantiles.root -n .saturated
combineTool.py -M GoodnessOfFit --algorithm saturated -d ../ttH_noHTT_noHTT_SUM_M_nbin_5_quantiles.root -n .saturated.toys -t 200 -s 0:4:1 --parallel 5
combineTool.py -M CollectGoodnessOfFit --input higgsCombine.saturated.GoodnessOfFit.mH120.root higgsCombine.saturated.toys.GoodnessOfFit.mH120.*.root -o GoF_saturated.json
$CMSSW_BASE/src/CombineHarvester/CombineTools/scripts/plotGof.py --statistic saturated --mass 120.0 GoF_saturated.json -o GoF_saturated

mkdir impacts; cd impacts
combineTool.py -M Impacts -m 125 -d /home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun02/ttH_2lss1tau_41p53invfb_Binned_2018jun02.root --expectSignal 1 --allPars --parallel 8 -t -1 --doInitialFit
combineTool.py -M Impacts -m 125 -d /home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun02/ttH_2lss1tau_41p53invfb_Binned_2018jun02.root --expectSignal 1 --allPars --parallel 8 -t -1 --robustFit 1 --doFits
combineTool.py -M Impacts -m 125 -d /home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun02/ttH_2lss1tau_41p53invfb_Binned_2018jun02.root -o impacts.json
plotImpacts.py -i impacts.json -o  impacts


For pre/post fit plots
PostFitShapes -d /home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun02/ttH_2lss1tau_41p53invfb_Binned_2018jun02.txt -o /home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun02/ttH_2lss1tau_41p53invfb_Binned_2018jun02_shapes.root -m 125

root -l -b -n -q /home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"2lss1tau_41p53invfb_Binned_2018jun02\",\"2018jun02/\",\"2l_2tau\",\"/home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/\",false,false,\"BDT\",\"\",0.0,10.0)

WriteDatacards_2lss_1tau --input_file=/home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun02/datacards_2lss1tau_41p53invfb_Binned_2018jun02.root --output_file=/home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun02/ttH_2lss1tau_41p53invfb_Binned_2018jun02.root --add_shape_sys=true

combine -M Asymptotic -m 125 -t -1 /home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun02/ttH_2lss1tau_41p53invfb_Binned_2018jun02.txt

combineTool.py  -M T2W -i /home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun02/ttH_2lss1tau_41p53invfb_Binned_2018jun02.txt

combine -M Asymptotic -m 125 -t -1 /home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/2018jun02/ttH_2lss1tau_41p53invfb_Binned_2018jun02.txt



root -l -b -n -q /home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\"2l2tau_41p53invfb_Binned_2018jun02\",\"2018jun02/\",\"2l_2tau\",\"/home/acaan/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/\",false,false,\"BDT\",\"\",0.0,20.0)
"""
