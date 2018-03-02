#!/usr/bin/env python
import os, subprocess, sys
workingDir = os.getcwd()

from ROOT import *
from math import sqrt, sin, cos, tan, exp
import numpy as np
from pathlib2 import Path
execfile("../python/data_manager.py")

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# ./rebin_datacards.py --channel "2lss_1tau" --variables "HTT" --BINtype "quantiles" --doLimits
# ./rebin_datacards.py --channel "2l_2tau" --variables "mTauTauVis" --BINtype "mTauTauVis"
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--channel ", type="string", dest="channel", help="The ones whose variables implemented now are:\n   - 1l_2tau\n   - 2lss_1tau\n It will create a local folder and store the report*/xml", default="2lss_1tau")
parser.add_option("--variables", type="string", dest="variables", help="  Set of variables to use -- it shall be put by hand in the code", default=1000)
parser.add_option("--BDTtype", type="string", dest="BDTtype", help="Variable set", default="1B")
parser.add_option("--BINtype", type="string", dest="BINtype", help="regular / ranged / quantiles", default="regular")
parser.add_option("--doPlots", action="store_true", dest="doPlots", help="If you call this will not do plots with repport", default=False)
parser.add_option("--doLimits", action="store_true", dest="doLimits", help="If you call this will not do plots with repport", default=False)
(options, args) = parser.parse_args()

doLimits=options.doLimits
doPlots=options.doPlots
user="acaan"
year="2016"
channel=options.channel
if channel == "2lss_1tau" :
    label="2lss_1tau_2018Feb28_VHbb_TLepMTau_shape" #"2lss_1tau_2018Feb26_VHbb_TLepTTau"
    bdtTypes=["tt","ttV","SUM_T","SUM_M","1B_T","1B_M"] #,"2MEM","2HTT"]
if channel == "1l_2tau" :
    label= "1l_2tau_2018Feb28_VHbb_TLepTTau_shape" #"1l_2tau_2018Feb08_VHbb_TightTau" # "1l_2tau_2018Feb02_VHbb_VTightTau" #  "1l_2tau_2018Jan30_VHbb_VVTightTau" # "1l_2tau_2018Jan30_VHbb" # "1l_2tau_2018Jan30_VHbb_VTightTau" #
    bdtTypes=["ttbar","ttV","SUM_T","SUM_VT","1B_T","1B_VT"] #"1B"] #
if channel == "2l_2tau" :
    label= "2l_2tau_2018Feb20_VHbb_TLepVTTau" #
    bdtTypes= ["tt","ttV","SUM_M","SUM_T","SUM_VT","1B_M","1B_T","1B_VT"] #[] #,,

sources=[]
bdtTypesToDo=[]
bdtTypesToDoLabel=[]
bdtTypesToDoFile=[]

import shutil,subprocess
proc=subprocess.Popen(["mkdir "+options.channel+"_"+label],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()
proc=subprocess.Popen(["mkdir "+options.channel+"_"+label+"/"+options.variables],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()
#for test in [1000,900,800,700,600,500,400,300,200,100] : print (test, list(divisorGenerator(test)) )
mom="/home/"+user+"/ttHAnalysis/"+year+"/"+label+"/datacards/"+channel

local=workingDir+"/"+options.channel+"_"+label+"/"+options.variables+"/"
originalBinning=100
nbinRegular=np.arange(1,20) #list(divisorGenerator(originalBinning))
nbinQuant= np.arange(1,19)
counter=0

if channel == "2lss_1tau" :
    sourceoriginal=mom+"/prepareDatacards_"+channel+"_sumOS_"
    source=local+"/prepareDatacards_"+channel+"_sumOS_"
    print sourceoriginal
    if options.variables=="MEM" :
        my_file = Path(sourceoriginal+"memOutput_LR.root")
        if my_file.exists() :
            proc=subprocess.Popen(["cp "+sourceoriginal+"memOutput_LR.root "+local ],shell=True,stdout=subprocess.PIPE)
            out = proc.stdout.read()
            sources = sources + [source+"memOutput_LR"]
            bdtTypesToDo = bdtTypesToDo +["1D"]
            print ("rebinning ",sources[counter])
        else : print ("does not exist ",sourceoriginal+"memOutput_LR.root")
    else  :
        for bdtType in bdtTypes :
            my_file = Path(sourceoriginal+"mvaOutput_2lss_"+options.variables+"_"+bdtType+".root")
            if my_file.exists() :
                proc=subprocess.Popen(["cp "+
                    sourceoriginal+ "mvaOutput_2lss_"+options.variables+"_"+bdtType+".root " +local
                    ],shell=True,stdout=subprocess.PIPE)
                out = proc.stdout.read()
                sources = sources + [source+"mvaOutput_2lss_"+options.variables+"_"+bdtType]
                bdtTypesToDo = bdtTypesToDo +[bdtType]
                print (sources[counter],"rebinning ")
                counter=counter+1
            else : print (sourceoriginal+"mvaOutput_2lss_"+options.variables+"_"+bdtType+".root","does not exist ")
    bdtTypesToDoLabel=bdtTypesToDo
    bdtTypesToDoFile=bdtTypesToDo
if channel == "1l_2tau" or channel == "2l_2tau" :
    sourceoriginal=mom+"/prepareDatacards_"+channel+"_"
    source=local+"/prepareDatacards_"+channel+"_"
    if options.variables=="oldTrain" :
        oldVar=["1l_2tau_ttbar_Old", "1l_2tau_ttbar_OldVar","ttbar_OldVar"] # "1l_2tau_ttbar",
        typeBDT=[ "oldTrainM","oldVar loose lep" ,"oldVar tight lep"] # "oldTrain",
        for ii,nn in enumerate(oldVar) :
            my_file = Path(sourceoriginal+"mvaOutput_"+nn+".root")
            print sourceoriginal+nn+".root"
            if my_file.exists() :
                proc=subprocess.Popen(['cp '+sourceoriginal+"mvaOutput_"+nn+".root "+local],shell=True,stdout=subprocess.PIPE)
                out = proc.stdout.read()
                sources = sources + [source+"mvaOutput_"+nn]
                bdtTypesToDo = bdtTypesToDo +["1D"]
                bdtTypesToDoLabel = bdtTypesToDoLabel +[typeBDT[ii]]
                bdtTypesToDoFile=bdtTypesToDoFile+[oldVar[ii]]
                print ("rebinning ",sources[counter])
            else : print ("does not exist ",source+nn)
    elif "HTT" in options.variables :
        for bdtType in bdtTypes :
            #if channel == "2l_2tau" :
            fileName=options.variables+"_"+bdtType
            #elif channel == "1l_2tau" : fileName=bdtType+"_"+options.variables
            my_file = Path(sourceoriginal+"mvaOutput_"+fileName+".root")
            if my_file.exists() :
                proc=subprocess.Popen(["cp "+sourceoriginal+"mvaOutput_"+fileName+".root " +local],shell=True,stdout=subprocess.PIPE)
                out = proc.stdout.read()
                sources = sources + [source+"mvaOutput_"+fileName]
                bdtTypesToDo = bdtTypesToDo +[bdtType]
                bdtTypesToDoFile=bdtTypesToDoFile+[fileName]
                print (sources[counter],"rebinning ")
                counter=counter+1
            else : print (sourceoriginal+"mvaOutput_"+fileName+".root","does not exist ")
        bdtTypesToDoLabel=bdtTypesToDo
    elif "mTauTauVis" in options.variables :
        my_file = Path(sourceoriginal+options.variables+".root")
        if my_file.exists() :
            proc=subprocess.Popen(["cp "+sourceoriginal+options.variables+".root " +local],shell=True,stdout=subprocess.PIPE)
            out = proc.stdout.read()
            sources = sources + [source+options.variables]
            bdtTypesToDo = bdtTypesToDo +[options.variables]
            bdtTypesToDoFile=bdtTypesToDoFile+[options.variables]
            print (sources[counter],"rebinning ")
            counter=counter+1
        else : print (sourceoriginal+options.variables+".root","does not exist ")
    else : print ("options",channel,options.variables,"are not compatible")


print ("I will rebin",bdtTypesToDoLabel,"(",len(sources),") BDT options")

if options.BINtype == "regular" or options.BINtype == "ranged" or options.BINtype == "mTauTauVis" : binstoDo=nbinRegular
if options.BINtype == "quantiles" : binstoDo=nbinQuant
if options.BINtype == "none" : binstoDo=np.arange(1,originalBinning)

colorsToDo=['r','g','b','m','y','c', 'fuchsia', "peachpuff"] #['r','g','b','m','y','c','k']
if not doLimits:
    #########################################
    ## make rebinned datacards
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.title(options.BINtype+" binning "+options.variables)
    lastQuant=[]
    xmaxQuant=[]
    xminQuant=[]
    for nn,source in enumerate(sources) :
        errOcont=rebinRegular(source, binstoDo, options.BINtype,originalBinning,doPlots,options.variables,bdtTypesToDo[nn])
        print bdtTypesToDo[nn]
        #print ("                 ",nbinRegular)
        #print ("TT-only,  last bin", errOcont[0])
        #print ("TT-only, Plast bin", errOcont[1])
        #print ("TT+TTV,   last bin", errOcont[2])
        #print ("TT+TTV,  Plast bin", errOcont[3])
        #print ("last quantile",len(errOcont[4]),errOcont[4][len(errOcont[4])-1],errOcont[4][len(errOcont[4])-2])
        lastQuant=lastQuant+[errOcont[4]]
        xmaxQuant=xmaxQuant+[errOcont[5]]
        xminQuant=xminQuant+[errOcont[6]]
        #
        plt.plot(binstoDo,errOcont[0], color=colorsToDo[nn],linestyle='-') # ,label=bdtTypesToDo[nn]
        plt.plot(binstoDo,errOcont[0], color=colorsToDo[nn],linestyle='-',marker='o',label=bdtTypesToDo[nn]) #
        #plt.plot(binstoDo,errOcont[2], color=colorsToDo[nn],linestyle='--',marker='x')
    ax.set_xlabel('nbins')
    if options.BINtype == "regular" : maxplot =1.0
    #elif options.BINtype == "mTauTauVis" : maxplot=200.
    else : maxplot =1.0 # 0.35
    plt.axis((min(binstoDo),max(binstoDo),0,maxplot))
    line_up, = plt.plot(binstoDo,linestyle='-',marker='o', color='k',label="fake-only")
    #line_down, = ax.plot(binstoDo,linestyle='--',marker='x', color='k',label="fake+ttV+EWK")
    legend1 = plt.legend(handles=[line_up], loc='best') # , line_down
    ax.set_ylabel('err/content last bin')
    ax.legend(loc='best', fancybox=False, shadow=False, ncol=1) #, ncol=3)
    plt.grid(True)
    if options.BINtype == "none" : namefig=options.channel+'_'+label+'/'+options.variables+'/'+options.variables+'_fullsim_ErrOcont_none.pdf'
    if options.BINtype == "quantiles" : namefig=options.channel+'_'+label+'/'+options.variables+'/'+options.variables+'_fullsim_ErrOcont_quantiles.pdf'
    if options.BINtype == "regular" or options.BINtype == "mTauTauVis": namefig=options.channel+'_'+label+'/'+options.variables+'/'+options.variables+'_fullsim_ErrOcont.pdf'
    if options.BINtype == "ranged" : namefig=options.channel+'_'+label+'/'+options.variables+'/'+options.variables+'_fullsim_ErrOcont_ranged.pdf'
    fig.savefig(namefig)
    print ("saved",namefig)
    #########################################
    ## plot quantiles boundaries
    if options.BINtype == "quantiles" :
        fig, ax = plt.subplots(figsize=(5, 5))
        plt.title(options.BINtype+" binning "+options.variables)
        #colorsToDo=['r','g','b','m','y','c']
        for nn,source in enumerate(sources) :
            print (len(binstoDo),len(lastQuant[nn]))
            plt.plot(binstoDo,lastQuant[nn], color=colorsToDo[nn],linestyle='-')
            plt.plot(binstoDo,lastQuant[nn], color=colorsToDo[nn],linestyle='-',marker='o') # ,label=bdtTypesToDo[nn]
            plt.plot(binstoDo,xmaxQuant[nn], color=colorsToDo[nn],linestyle='--',marker='x')
            plt.plot(binstoDo,xminQuant[nn], color=colorsToDo[nn],linestyle='--',marker='.')
        ax.set_xlabel('nbins')
        plt.axis((min(binstoDo),max(binstoDo),0,1.0))
        line_up, = plt.plot(binstoDo, 'o-', color='k',label="last bin low")
        line_down, = ax.plot(binstoDo, 'x--', color='k',label="Max")
        line_d, = ax.plot(binstoDo, '.--', color='k',label="Min")
        legend1 = plt.legend(handles=[line_up, line_down, line_d], loc='best')
        ax.set_ylabel('boundary')
        #ax.legend(loc='best', fancybox=False, shadow=False, ncol=2)
        plt.grid(True)
        fig.savefig(options.channel+"_"+label+"/"+options.variables+'/'+options.variables+'_fullsim_boundaries_quantiles.pdf')

#########################################
## make limits
print sources
if doLimits :
    print "do limits"
    print sources
    fig, ax = plt.subplots(figsize=(5, 5))
    #plt.title(options.BINtype+" binning")
    #colorsToDo=['r','g','b','m','y','c', 'fuchsia']
    for nn,source in enumerate(sources) :
        #options.variables+'_'+bdtTypesToDoFile[ns]+'_nbin_'+str(nbins)
        limits=ReadLimits(bdtTypesToDoFile[nn], binstoDo, options.BINtype,channel,local,0,0)
        print (len(binstoDo),len(limits[0]))
        plt.plot(binstoDo,limits[0], color=colorsToDo[nn],linestyle='-',marker='o',label=bdtTypesToDoLabel[nn])
        plt.plot(binstoDo,limits[1], color=colorsToDo[nn],linestyle='-')
        plt.plot(binstoDo,limits[3], color=colorsToDo[nn],linestyle='-')
    ax.legend(loc='best', fancybox=False, shadow=False , ncol=1)
    ax.set_xlabel('nbins')
    ax.set_ylabel('limits')
    maxsum=0
    if channel=="2lss_1tau" : plt.axis((min(binstoDo),max(binstoDo),0.7,2.5))
    if channel=="1l_2tau" :
        plt.axis((min(binstoDo),max(binstoDo),2.0,6.5))
        #plt.yscale('log')
        maxsum=5
    plt.text(2.3, 2.4, options.BINtype+" binning "+" "+options.variables )
    plt.text(2.3, 2.53+maxsum, "CMS"  ,  fontweight='bold' )
    plt.text(4.3, 2.53+maxsum, "preliminary" )
    plt.text(max(binstoDo)-6.0, 2.53+maxsum, "35.9/fb (13 TeV)"   )
    if options.BINtype == "quantiles" : namefig=options.channel+'_'+label+'/'+options.variables+'/'+options.variables+'_fullsim_limits_quantiles.pdf'
    if options.BINtype == "regular" or options.BINtype == "mTauTauVis": namefig=options.channel+'_'+label+'/'+options.variables+'/'+options.variables+'_fullsim_limits.pdf'
    if options.BINtype == "ranged" : namefig=options.channel+'_'+label+'/'+options.variables+'/'+options.variables+'_fullsim_limits_ranged.pdf'
    fig.savefig(namefig)
    print ("saved",namefig)
