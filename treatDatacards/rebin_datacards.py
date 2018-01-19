#!/usr/bin/env python
import os, subprocess, sys
workingDir = os.getcwd()

from array import array
from ROOT import *
from math import sqrt, sin, cos, tan, exp
import numpy as np
from pathlib2 import Path
execfile("../python/data_manager.py")

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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
channel="2lss_1tau"
label="2018Jan13_VHbb_addMEM"
year="2016"
bdtTypes=["tt","ttV","1B","2MEM","2HTT"]
sources=[]
bdtTypesToDo=[]

import shutil,subprocess
proc=subprocess.Popen(['mkdir '+options.channel+"_"+label],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()
proc=subprocess.Popen(['mkdir '+options.channel+"_"+label+"/"+options.variables],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()

#for test in [1000,900,800,700,600,500,400,300,200,100] : print (test, list(divisorGenerator(test)) )

#source="/home/acaan/ttHAnalysis/2016/2017Dec21-limFirstOpt/datacards/2lss_1tau/prepareDatacards_2lss_1tau_sumSS_"
# 2017Dec-VHbb-wMEM-LooseLepMedTau
sourceoriginal="/home/"+user+"/ttHAnalysis/"+year+"/"+label+"/datacards/"+channel+"/prepareDatacards_"+channel+"_sumOS_"
source=options.channel+"_"+label+"/"+options.variables+"/prepareDatacards_"+channel+"_sumOS_"
local=workingDir+"/"+options.channel+"_"+label+"/"+options.variables+"/"
originalBinning=600
nbinRegular=np.arange(1,20) #list(divisorGenerator(originalBinning)) #np.arange(2,15) #[20,15,12,8,6,5,4,3]
nbinQuant= np.arange(1,20)
counter=0
if options.variables=="MEM" :
    #source= source+"memOutput_LR"+".root"
    my_file = Path(sourceoriginal+"memOutput_LR.root")
    if my_file.exists() :
        proc=subprocess.Popen(['cp '+sourceoriginal+"memOutput_LR.root "+options.channel+"_"+label+"/"+options.variables ],shell=True,stdout=subprocess.PIPE)
        out = proc.stdout.read()
        sources = sources + [source+"memOutput_LR"]
        bdtTypesToDo = bdtTypesToDo +["1D"]
        print ("rebinning ",sources[counter])
    else : print ("does not exist ",source+"memOutput_LR.root")
else  :
    for bdtType in bdtTypes :
        my_file = Path(sourceoriginal+"mvaOutput_2lss_"+options.variables+"_"+bdtType+".root")
        if my_file.exists() :
            proc=subprocess.Popen(['cp '+sourceoriginal+"mvaOutput_2lss_"+options.variables+"_"+bdtType+".root "+options.channel+"_"+label+"/"+options.variables ],shell=True,stdout=subprocess.PIPE)
            out = proc.stdout.read()
            sources = sources + [source+"mvaOutput_2lss_"+options.variables+"_"+bdtType]
            bdtTypesToDo = bdtTypesToDo +[bdtType]
            print (sources[counter],"rebinning ")
            counter=counter+1
        else : print (sourceoriginal+"mvaOutput_2lss_"+options.variables+"_"+bdtType+".root","does not exist ")

print ("I will rebin",bdtTypesToDo,"(",len(sources),") BDT options")

if options.BINtype == "regular" or options.BINtype == "ranged" : binstoDo=nbinRegular
if options.BINtype == "quantiles" : binstoDo=nbinQuant
if options.BINtype == "none" : binstoDo=np.arange(1,originalBinning)

if not doLimits:
    #########################################
    ## make rebinned datacards
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.title(options.BINtype+" binning "+options.variables)
    colorsToDo=['r','g','b','m','y','c']
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
        plt.plot(binstoDo,errOcont[0], colorsToDo[nn]+'-')
        plt.plot(binstoDo,errOcont[0], colorsToDo[nn]+'o-') # ,label=bdtTypesToDo[nn]
        plt.plot(binstoDo,errOcont[2], colorsToDo[nn]+'x--')
    ax.set_xlabel('nbins')
    if options.BINtype == "regular" : maxplot =1.0
    else : maxplot =1.0 # 0.35
    plt.axis((min(binstoDo),max(binstoDo),0,maxplot))
    line_up, = plt.plot(binstoDo, 'o-', color='k',label="fake-only")
    line_down, = ax.plot(binstoDo, 'x--', color='k',label="fake+ttV+EWK")
    legend1 = plt.legend(handles=[line_up, line_down], loc='best')
    ax.set_ylabel('err/content last bin')
    #ax.legend(loc='best', fancybox=False, shadow=False, ncol=1) #, ncol=3)
    plt.grid(True)
    if options.BINtype == "none" : namefig=options.channel+'_'+label+'/'+options.variables+'/'+options.variables+'_fullsim_ErrOcont_none.pdf'
    if options.BINtype == "quantiles" : namefig=options.channel+'_'+label+'/'+options.variables+'/'+options.variables+'_fullsim_ErrOcont_quantiles.pdf'
    if options.BINtype == "regular": namefig=options.channel+'_'+label+'/'+options.variables+'/'+options.variables+'_fullsim_ErrOcont.pdf'
    if options.BINtype == "ranged" : namefig=options.channel+'_'+label+'/'+options.variables+'/'+options.variables+'_fullsim_ErrOcont_ranged.pdf'
    fig.savefig(namefig)
    print ("saved",namefig)
    #########################################
    ## plot quantiles boundaries
    #print ("quantiles", lastQuant)
    #print ("Pmaxbin", xmaxQuant)
    if options.BINtype == "quantiles" :
        fig, ax = plt.subplots(figsize=(5, 5))
        plt.title(options.BINtype+" binning "+options.variables)
        colorsToDo=['r','g','b','m','y','c']
        for nn,source in enumerate(sources) :
            print (len(binstoDo),len(lastQuant[nn]))
            plt.plot(binstoDo,lastQuant[nn], colorsToDo[nn]+'-')
            plt.plot(binstoDo,lastQuant[nn], colorsToDo[nn]+'o-') # ,label=bdtTypesToDo[nn]
            plt.plot(binstoDo,xmaxQuant[nn], colorsToDo[nn]+'x--')
            plt.plot(binstoDo,xminQuant[nn], colorsToDo[nn]+'.--')
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
if doLimits :
    print "do limits"
    print sources
    fig, ax = plt.subplots(figsize=(5, 5))
    #plt.title(options.BINtype+" binning")
    colorsToDo=['r','g','b','m','y','c']
    for nn,source in enumerate(sources) :
        limits=ReadLimits(bdtTypesToDo[nn], binstoDo, options.BINtype,originalBinning,local)
        print (len(binstoDo),len(limits[0]))
        plt.plot(binstoDo,limits[0], colorsToDo[nn]+'o-',label=bdtTypesToDo[nn])
        plt.plot(binstoDo,limits[1], colorsToDo[nn]+'-')
        plt.plot(binstoDo,limits[3], colorsToDo[nn]+'-')
    ax.legend(loc='best', fancybox=False, shadow=False , ncol=1)
    ax.set_xlabel('nbins')
    ax.set_ylabel('limits')
    plt.text(2.3, 2.4, options.BINtype+" binning "+" "+options.variables )
    plt.text(2.3, 2.53, "CMS"  ,  fontweight='bold' )
    plt.text(4.3, 2.53, "preliminary" )
    plt.text(max(binstoDo)-6.0, 2.53, "35.9/fb (13 TeV)"   )
    plt.axis((min(binstoDo),max(binstoDo),0.7,2.5))
    if options.BINtype == "quantiles" : namefig=options.channel+'_'+label+'/'+options.variables+'/'+options.variables+'_fullsim_limits_quantiles.pdf'
    if options.BINtype == "regular": namefig=options.channel+'_'+label+'/'+options.variables+'/'+options.variables+'_fullsim_limits.pdf'
    if options.BINtype == "ranged" : namefig=options.channel+'_'+label+'/'+options.variables+'/'+options.variables+'_fullsim_limits_ranged.pdf'
    fig.savefig(namefig)
    print ("saved",namefig)
