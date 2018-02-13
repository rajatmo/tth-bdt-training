#!/usr/bin/env python
import os, subprocess, sys
workingDir = os.getcwd()
from array import array
from ROOT import *
from math import sqrt, sin, cos, tan, exp

# python do_limits_2D.py --channel "2lss_1tau" --variables "noHTT" --doLimits
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--channel ", type="string", dest="channel", help="The ones whose variables implemented now are:\n   - 1l_2tau\n   - 2lss_1tau\n It will create a local folder and store the report*/xml", default="2lss_1tau")
parser.add_option("--variables", type="string", dest="variables", help="  Set of variables to use -- it shall be put by hand in the code", default=1000)
parser.add_option("--doPlots", action="store_true", dest="doPlots", help="If you call this will not do plots with repport", default=False)
parser.add_option("--do1D", action="store_true", dest="do1D", help="If you call this will not do plots with repport", default=False)
parser.add_option("--doLimits", action="store_true", dest="doLimits", help="If you call this will not do plots with repport", default=False)
parser.add_option("--plotLimits", action="store_true", dest="plotLimits", help="If you call this will not do plots with repport", default=False)
(options, args) = parser.parse_args()

user="acaan"
channel=options.channel
year="2016"

# 1l_2tau_2018Feb08_VHbb_VTightTau


if channel == "2lss_1tau" :
    label="2018Jan13_VHbb_addMEM"
    mom="/home/"+user+"/ttHAnalysis/"+year+"/"+label+"/datacards/"+channel
    sourceoriginal=mom+"/prepareDatacards_"+channel+"_sumOS_"
    source=workingDir+"/"+options.channel+"_"+label+"/"+options.variables+"/prepareDatacards_"+channel+"_sumOS_"
    nStart= [15,20]
    nTarget= [5,6,7,8,9,10]
if channel == "1l_2tau" :
    label= "1l_2tau_2018Fev12_TightTau/" #"1l_2tau_2018Feb01_VHbb_VTightTau" #"1l_2tau_2018Jan30_VHbb_VVTightTau" #"1l_2tau_2018Jan30_VHbb_VTightTau" # "1l_2tau_2018Jan30_VHbb"
    mom="/home/"+user+"/ttHAnalysis/"+year+"/"+label+"/datacards/"+channel
    sourceoriginal=mom+"/prepareDatacards_"+channel+"_"
    #nStart= [15,10,8]
    #nTarget= [4,5,6,7,8,9,10]
    nStart= [15,20]
    nTarget= [4,5,6,7,8,9,10,11,12,13,18,20,27]
    binsHTT=[4,5,6,7,8,9,10,12,14]
    binsnoHTT=[4,5,6,7,8,9,10,15,16]

if options.do1D==False :
    iterator1=nStart
    iterator2=nTarget
if options.do1D==True :
    iterator1=[1]
    if  "_noHTT" in options.variables : iterator2=binsnoHTT
    if  "_HTT" in options.variables : iterator2=binsHTT

local=workingDir+"/"+options.channel+"_"+label+"/"+options.variables+"/"
sources=[]
print sourceoriginal

import shutil
proc=subprocess.Popen(['mkdir '+options.channel+"_"+label],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()
proc=subprocess.Popen(['mkdir '+options.channel+"_"+label+"/"+options.variables],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()

def run_cmd(command):
  print "executing command = '%s'" % command
  p = subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
  stdout, stderr = p.communicate()
  print stderr
  return stdout


if not options.doLimits and not options.plotLimits :
    execfile("../python/data_manager.py")
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.title(options.variables+' 2D map' )
    # ' 2D map (from'+str(nstart)+' to '+str(ntarget)+' bins)'
    colorsToDo=['r','g','b','m','y','c']
    for nn,nstart in enumerate(iterator1) :
        errOcontTTLast=[]
        errOcontTTPLast=[]
        errOcontSUMLast=[]
        errOcontSUMPLast=[]
        for ntarget in iterator2 :
            ## prepareDatacards_2lss_1tau_sumOS_HTTMEM_from15_to_10.root
            # "mvaOutput_ttbar_noHTT_15bins"
            if options.do1D==False : my_file = sourceoriginal+options.variables+'_from'+str(nstart)+'_to_'+str(ntarget)+'.root'
            if options.do1D==True : my_file = sourceoriginal+'mvaOutput_'+options.variables+'_'+str(ntarget)+'bins.root'
            if os.path.exists(my_file) : print (my_file,"reading ")
            else : print (my_file,"does not exist ")
            if options.do1D==False : namepdf=local+options.variables+'_from'+str(nstart)+'_to_'+str(ntarget)+'_plots.pdf'
            if options.do1D==True : namepdf=local+options.variables+"_"+str(ntarget)+"bins.pdf"
            result=GetRatio(my_file,namepdf)
            errOcontTTLast=errOcontTTLast+[result[0]] if result[0]<1.001 else errOcontSUMLast+[1.0]
            #errOcontTTPLast=errOcontTTPLast+[result[1]] if ratiohSumP<1.001 else errOcontSUMPLast+[1.0]
            errOcontSUMLast=errOcontSUMLast+[result[2]] if result[2]<1.001 else errOcontSUMLast+[1.0]
            #errOcontSUMPLast=errOcontSUMPLast+[result[3]] if ratiohSumP<1.001 else errOcontSUMPLast+[1.0]
        print ("TT",errOcontTTLast)
        print ("sum",errOcontSUMLast)
        plt.plot(iterator2,errOcontTTLast, colorsToDo[nn]+'o-',label=str(nstart)+' star bins')
        plt.plot(iterator2,errOcontSUMLast, colorsToDo[nn]+'x--')
    ax.set_xlabel('nbins')
    plt.axis((min(iterator2),max(iterator2),0,1.0))
    line_up, = plt.plot(iterator2, 'o-', color='k',label="fake-only")
    line_down, = ax.plot(iterator2, 'x--', color='k',label="fake+ttV+EWK")
    legend1 = plt.legend(handles=[line_up, line_down], loc='best')
    ax.set_ylabel('err/cont')
    ax.legend(loc='best', fancybox=False, shadow=False, ncol=2)
    plt.grid(True)
    fig.savefig(local+options.variables+'_2D_errOcont.pdf')



if options.doLimits:
    print ("to run this script your CMSSW_base should be the one that CombineHavester installed")
    folder=options.channel+"_"+label+"/"+options.variables+"/"
    file = open("execute_plots_2D_"+options.channel+"_"+options.variables+".sh","w")
    file.write("#!/bin/bash\n")
    for nn,nstart in enumerate(iterator1) :
        for ntarget in iterator2 :
            if options.do1D==False : name=sourceoriginal+options.variables+'_from'+str(nstart)+'_to_'+str(ntarget)+'.root'
            if options.do1D==True : name=sourceoriginal+'mvaOutput_'+options.variables+'_'+str(ntarget)+'bins.root'
            print ("doing", name)
            if options.do1D==False : shapeVariable=options.variables+'_from'+str(nstart)+'_to_'+str(ntarget)
            if options.do1D==True : shapeVariable=options.variables+'_'+str(ntarget)+'bins'
            datacardFile_output = os.path.join(workingDir, folder, "ttH_%s.root" % shapeVariable)
            run_cmd('%s --input_file=%s --output_file=%s --add_shape_sys=false' % ('WriteDatacards_'+channel, name, datacardFile_output))
            txtFile = datacardFile_output.replace(".root", ".txt")
            logFile = datacardFile_output.replace(".root", ".log")
            run_cmd('combine -M Asymptotic -m %s -t -1 %s &> %s' % (str(125), txtFile, logFile))
            run_cmd('rm higgsCombineTest.Asymptotic.mH125.root')
            rootFile = os.path.join(workingDir, folder, "ttH_%s_shapes.root" % (shapeVariable))
            run_cmd('PostFitShapes -d %s -o %s -m 125 ' % (txtFile, rootFile))
            ##### 2lss_1taumvaOutput_2lss_1tau_ttV_nbin_6
            makeplots='root -l -b -n -q /home/acaan/VHbbNtuples_8_0_x/CMSSW_7_4_7/src/CombineHarvester/ttH_htt/macros/makePostFitPlots.C++(\\"'+shapeVariable+'\\",\\"'+folder+'\\",\\"'+options.channel+'\\",\\"'+local+'\\")'
            file.write(makeplots+ "\n")
    file.close()

#########################################
## make limits
if options.plotLimits :
     execfile("../python/data_manager.py")
     import matplotlib
     matplotlib.use('agg')
     import matplotlib.pyplot as plt
     print "plot limits"
     fig, ax = plt.subplots(figsize=(5, 5))
     #plt.title(options.BINtype+" binning")
     colorsToDo=['r','g','b','m','y','c']
     for nn,nstart in enumerate(iterator1) :
         limits=ReadLimits("none", iterator2, "2D",20,local,nstart,0)
         print (len(iterator2),len(limits[0]))
         plt.plot(iterator2,limits[0], colorsToDo[nn]+'o-',label=str(nstart)+' start bins')
         plt.plot(iterator2,limits[1], colorsToDo[nn]+'-')
         plt.plot(iterator2,limits[3], colorsToDo[nn]+'-')
     ax.legend(loc='best', fancybox=False, shadow=False , ncol=1)
     ax.set_xlabel('nbins')
     ax.set_ylabel('limits')
     if options.do1D==False : plt.text(5.1, 2.4, "2D map -"+" "+options.variables )
     if options.do1D==True : plt.text(5.1, 2.4, "plain filling (TT-BDT) -"+" "+options.variables )
     #plt.text(5.1, 2.53, "CMS"  ,  fontweight='bold' )
     #plt.text(5.7, 2.53, "preliminary" )
     #plt.text(8, 2.53, "35.9/fb (13 TeV)"   )
     plt.axis((min(iterator2),max(iterator2),0.7,6.5))
     if options.do1D==False :  namefig=options.channel+'_'+label+'/'+options.variables+'/'+options.variables+'_fullsim_limits_2Dmap.pdf'
     if options.do1D==True : namefig=options.channel+'_'+label+'/'+options.variables+'/'+options.variables+'_fullsim_limits_plainFilling.pdf'
     fig.savefig(namefig)
     print ("saved",namefig)

     fig, ax = plt.subplots(figsize=(5, 5))
     for nn,nstart in enumerate(iterator1) :
        fakeOnly=[]
        allBKG=[]
        for ntarget in iterator2:
             if options.do1D==False :
                 histSource=sourceoriginal+options.variables+"_from"+str(nstart)+"_to_"+str(ntarget)+".root"
                 namepdf=options.channel+'_'+label+'/'+options.variables+'/'+options.variables+'_fullsim_errOcont_2Dmap.pdf'
             if options.do1D==True :
                 histSource=sourceoriginal+"mvaOutput_"+options.variables+"_"+str(ntarget)+"bins.root"
                 namepdf=options.channel+'_'+label+'/'+options.variables+'/'+options.variables+'_fullsim_errOcont_'+str(ntarget)+"bins.pdf"
             ratios=GetRatio(histSource,namepdf)
             fakeOnly=fakeOnly+[ratios[0]]
             allBKG=allBKG+[ratios[2]]
        print (len(iterator2),len(fakeOnly),len(allBKG))
        plt.plot(iterator2,fakeOnly, colorsToDo[nn]+'o-')
        plt.plot(iterator2,allBKG, colorsToDo[nn]+'x--')
        print ("fakeOnly",fakeOnly)
     line_up, = plt.plot(iterator2, 'o-', color='k',label="fake-only")
     line_down, = ax.plot(iterator2, 'x--', color='k',label="fake+ttV+EWK")
     legend1 = plt.legend(handles=[line_up, line_down], loc='best')
     plt.axis((min(iterator2),max(iterator2),0.0,1.0))
     ax.set_ylabel('err/content last bin')
     ax.legend(loc='best', fancybox=False, shadow=False, ncol=1) #, ncol=3)
     plt.grid(True)
     if options.do1D==False :  namefig=options.channel+'_'+label+'/'+options.variables+'/'+options.variables+'_fullsim_ErrOcont_2Dmap.pdf'
     if options.do1D==True :  namefig=options.channel+'_'+label+'/'+options.variables+'/'+options.variables+'_fullsim_ErrOcont_plainFilling.pdf'
     fig.savefig(namefig)
     print ("saved",namefig)
