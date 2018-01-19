#!/usr/bin/env python
import os, subprocess, sys
workingDir = os.getcwd()
from array import array
from ROOT import *
from math import sqrt, sin, cos, tan, exp


from optparse import OptionParser
parser = OptionParser()
parser.add_option("--channel ", type="string", dest="channel", help="The ones whose variables implemented now are:\n   - 1l_2tau\n   - 2lss_1tau\n It will create a local folder and store the report*/xml", default="2lss_1tau")
parser.add_option("--variables", type="string", dest="variables", help="  Set of variables to use -- it shall be put by hand in the code", default=1000)
parser.add_option("--doPlots", action="store_true", dest="doPlots", help="If you call this will not do plots with repport", default=False)
parser.add_option("--doLimits", action="store_true", dest="doLimits", help="If you call this will not do plots with repport", default=False)
(options, args) = parser.parse_args()

user="acaan"
channel='2lss_1tau'
label='2018Jan18_VHbb_addMEM'
year="2016"
sourceoriginal="/home/"+user+"/ttHAnalysis/"+year+"/"+label+"/datacards/"+channel+"/prepareDatacards_"+channel+"_sumOS_"
source=workingDir+"/"+options.channel+"_"+label+"/"+options.variables+"/prepareDatacards_"+channel+"_sumOS_"
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

nStart= [15,20]
nTarget= [5,6,7,8,9,10]

if not options.doLimits:
    execfile("../python/data_manager.py")
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.title(options.variables+' 2D map' )
    # ' 2D map (from'+str(nstart)+' to '+str(ntarget)+' bins)'
    colorsToDo=['r','g','b','m','y','c']
    for nn,nstart in enumerate(nStart) :
        errOcontTTLast=[]
        errOcontTTPLast=[]
        errOcontSUMLast=[]
        errOcontSUMPLast=[]
        for ntarget in nTarget :
            ## prepareDatacards_2lss_1tau_sumOS_HTTMEM_from15_to_10.root
            my_file = sourceoriginal+options.variables+'_from'+str(nstart)+'_to_'+str(ntarget)+'.root'
            if os.path.exists(my_file) : print (my_file,"reading ")
            else : print (my_file,"does not exist ")
            result=GetRatio(my_file)
            errOcontTTLast=errOcontTTLast+[result[0]] if result[0]<1.001 else errOcontSUMLast+[1.0]
            #errOcontTTPLast=errOcontTTPLast+[result[1]] if ratiohSumP<1.001 else errOcontSUMPLast+[1.0]
            errOcontSUMLast=errOcontSUMLast+[result[2]] if result[2]<1.001 else errOcontSUMLast+[1.0]
            #errOcontSUMPLast=errOcontSUMPLast+[result[3]] if ratiohSumP<1.001 else errOcontSUMPLast+[1.0]
        print ("TT",errOcontTTLast)
        print ("sum",errOcontSUMLast)
        plt.plot(nTarget,errOcontTTLast, colorsToDo[nn]+'o-',label=str(nstart)+' star bins')
        plt.plot(nTarget,errOcontSUMLast, colorsToDo[nn]+'x--')
    ax.set_xlabel('nbins')
    plt.axis((min(nTarget),max(nTarget),0,1.0))
    line_up, = plt.plot(nTarget, 'o-', color='k',label="fake-only")
    line_down, = ax.plot(nTarget, 'x--', color='k',label="fake+ttV+EWK")
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
    for nn,nstart in enumerate(nStart) :
        for ntarget in nTarget :
            name=sourceoriginal+options.variables+'_from'+str(nstart)+'_to_'+str(ntarget)+'.root'
            print ("doing", name)
            shapeVariable=options.variables+'_from'+str(nstart)+'_to_'+str(ntarget)
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
 if plotLimits :
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
