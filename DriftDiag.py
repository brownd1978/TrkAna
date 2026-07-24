#
# Drift diagnostics
#
#  TH2D* dd = new TH2D("dd","Reco DOCA vs MC true DOCA;MC DOCA (mm);Reco DOCA (mm)",50,-3,3,50,-3,3);
#  TH1D* dres = new TH1D("dres","Drift Resolution;R_{drift}-MC DOCA (mm)",NBINS,-2.5,2.5);
#  TH1D* dresg = new TH1D("dresg","Drift Resolution;R_{drift}-MC DOCA (mm)",NBINS,-2.5,2.5);
#  TH1D* dresb = new TH1D("dresb","Drift Resolution;R_{drift}-MC DOCA (mm)",NBINS,-2.5,2.5);
#  TH1D* dresp = new TH1D("dresp","Drift Resolution Pull;(R_{drift}-MC DOCA)/#sigma_{rdrift}",NBINS,-8.0,8.0);
#  TH1D* drespg = new TH1D("drespg","Drift Resolution Pull;(R_{drift}-MC DOCA)/#sigma_{rdrift}",NBINS,-8.0,8.0);
#  TH1D* drespb = new TH1D("drespb","Drift Resolution Pull;(R_{drift}-MC DOCA)/#sigma_{rdrift}",NBINS,-8.0,8.0);
#  TH1D* udoca = new TH1D("udoca","Unbiased DOCA Resolution;UDOCA-MC DOCA) (mm)",NBINS,-5,5);
#  TH1D* udocap = new TH1D("udocap","Unbiased DOCA Resolution Pull;UDOCA-MC DOCA/UDOCA error",NBINS,-15,15);
import uproot
import awkward as ak
import behaviors
from matplotlib import pyplot as plt
import uproot
import numpy as np
from scipy.optimize import curve_fit
import math
from scipy import special
import MyHist



class DriftDiag(object):
    def __init__(self):
        dresbins = 100
        dresrange = (-2.5,2.5)
        drestitle = "Drift Resolution"
        dresxlabel = "$R_{drift}$ - MC DOCA (mm)"
        self.Hdres = MyHist.MyHist(name="dres",label="All",title=drestitle,xlabel=dresxlabel,bins=dresbins,range=dresrange)
        self.Hstate = MyHist.MyHist(name="state",label="All",title="Hit State",bins=5,range=[-3.5,1.5])
        nhitsrange = [-0.5,99.5]
        nhitsbins = 100
        self.Hnhits = MyHist.MyHist(name="nhits",label="All",title="N Hits",bins=nhitsbins,range=nhitsrange)
        self.Hnactive = MyHist.MyHist(name="nactive",label="Active",title="N Hits",bins=nhitsbins,range=nhitsrange)
        self.Hndrift = MyHist.MyHist(name="ndrift",label="Drift",title="N Hits",bins=nhitsbins,range=nhitsrange)
        self.Hnnull = MyHist.MyHist(name="nnull",label="Null",title="N Hits",bins=nhitsbins,range=nhitsrange)

    def Loop(self,files):
        ibatch = 0
        np.set_printoptions(precision=5,floatmode='fixed')
        print("Processing batch ",end=' ')
        for batch,rep in uproot.iterate(files,filter_name="/evtinfo|trk.trk|trksegs|trkmcsim|trksegsmc|trkqual|trkhits|trkhitsmc/i",report=True):
            print(ibatch,end=' ')
            # should be 1 track/event
            nhits = batch['trk.nhits']  # track N hits
            assert(ak.sum(ak.count_nonzero(nhits,axis=1)!=1) == 0)
            nhits = nhits[:,0]
            nactive = batch['trk.nactive']
            nactive = nactive[:,0]
            self.Hnhits.fill(np.array(nhits))
            self.Hnactive.fill(np.array(nactive))
            ibatch = ibatch+1
            trkhits = batch['trkhits']
            trkhits = trkhits[:,0] # first track only
            trkhitsmc = batch['trkhitsmc']
            trkhitsmc = trkhitsmc[:,0]

            # hit state
            self.Hstate.fill(np.array(ak.flatten(trkhits.state)))
            # active hits
            drift = np.absolute(trkhits.state) == 1
            null = trkhits.state == 0
            drifthits = trkhits[drift]
            nullhits = trkhits[null]
            ndrift = ak.count(drifthits.algo,axis=1)
            nnull = ak.count(nullhits.algo,axis=1)
            self.Hndrift.fill(np.array(ndrift))
            self.Hnnull.fill(np.array(nnull))

            # select mc truth only of hits utilized in reconstruction
            # i.e., ``clip the tail'' of off-track hits
            mask = ak.local_index(trkhitsmc, axis=1) < ak.num(trkhits, axis=1)
            trkhitsmc = trkhitsmc[mask]
            self.Hdres.fill(np.array(ak.flatten(trkhits.rdrift-trkhitsmc.dist)))
        print()

    def Plot(self):
        fig, (anhits,astate) = plt.subplots(1,2,layout='constrained', figsize=(15,5))
        self.Hnhits.plot(anhits)
        self.Hnactive.plot(anhits)
        self.Hndrift.plot(anhits)
        self.Hnnull.plot(anhits)
        anhits.legend(loc="upper right")
        self.Hstate.plot(astate)

        fig, (res,dist) = plt.subplots(1,2,layout='constrained', figsize=(15,5))
        self.Hdres.plot(res)

