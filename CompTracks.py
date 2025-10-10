#
# compare 2 reconstructions of the same track
#
import uproot
import awkward as ak
import behaviors
from matplotlib import pyplot as plt
import uproot
import numpy as np
import math
from scipy import special
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import uniform
import SurfaceIds as SID
import MyHist
import h5py

def fxn_Gauss(x, amp, mean, sigma) :
    return amp*norm.pdf(x,loc=mean,scale=sigma)

class CompTracks(object):
    def __init__(self,sid,pdg):
        # PDG cods of signal and background particles
        self.PDG = pdg
        PDGNames = {-13:"$\\mu^+$",-11:"$e^+$",11:"$e^-$",13:"$\\mu^-$"}
        self.PDGName = PDGNames[self.PDG]
        # setup cuts; these should be overrideable FIXME
        self.MinNHits = 20
        self.MinFitCon = 1.0e-5
        self.MaxDeltaT = 5.0 # nsec
        self.MinTQ = 0.2 # ANN output
        # Surface Ids
        self.SID = sid
        self.CompName = SID.SurfaceName(sid)
        # fit quality histograms
        self.HPriTQ = MyHist.MyHist(name="HPriTQ",bins=100,range=[0.0,1.0],label="Primary TrkQual",title="Track Quality",xlabel="ANN Result")
        self.HSecTQ = MyHist.MyHist(name="HSecTQ",bins=100,range=[0.0,1.0],label="Secondary TrkQual",title="Track Quality",xlabel="ANN Result")
        self.HPriFitCon = MyHist.MyHist(name="HPriFitCon",bins=100,range=[0.0,1.0],label="Primary FitCon",title="Fit Consistency",xlabel="")
        self.HSecFitCon = MyHist.MyHist(name="HSecFitCon",bins=100,range=[0.0,1.0],label="Secondary FitCon",title="Fit Consistency",xlabel="")
        self.HPriNHits = MyHist.MyHist(name="HPriNHits",bins=100,range=[0.5,100.5],label="Primary NHits",title="Fit Count",xlabel="N Hits")
        self.HSecNHits = MyHist.MyHist(name="HSecNHits",bins=100,range=[0.5,100.5],label="Secondary NHits",title="Fit Count",xlabel="N Hits")
        self.HDNHits = MyHist.MyHist(name="HDNHits",bins=21,range=[-10.5,10.5],label="Secondary - Primary NHits",title="Fit $\\Delta$",xlabel="$\\Delta$ N")
        self.HPriNStraws = MyHist.MyHist(name="HPriNStraws",bins=100,range=[0.5,100.5],label="Primary NStraws",title="Fit Count",xlabel="N Straws")
        self.HSecNStraws = MyHist.MyHist(name="HSecNStraws",bins=100,range=[0.5,100.5],label="Secondary NStraws",title="Fit Count",xlabel="N Straws")
        self.HDNStraws = MyHist.MyHist(name="HDNStraws",bins=21,range=[-10.5,10.5],label="Secondary - Primary NStraws",title="Fit $\\Delta$",xlabel="$\\Delta$ N")
        self.HPriRadlen = MyHist.MyHist(name="HPriRadlen",bins=100,range=[0.0,0.04],label="Primary Radlen",title="Fit Radlen",xlabel="$X_{0}$")
        self.HSecRadlen = MyHist.MyHist(name="HSecRadlen",bins=100,range=[0.0,0.04],label="Secondary Radlen",title="Fit Radlen",xlabel="$X_{0}$")

        # Momentum histograms
        nMomBins = 100
        momrange=(90.0,110.0)
        nDeltaMomBins = 200
        deltamomrange=(-0.2,0.2)
        self.HPriMom = MyHist.MyHist(name="PriMom",label="All", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)",title=self.PDGName+" Primary Momentum at "+self.CompName)
        self.HSecMom = MyHist.MyHist(name="SecMom",label="All", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+" Secondary Momentum at "+self.CompName)
        # Momentum comparison histograms
        self.HDeltaMom = MyHist.MyHist(name="DeltaMom",label="All", bins=nDeltaMomBins, range=deltamomrange, xlabel="Secondary - Primary Momentum (MeV)",title=self.PDGName+" $\\Delta$ Momentum at "+self.CompName)


    def Loop(self,files,treename):
        # append tree to files for uproot
        Files = [None]*len(files)
        for i in range(0,len(files)):
            Files[i] = files[i]+":"+treename
        ibatch = 0
        print("Processing batch ",end=' ')
        for batch,rep in uproot.iterate(Files,filter_name="/evtinfo|trk|trksegs|trkmcsim|trksegsmc/i",report=True):
            print(ibatch,end=' ')
            ibatch = ibatch+1
            segs = batch['trksegs'] # track fit samples
            nhits = batch['trk.nactive']  # track N hits
            nstraws = batch['trk.nmatactive']  # track N hits
            radlen = batch['trk.radlen']  # track N hits
            fitcon = batch['trk.fitcon']  # track fit consistency
            trkQual = batch['trkqual.result']  # track fit quality
            fitpdg = batch['trk.pdg']  # track fit consistency
            # compress out unneeded dimensions
            priSegs = segs[:,0] # primary track fits
            secSegs = segs[:,1] # secondary track fits
            priFitPDG = fitpdg[:,0]
            secFitPDG = fitpdg[:,1]
            priFitCon = fitcon[:,0]
            secFitCon = fitcon[:,1]
            priNhits = nhits[:,0]
            secNhits = nhits[:,1]
            priNstraws = nstraws[:,0]
            secNstraws = nstraws[:,1]
            priRL = radlen[:,0]
            secRL = radlen[:,1]
            priTQ = trkQual[:,0]
            secTQ = trkQual[:,1]
            runnum = batch['run']
            subrun = batch['subrun']
            event = batch['event']

            # basic consistency test
            assert((len(priSegs) == len(secSegs)) & (len(priSegs) == len(priNhits)) & (len(priNhits) == len(secNhits)) & (len(priTQ) == len(secTQ)) & (len(priTQ) == len(priSegs)) )
            # select fits that match PDG code
            priSigPart = (priFitPDG == self.PDG)
            secSigPart = (secFitPDG == self.PDG)
            sigPartFit = priSigPart & secSigPart
            # find momentum to test
            primom = priSegs[(priSegs.sid == self.SID) & (priSegs.mom.Z() > 0.0)].mom.magnitude()
            secmom = secSegs[(secSegs.sid == self.SID) & (secSegs.mom.Z() > 0.0)].mom.magnitude()
            # check pri and down match
            trkmatch = ak.num(primom) == ak.num(secmom)
            self.HPriFitCon.fill(np.array(priFitCon[trkmatch]))
            self.HSecFitCon.fill(np.array(secFitCon[trkmatch]))
            self.HPriNHits.fill(np.array(priNhits[trkmatch]))
            self.HSecNHits.fill(np.array(secNhits[trkmatch]))
            self.HDNHits.fill(np.array(secNhits[trkmatch]-priNhits[trkmatch]))
            self.HPriNStraws.fill(np.array(priNstraws[trkmatch]))
            self.HSecNStraws.fill(np.array(secNstraws[trkmatch]))
            self.HDNStraws.fill(np.array(secNstraws[trkmatch]-priNstraws[trkmatch]))
            self.HPriRadlen.fill(np.array(priRL[trkmatch]))
            self.HSecRadlen.fill(np.array(secRL[trkmatch]))
            self.HPriTQ.fill(np.array(priTQ[trkmatch]))
            self.HSecTQ.fill(np.array(secTQ[trkmatch]))
            # select based on fit quality
            priGoodFit = (priNhits >= self.MinNHits) & (priFitCon > self.MinFitCon) & (priTQ > self.MinTQ)
            secGoodFit = (secNhits >= self.MinNHits) & (secFitCon > self.MinFitCon) & (secTQ > self.MinTQ)
            goodReco = trkmatch & priGoodFit & secGoodFit & sigPartFit
            priEntTime = priSegs[(priSegs.sid==self.SID) & (priSegs.mom.z() > 0.0) & goodReco ].time
            secEntTime = secSegs[(secSegs.sid==self.SID) & (secSegs.mom.z() > 0.0) & goodReco ].time
            deltaEntTime = secEntTime-priEntTime
            goodDeltaT = abs(deltaEntTime) < self.MaxDeltaT
            # total momentum of primary and secondary fits at the comparison point
            primom = priSegs[(priSegs.sid == self.SID) & (priSegs.mom.Z() > 0.0)].mom.magnitude()
            secmom = secSegs[(secSegs.sid == self.SID) & (secSegs.mom.Z() > 0.0)].mom.magnitude()
            # flatten
            priMom = np.array(ak.flatten(primom[goodReco],axis=1))
            secMom = np.array(ak.flatten(secmom[goodReco],axis=1))
            if len(priMom) != len(secMom):
                print()
                print("Primary and Secondary fits don't match!",len(priMom),len(secMom))
                continue
            # good fits
            goodFit = goodReco & goodDeltaT & trkmatch
            goodFit = ak.ravel(goodFit)
            self.HPriMom.fill(priMom[goodFit])
            self.HSecMom.fill(secMom[goodFit])
            deltaMom = secMom - priMom
            self.HDeltaMom.fill(deltaMom[goodFit])
#            #
            pnh = np.array(priNhits[trkmatch])
            snh = np.array(secNhits[trkmatch])
            neh = np.not_equal(pnh,snh)
            dh_runnum = runnum[neh]
            dh_subrun = subrun[neh]
            dh_event = event[neh]
            pnh_diff = pnh[neh]
            snh_diff = snh[neh]
            if len(dh_runnum) > 0:
                print("unmatched nhits")
                for ievt in range(len(dh_runnum)):
                    print(dh_runnum[ievt],":", dh_subrun[ievt],":", dh_event[ievt],",",pnh_diff[ievt],",",snh_diff[ievt])
            #
            pns = np.array(priNstraws[trkmatch])
            sns = np.array(secNstraws[trkmatch])
            nes = np.not_equal(pns,sns)
            ds_runnum = runnum[nes]
            ds_subrun = subrun[nes]
            ds_event = event[nes]
            pns_diff = pns[nes]
            sns_diff = sns[nes]
            if len(ds_runnum) > 0:
                print("unmatched nstraws")
                for ievt in range(len(ds_runnum)):
                    print(ds_runnum[ievt],":", ds_subrun[ievt],":", ds_event[ievt],",",pns_diff[ievt],",",sns_diff[ievt])

    def PlotMom(self):
        fig, (priMom, secMom, deltaMom) = plt.subplots(1,3,layout='constrained', figsize=(15,5))
        primom = self.HPriMom.plot(priMom)
        secmom = self.HSecMom.plot(secMom)
        delmom = self.HDeltaMom.plot(deltaMom)

        # fit momentum difference (core)
        maxval = self.HDeltaMom.maxVal()
        #print("maxval",maxval)
        binrange = self.HDeltaMom.binRange(0.1*maxval)
        binsize = self.HDeltaMom.binWidth()
        amp_0 = np.sum(self.HDeltaMom.data)*binsize # initial amplitude
        p0 = np.array([amp_0,0.0,0.01])
        binmid = np.zeros(len(self.HDeltaMom.data))
        binerr = np.zeros(len(self.HDeltaMom.data))
        core = np.zeros(len(self.HDeltaMom.data))
        #print(binrange)
        for ibin in range(len(self.HDeltaMom.data)):
            binmid[ibin] = 0.5*(self.HDeltaMom.edges[ibin] + self.HDeltaMom.edges[ibin+1])
            binerr[ibin] = max(1.0,math.sqrt(self.HDeltaMom.data[ibin]))
            if ((ibin >= binrange[0]) & (ibin <= binrange[1])):
                core[ibin] = self.HDeltaMom.data[ibin]
        popt, pcov = curve_fit(fxn_Gauss, binmid, core, p0, sigma=binerr)
        # compute full standard deviation
        mean = np.average(binmid,weights=self.HDeltaMom.data)
        var = np.average(np.square(binmid-mean),weights=self.HDeltaMom.data)
        stdev = math.sqrt(var)
        print(mean,var,stdev)
        deltaMom.set_yscale("log")
        deltaMom.set_ylim(ymin=0.001*maxval,ymax=2.0*maxval)
        deltaMom.plot(binmid, fxn_Gauss(binmid, *popt), 'r-',label="Fit")
        deltaMom.text(0.6, 0.9, f"Core $\\mu$ = {popt[1]:.3f} $\\pm$ {np.sqrt(pcov[1][1]):.3f}",transform=deltaMom.transAxes)
        deltaMom.text(0.6, 0.8, f"Core $\\sigma$ = {popt[2]:.3f} $\\pm$ {np.sqrt(pcov[2][2]):.3f}",transform=deltaMom.transAxes)
        deltaMom.text(0.6, 0.7, f"Full Mean = {mean:.3f}",transform=deltaMom.transAxes)
        deltaMom.text(0.6, 0.6, f"Full RMS = {stdev:.3f}",transform=deltaMom.transAxes)

    def PlotCount(self):
        fig, (anhit,anstraw,adelta) = plt.subplots(1,3,layout='constrained', figsize=(15,5))
        prinhit = self.HPriNHits.plot(anhit)
        secnhit = self.HSecNHits.plot(anhit)
        anhit.legend(loc="upper right")

        prinstraw = self.HPriNStraws.plot(anstraw)
        secnstraw = self.HSecNStraws.plot(anstraw)
        anstraw.legend(loc="upper right")

        dnhit = self.HDNHits.plot(adelta)
        dnstraw = self.HDNStraws.plot(adelta)
        adelta.legend(loc="upper right")

    def PlotQuality(self):
        fig, (aradl,afc,atq) = plt.subplots(1,3,layout='constrained', figsize=(15,5))
        prifc = self.HPriFitCon.plot(aradl)
        secfc = self.HSecFitCon.plot(aradl)
        aradl.legend(loc="upper right")
        prirl = self.HPriRadlen.plot(afc)
        secrl = self.HSecRadlen.plot(afc)
        afc.legend(loc="upper right")
        pritq = self.HPriTQ.plot(atq)
        sectq = self.HSecTQ.plot(atq)
        atq.legend(loc="upper right")

