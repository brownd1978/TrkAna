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
import SurfaceIds as SID
import MyHist
import h5py

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
        self.HPriNHits = MyHist.MyHist(name="HPriNHits",bins=100,range=[0.5,100.5],label="Primary NActive",title="Fit N Hits",xlabel="N Hits")
        self.HSecNHits = MyHist.MyHist(name="HSecNHits",bins=100,range=[0.5,100.5],label="Secondary NActive",title="Fit N Hits",xlabel="N Hits")

        # Momentum histograms
        nMomBins = 100
        momrange=(90.0,110.0)
        nDeltaMomBins = 200
        deltamomrange=(-0.25,0.25)
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
        for batch,rep in uproot.iterate(Files,filter_name="/trk|trksegs|trkmcsim|gtrksegsmc/i",report=True):
            print(ibatch,end=' ')
            ibatch = ibatch+1
            segs = batch['trksegs'] # track fit samples
            nhits = batch['trk.nactive']  # track N hits
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
            priTQ = trkQual[:,0]
            secTQ = trkQual[:,1]

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
    def PlotMom(self):
        fig, (priMom, secMom, deltaMom) = plt.subplots(1,3,layout='constrained', figsize=(15,5))
        primom = self.HPriMom.plot(priMom)
        secmom = self.HSecMom.plot(secMom)
        delmom = self.HDeltaMom.plot(deltaMom)

    def PlotQuality(self):
        fig, (anhit,afc,atq) = plt.subplots(1,3,layout='constrained', figsize=(15,5))
        prinhit = self.HPriNHits.plot(anhit)
        secnhit = self.HSecNHits.plot(anhit)
        anhit.legend(loc="upper right")
        prifc = self.HPriFitCon.plot(afc)
        secfc = self.HSecFitCon.plot(afc)
        afc.legend(loc="upper right")
        pritq = self.HPriTQ.plot(atq)
        sectq = self.HSecTQ.plot(atq)
        atq.legend(loc="upper right")

