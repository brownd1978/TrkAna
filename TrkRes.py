#
# class to measure the resolution of tracks, given a reduced ntuple with fixed-index tracks
#
import uproot
import awkward as ak
import behaviors
from matplotlib import pyplot as plt
import uproot
import numpy as np
from scipy.optimize import curve_fit
import math
from scipy import special
import SurfaceIds as SID
import TrkDirection as TD
import PDGCodes as PDG
import HistUtil
import h5py
from scipy.stats import crystalball

def fxn_CrystalBall(x, amp, beta, m, loc, scale):
    pars = np.array([beta, m, loc, scale])
    return amp*crystalball.pdf(x,*pars)

def TargetFoil(tgtz):
    tgtz0 = -4300. # target center in detector coordinates
    tgtdz = 22.222222 # target spacing
    ntgt = 37 # number of target foils
    tgt0z = tgtz0 - 0.5*(ntgt-1)*tgtdz
    tgtnum = (tgtz-tgt0z)/tgtdz
    itgt = int(round(tgtnum))
    return itgt


class TrkRes(object):
    def __init__(self,momrange,costrange,minNHits,minFitCon,minTrkQual,index=0,trkdir=TD.Downstream(),pdgcode=PDG.EMinus(),testcharge=False):
        self.MomRange = momrange
        self.CosTRange = costrange
        self.minNHits = minNHits
        self.minFitCon = minFitCon
        self.minTrkQual = minTrkQual
        self.index = index
        self.tdir = trkdir
        self.pdg = pdgcode
        self.testc = testcharge

        nDeltaMomBins = 200
        nMomBins = 200
        momresorange=(-2.5,2.5)
        momresprange=(-10,5)
        momtitle="Momentum at "
        momxlabel="Momentum (MeV)"

        self.TrkLoc = [None]*3
        self.HOriginMom = HistUtil.new_hist(name="OriginMom",label="MC Origin",bins=nMomBins, range=self.MomRange,title="Momentum at Origin",xlabel="Momentum (MeV)")

        self.HTrkFitMom = [None]*3
        self.HTrkMCMom = [None]*3
        self.HTrkRespMom = [None]*3
        self.HTrkResoMom = [None]*3
        self.HTrkRefRespMom = [None]*3
        self.HTrkNotRefRespMom = [None]*3
        self.TrackerSIDs = [SID.TT_Front(), SID.TT_Mid(), SID.TT_Back()]
        momxlabel = "Momentum (MeV)"
        momresotitle = "Momentum Resolution"
        momresotitle = "Momentum Resolution"
        momresptitle = "Momentum Response"
        dmomxlabel = "$\\Delta$ Momentum (MeV)"
        # momentum in tracker
        for isid in range(len(self.TrackerSIDs)):
            loc = "@"+SID.SurfaceName(self.TrackerSIDs[isid])
            self.HTrkFitMom[isid] = HistUtil.new_hist(name=loc+"Mom",label="Fit",bins=nMomBins, range=self.MomRange,title="Momentum"+loc,xlabel=momxlabel)
            self.HTrkMCMom[isid] = HistUtil.new_hist(name=loc+"Mom",label="MC",bins=nMomBins, range=self.MomRange,title="Momentum"+loc,xlabel=momxlabel)
            self.HTrkResoMom[isid] = HistUtil.new_hist(name=loc+"Resolution",label="",bins=nDeltaMomBins, range=momresorange,title=momresotitle+loc,xlabel=dmomxlabel)
            self.HTrkRespMom[isid] = HistUtil.new_hist(name=loc+"Response",label="All",bins=nDeltaMomBins, range=momresprange,title=momresptitle+loc,xlabel=dmomxlabel)
            self.HTrkRefRespMom[isid] = HistUtil.new_hist(name=loc+"Response",label="NTSDA == 0",bins=nDeltaMomBins, range=momresprange,title=momresptitle+loc,xlabel=dmomxlabel)
            self.HTrkNotRefRespMom[isid] = HistUtil.new_hist(name=loc+"Response",label="NTSDA > 0",bins=nDeltaMomBins, range=momresprange,title=momresptitle+loc,xlabel=dmomxlabel)
        # passive material
        nNMatBins = 15
        NMatRange = [-0.5,14.5]
        NMatxlabel = "N Intersections"
        NMattitle = "KKFit Intersections"
        NMattitleMC = "MC Intersections"
        self.HNSTF = HistUtil.new_hist(bins=nNMatBins,range=NMatRange,name="NInter",label="ST Foil",xlabel=NMatxlabel,title=NMattitle)
        self.HNSTC = HistUtil.new_hist(bins=nNMatBins,range=NMatRange,name="NInter",label="ST Cyl.",xlabel=NMatxlabel,title=NMattitle)
        self.HNIPA = HistUtil.new_hist(bins=nNMatBins,range=NMatRange,name="NInter",label="IPA",xlabel=NMatxlabel,title=NMattitle)
        self.HNTSDA = HistUtil.new_hist(bins=nNMatBins,range=NMatRange,name="NInter",label="TSDA",xlabel=NMatxlabel,title=NMattitle)
        self.HNOPA = HistUtil.new_hist(bins=nNMatBins,range=NMatRange,name="NInter",label="OPA",xlabel=NMatxlabel,title=NMattitle)
        self.HNSTFMC = HistUtil.new_hist(bins=nNMatBins,range=NMatRange,name="NInterMC",label="ST",xlabel=NMatxlabel,title=NMattitleMC)
        self.HNIPAMC = HistUtil.new_hist(bins=nNMatBins,range=NMatRange,name="NInterMC",label="IPA",xlabel=NMatxlabel,title=NMattitleMC)
        # momentum change
        nDMomBins = 50
        dMomRange = [-3.0,0.0]
        dMomxlabel = "$\\Delta$ E (MeV)"
        dMomtitle = "KKFit Energy Loss"
        dMomtitleMC = "MC Energy Loss"
        self.HSTDMom = HistUtil.new_hist(bins=nDMomBins,range=dMomRange,name="DMom",label="ST",xlabel=dMomxlabel,title=dMomtitle)
        self.HIPADMom = HistUtil.new_hist(bins=nDMomBins,range=dMomRange,name="DMom",label="IPA",xlabel=dMomxlabel,title=dMomtitle)
        self.HAllDMom = HistUtil.new_hist(bins=nDMomBins,range=dMomRange,name="DMom",label="All",xlabel=dMomxlabel,title=dMomtitle)
        self.HSTDMomMC = HistUtil.new_hist(bins=nDMomBins,range=dMomRange,name="DMomMC",label="ST",xlabel=dMomxlabel,title=dMomtitleMC)
        self.HIPADMomMC = HistUtil.new_hist(bins=nDMomBins,range=dMomRange,name="DMomMC",label="IPA",xlabel=dMomxlabel,title=dMomtitleMC)
        self.HAllDMomMC = HistUtil.new_hist(bins=nDMomBins,range=dMomRange,name="DMomMC",label="All",xlabel=dMomxlabel,title=dMomtitleMC)

        # target intersections
        # momentum at target intersections

        tgtmomresptitle = "Target Momentum Response"
        self.HTgtAvgResp = HistUtil.new_hist(name="AvgTgtResponse",label="Average",bins=nDeltaMomBins, range=momresprange,title=tgtmomresptitle,xlabel=dmomxlabel)
        self.HTgtAvgRespRef = HistUtil.new_hist(name="AvgTgtResponseRef",label="Average (NTSDA == 0)",bins=nDeltaMomBins, range=momresprange,title=tgtmomresptitle,xlabel=dmomxlabel)
        self.HTgtAvgRespNotRef = HistUtil.new_hist(name="AvgTgtResponseNotRef",label="Average (NTSDA > 0)",bins=nDeltaMomBins, range=momresprange,title=tgtmomresptitle,xlabel=dmomxlabel)

        self.HTgtLatestResp = HistUtil.new_hist(name="LatestTgtResponse",label="Latest",bins=nDeltaMomBins, range=momresprange,title=tgtmomresptitle,xlabel=dmomxlabel)
        self.HTgtLatestRespRef = HistUtil.new_hist(name="LatestTgtResponseRef",label="Latest (NTSDA == 0)",bins=nDeltaMomBins, range=momresprange,title=tgtmomresptitle,xlabel=dmomxlabel)
        self.HTgtLatestRespNotRef = HistUtil.new_hist(name="LatestTgtResponseNotRef",label="Latest (NTSDA > 0)",bins=nDeltaMomBins, range=momresprange,title=tgtmomresptitle,xlabel=dmomxlabel)

        rhorange = [20,80]
        rhotitle ="Target Rho"
        rhoxlabel = "Rho (mm)"
        rhonbins=50
        self.HTgtRho = HistUtil.new_hist(name="TgtRho",bins=rhonbins,range=rhorange,label="Fit",title=rhotitle,xlabel=rhoxlabel)
        self.HTgtRhoRef = HistUtil.new_hist(name="TgtRho",bins=rhonbins,range=rhorange,label="Fit (NTSDA == 0)",title=rhotitle,xlabel=rhoxlabel)
        self.HTgtRhoNotRef = HistUtil.new_hist(name="TgtRho",bins=rhonbins,range=rhorange,label="Fit (NTSDA > 0)",title=rhotitle,xlabel=rhoxlabel)
        self.HTgtRhoMC = HistUtil.new_hist(name="TgtRho",bins=rhonbins,range=rhorange,label="MC",title=rhotitle,xlabel=rhoxlabel)
        self.HOriginRho = HistUtil.new_hist(name="TgtRho",bins=rhonbins,range=rhorange,label="MC Origin",title=rhotitle,xlabel=rhoxlabel)
        foilrange = [-0.5,36.5]
        foiltitle ="Target Foil"
        foilxlabel="Foil #"
        foilnbins=37
        self.HTgtFoil = HistUtil.new_hist(name="TgtFoil",bins=foilnbins,range=foilrange,label="Fit",title=foiltitle,xlabel=foilxlabel)
        self.HTgtFoilRef = HistUtil.new_hist(name="TgtFoil",bins=foilnbins,range=foilrange,label="Fit (NTSDA == 0)",title=foiltitle,xlabel=foilxlabel)
        self.HTgtFoilNotRef = HistUtil.new_hist(name="TgtFoil",bins=foilnbins,range=foilrange,label="Fit (NTSDA > 0)",title=foiltitle,xlabel=foilxlabel)
        self.HTgtFoilMC = HistUtil.new_hist(name="TgtFoil",bins=foilnbins,range=foilrange,label="MC",title=foiltitle,xlabel=foilxlabel)
        self.HOriginFoil = HistUtil.new_hist(name="TgtFoil",bins=foilnbins,range=foilrange,label="MC Origin",title=foiltitle,xlabel=foilxlabel)
        costrange = [-0.8,0.8]
        costtitle ="Target Cos($\\Theta$)"
        costxlabel="Cos($\\Theta$)"
        costnbins=50
        self.HTgtCosT = HistUtil.new_hist(name="TgtCosT",bins=costnbins,range=costrange,label="Fit",title=costtitle,xlabel=costxlabel)
        self.HTgtCosTRef = HistUtil.new_hist(name="TgtCosT",bins=costnbins,range=costrange,label="Fit (NTSDA == 0)",title=costtitle,xlabel=costxlabel)
        self.HTgtCosTNotRef = HistUtil.new_hist(name="TgtCosT",bins=costnbins,range=costrange,label="Fit (NTSDA > 0)",title=costtitle,xlabel=costxlabel)
        self.HTgtCosTMC = HistUtil.new_hist(name="TgtCosT",bins=costnbins,range=costrange,label="MC",title=costtitle,xlabel=costxlabel)
        self.HOriginCosT = HistUtil.new_hist(name="TgtCosT",bins=costnbins,range=costrange,label="MC Origin",title=costtitle,xlabel=costxlabel)
        # fit quality
        self.HTrkQual = HistUtil.new_hist(name="TrkQual",bins=100,range=[0.0,1.0],label="TrkQual",title="Track Quality",xlabel="ANN Result")
        self.HFitCon = HistUtil.new_hist(name="FitCon",bins=100,range=[0.0,1.0],label="FitCon",title="Fit Consistency",xlabel="")
        self.HNHits = HistUtil.new_hist(name="NHits",bins=100,range=[0.5,100.5],label="NActive",title="Fit N Hits",xlabel="N Hits")

        # legacy variables
#        for isid in range(len(self.TrackerSIDs)):
#            loc = "@"+SID.SurfaceName(self.TrackerSIDs[isid])

        TDrange = [-1.0,2.0]
        self.HTDmomall = HistUtil.new_hist(name="TD",bins=50,range=TDrange,label="Pz/Pt (all)",title="TanDip",xlabel="Tan($\\lambda$)")
        self.HTDLHall = HistUtil.new_hist(name="TD",bins=50,range=TDrange,label="$\\Lambda/R$ (all)",title="TanDip",xlabel="Tan($\\lambda$)")
        self.HTDparall = HistUtil.new_hist(name="TD",bins=50,range=TDrange,label="tanDip (all)",title="TanDip",xlabel="Tan($\\lambda$)")
        self.HTDmom = HistUtil.new_hist(name="TD",bins=50,range=TDrange,label="Pz/Pt@TT_Front",title="TanDip",xlabel="Tan($\\lambda$)")
        self.HTDLH = HistUtil.new_hist(name="TD",bins=50,range=TDrange,label="$\\Lambda/R$@TT_Front",title="TanDip",xlabel="Tan($\\lambda$)")
        self.HTDpar = HistUtil.new_hist(name="TD",bins=50,range=TDrange,label="tanDip@TT_Front",title="TanDip",xlabel="Tan($\\lambda$)")
        d0range = [-10,400]
        self.Hd0 = HistUtil.new_hist(name="d0",bins=50,range=d0range,label="No Cut",title="d0@TT_Front",xlabel="d$_{0}$ (mm)")
        self.Hd0nstfe0 = HistUtil.new_hist(name="d0",bins=50,range=d0range,label="N$_{ST Foil}$==0",title="d0@TT_Front",xlabel="d$_{0}$ (mm)")
        self.Hd0nstfg0 = HistUtil.new_hist(name="d0",bins=50,range=d0range,label="N$_{ST Foil}$>0",title="d0@TT_Front",xlabel="d$_{0}$ (mm)")
        self.Hd0nstce0 = HistUtil.new_hist(name="d0",bins=50,range=d0range,label="N$_{ST Cyl.}$==0",title="d0@TT_Front",xlabel="d$_{0}$ (mm)")
        self.Hd0nstcg0 = HistUtil.new_hist(name="d0",bins=50,range=d0range,label="N$_{ST Cyl.}$>0",title="d0@TT_Front",xlabel="d$_{0}$ (mm)")
        rmaxrange = [425,725]
        self.Hrmax = HistUtil.new_hist(name="rmax",bins=50,range=rmaxrange,label="No Cut",title="R$_{max}$@TT_Front",xlabel="R$_{max}$ (mm)")
        self.Hrmaxnopae0 = HistUtil.new_hist(name="rmax",bins=50,range=rmaxrange,label="N$_{OPA}$==0",title="R$_{max}$@TT_Front",xlabel="R$_{max}$ (mm)")
        self.Hrmaxnopag0 = HistUtil.new_hist(name="rmax",bins=50,range=rmaxrange,label="N$_{OPA}$>0",title="R$_{max}$@TT_Front",xlabel="R$_{max}$ (mm)")
        TDrange = [0.0,2.0]
        self.HTD = HistUtil.new_hist(name="TD",bins=50,range=TDrange,label="No Cut",title="TanDip",xlabel="Tan($\\lambda$)")
        self.HTDcc = HistUtil.new_hist(name="TD",bins=50,range=TDrange,label="Cutset C",title="TanDip",xlabel="Tan($\\lambda$)")
        self.HTDnf = HistUtil.new_hist(name="TD",bins=50,range=TDrange,label="N$_{ST Foil}$>0 & N$_{OPA}$==0",title="TanDip",xlabel="Tan($\\lambda$)")
        self.HTDnc = HistUtil.new_hist(name="TD",bins=50,range=TDrange,label="N$_{ST Cyl.}$>0 & N$_{OPA}$==0",title="TanDip",xlabel="Tan($\\lambda$)")
        momrange=[85,125]
        self.HMom = HistUtil.new_hist(name="Mom",bins=50,range=momrange,label="No Cut",title="Momentum@TT_Front",xlabel="Momentum (MeV)")
        self.HMomcc = HistUtil.new_hist(name="Mom",bins=50,range=momrange,label="Cutset C",title="Momentum@TT_Front",xlabel="Momentum (MeV)")
        self.HMomnf = HistUtil.new_hist(name="Mom",bins=50,range=momrange,label="N$_{ST Foil}$>0 & N$_{OPA}$==0",title="Momentum@TT_Front",xlabel="Momentum (MeV)")
        self.HMomnc = HistUtil.new_hist(name="Mom",bins=50,range=momrange,label="N$_{ST Cyl.}$>0 & N$_{OPA}$==0",title="Momentum@TT_Front",xlabel="Momentum (MeV)")
        costrange=[0.0,1.0]
        self.HCosT = HistUtil.new_hist(name="CosT",bins=50,range=costrange,label="No Cut",title="Cos($\\Theta$)@TT_Front",xlabel="P$_{z}$/P")
        self.HCosTcc = HistUtil.new_hist(name="CosT",bins=50,range=costrange,label="Cutset C",title="Cos($\\Theta$)@TT_Front",xlabel="P$_{z}$/P")
        self.HCosTnf = HistUtil.new_hist(name="CosT",bins=50,range=costrange,label="N$_{ST Foil}$>0 & N$_{OPA}$==0",title="Cos($\\Theta$)@TT_Front",xlabel="P$_{z}$/P")
        self.HCosTnc = HistUtil.new_hist(name="CosT",bins=50,range=costrange,label="N$_{ST Cyl.}$>0 & N$_{OPA}$==0",title="Cos($\\Theta$)@TT_Front",xlabel="P$_{z}$/P")

    def Loop(self,files):
        ibatch = 0
        np.set_printoptions(precision=5,floatmode='fixed')
        print("Processing batch ",end=' ')
        for batch,rep in uproot.iterate(files,filter_name="/evtinfo|trk.trk|trksegs|trkmcsim|trksegsmc|trkqual|trksegpars_lh/i",report=True):
            print(ibatch,end=' ')
            ibatch = ibatch+1
            runnum = batch['run']
            subrun = batch['subrun']
            event = batch['event']
            segs = batch['trksegs'] # track fit samples
            lhpars = batch['trksegpars_lh'] # track fit samples
            nhits = batch['trk.nactive']  # track N hits
            fitcon = batch['trk.fitcon']  # track fit consistency
            fpdg = batch['trk.pdg'] # track fit PDG
            trkQual = batch['trkqual.result']  # track fit quality
            trkMC = batch['trkmcsim']  # MC genealogy of particles
            segsMC = batch['trksegsmc'] # SurfaceStep infor for true primary particle
            # should be 1 track/event
#            assert(ak.sum(ak.count_nonzero(nhits,axis=1)!=1) == 0)
            Segs = segs[:,self.index]
            lhpars = lhpars[:,self.index]
#            print("Segs len",len(Segs),"lhpars len",len(lhpars))
            FitCon = fitcon[:,self.index]
            NHits = nhits[:,self.index]
            FPDG = fpdg[:,self.index]

            TrkQual = trkQual[:,self.index]
            assert(len(Segs)==len(NHits))

            # define good MC selection first, to allow downstfream comparisons
            SegsMC = segsMC[:,self.index] # segments (of 1st MC match) of indexed track
            TrkMC = trkMC[:,self.index,0] # primary MC match of indexed track
            # basic consistency test
            assert((len(runnum) == len( Segs)) & (len(Segs) == len(SegsMC)) & (len(Segs) == len(TrkMC)) & (len(NHits) == len(Segs)))
            goodMC = PDG.Match(self.pdg,TrkMC.pdg,self.testc)
            OMom = TrkMC.mom.magnitude()
            goodMC = goodMC & (OMom>self.MomRange[0]) & (OMom < self.MomRange[1])
            OMom = OMom[goodMC]
            self.HOriginMom.fill(np.array(OMom))
            self.HOriginRho.fill(np.array(TrkMC[goodMC].pos.rho()))
            self.HOriginCosT.fill(np.array(TrkMC[goodMC].mom.cosTheta()))
            self.HOriginFoil.fill(np.array(list(map(TargetFoil,TrkMC[goodMC].pos.z()))))

            goodFit = PDG.Match(self.pdg,FPDG,self.testc)
            # truncate accordingly
            SegsMC = SegsMC[goodMC & goodFit]
            Segs = Segs[goodMC & goodFit]
            lhpars = lhpars[goodMC & goodFit]
            NHits = NHits[goodMC & goodFit]
            FitCon = FitCon[goodMC & goodFit]
            TrkQual = TrkQual[goodMC & goodFit]
            midsegs = Segs[(Segs.sid == SID.TT_Mid()) & TD.Match(self.tdir,Segs.mom.z()) ]

# not all (cosmic) tracks go through TT_Mid
            goodFit = ak.num(midsegs) > 0

            self.HTrkQual.fill(np.array(TrkQual[goodFit]))
            self.HFitCon.fill(np.array(FitCon[goodFit]))
            self.HNHits.fill(np.array(NHits[goodFit]))

            goodFit = (NHits >= self.minNHits) & (FitCon > self.minFitCon) & (TrkQual > self.minTrkQual) & ak.num(midsegs) > 0

            TSDASeg = Segs[Segs.sid == SID.TSDA() ]
            noTSDA = ak.num(TSDASeg)==0

            # sample the fits at the specified surface
            for isid in range(len(self.TrackerSIDs)) :
                sid = self.TrackerSIDs[isid]
                segs = Segs[(Segs.sid == sid) & TD.Match(self.tdir,Segs.mom.z()) ]
                assert(len(segs) == len(Segs))
                mom = segs.mom.magnitude()
                mom = mom[(mom > self.MomRange[0]) & (mom < self.MomRange[1])]
                hasmom = ak.count_nonzero(mom,axis=1)==1
                segsMC = SegsMC[(SegsMC.sid == sid) & TD.Match(self.tdir,SegsMC.mom.z()) ]
                momMC = segsMC.mom.magnitude()
                hasMC = ak.count_nonzero(momMC,axis=1)==1
                good = hasMC & goodFit & hasmom
                reflectable = good & noTSDA
                notreflectable = good & np.logical_not(noTSDA)
                goodmom = mom[good]
                goodmom = ak.flatten(goodmom,axis=1)
                refmom = mom[reflectable]
                refmom = ak.flatten(refmom,axis=1)
                notrefmom = mom[notreflectable]
                notrefmom = ak.flatten(notrefmom,axis=1)
                goodmomMC = momMC[good]
                goodmomMC = ak.flatten(goodmomMC,axis=1)
                assert(len(goodmom) == len(goodmomMC) )
                self.HTrkFitMom[isid].fill(np.array(goodmom))
                self.HTrkMCMom[isid].fill(np.array(goodmomMC))
                momreso = goodmom - goodmomMC
                self.HTrkResoMom[isid].fill(np.array(momreso))
                momresp = goodmom - OMom[good]
                self.HTrkRespMom[isid].fill(np.array(momresp))
                momrefresp = refmom - OMom[reflectable]
                momnotrefresp = notrefmom - OMom[notreflectable]
                self.HTrkRefRespMom[isid].fill(np.array(momrefresp))
                self.HTrkNotRefRespMom[isid].fill(np.array(momnotrefresp))

            # count IPA and target intersections
            gSegs = Segs[goodFit]
            glhpars = lhpars[goodFit]
            gstf = gSegs.sid==SID.ST_Foils()
            gstc = np.logical_or((gSegs.sid==SID.ST_Outer()),(gSegs.sid==SID.ST_Front()))
            nstf = ak.count_nonzero(gstf,axis=1)
            nstc = ak.count_nonzero(gstc,axis=1)

            self.HNSTF.fill(np.array(nstf))
            self.HNSTC.fill(np.array(nstc))
            self.HNIPA.fill(np.array(ak.count_nonzero(gSegs.sid==SID.IPA(),axis=1)))
            self.HNTSDA.fill(np.array(ak.count_nonzero(gSegs.sid==SID.TSDA(),axis=1)))
            self.HNOPA.fill(np.array(ak.count_nonzero(gSegs.sid==SID.OPA(),axis=1)))
            foilsegs = Segs.sid==SID.ST_Foils()
            ipasegs = Segs.sid==SID.IPA()
            stdmom = ak.sum(Segs[foilsegs].dmom,axis=1)
            ipadmom = ak.sum(Segs[ipasegs].dmom,axis=1)
            stdmom = stdmom[goodFit]
            ipadmom = ipadmom[goodFit]
            self.HSTDMom.fill(np.array(stdmom))
            self.HIPADMom.fill(np.array(ipadmom))
            self.HAllDMom.fill(np.array(stdmom + ipadmom))
            # Also for MC
            self.HNSTFMC.fill(np.array(ak.count_nonzero(SegsMC[goodFit].sid==SID.ST_Foils(),axis=1)))
            self.HNIPAMC.fill(np.array(ak.count_nonzero(SegsMC[goodFit].sid==SID.IPA(),axis=1)))
            foilsegsMC = SegsMC.sid==SID.ST_Foils()
            ipasegsMC = SegsMC.sid==SID.IPA()
            stdmomMC = ak.sum(-SegsMC[foilsegsMC].edep,axis=1)
            ipadmomMC = ak.sum(-SegsMC[ipasegsMC].edep,axis=1)
            stdmomMC = stdmomMC[goodFit]
            ipadmomMC = ipadmomMC[goodFit]
            self.HSTDMomMC.fill(np.array(stdmomMC))
            self.HIPADMomMC.fill(np.array(ipadmomMC))
            self.HAllDMomMC.fill(np.array(stdmomMC + ipadmomMC))

            #foil response
            reflectable = noTSDA
            notreflectable = np.logical_not(noTSDA)
            tgtsegs = Segs[(Segs.sid == SID.ST_Foils()) & goodFit]
            tgtsegsref = tgtsegs[reflectable]
            tgtsegsnotref = tgtsegs[notreflectable]
            tgtmom = tgtsegs.mom.magnitude()
            tgtmomref = tgtsegsref.mom.magnitude()
            tgtmomnotref = tgtsegsnotref.mom.magnitude()
            tgtrho = tgtsegs.pos.rho()
            tgtrhoref = tgtsegsref.pos.rho()
            tgtrhonotref = tgtsegsnotref.pos.rho()
            tgtcost = tgtsegs.mom.cosTheta()
            tgtcostref = tgtsegsref.mom.cosTheta()
            tgtcostnotref = tgtsegsnotref.mom.cosTheta()
            tgtz = tgtsegs.pos.z()
            tgtzref = tgtsegsref.pos.z()
            tgtznotref = tgtsegsnotref.pos.z()
            ntgts = ak.count(tgtmom,axis=1)
            ntgtsref = ak.count(tgtmomref,axis=1)
            ntgtsnotref = ak.count(tgtmomnotref,axis=1)
            goodtgt = (ntgts > 0)
            goodtgtref = (ntgtsref > 0)
            goodtgtnotref = (ntgtsnotref > 0)
            ntgts = ntgts[goodtgt]
            ntgtsref = ntgtsref[goodtgtref]
            ntgtsnotref = ntgtsnotref[goodtgtnotref]
            avgmom = ak.sum(tgtmom,axis=1)
            avgmomref = ak.sum(tgtmomref,axis=1)
            avgmomnotref = ak.sum(tgtmomnotref,axis=1)
            avgmom = avgmom[goodtgt]/ntgts
            avgmomref = avgmomref[goodtgtref]/ntgtsref
            avgmomnotref = avgmomnotref[goodtgtnotref]/ntgtsnotref
            omomtgt = OMom[goodtgt]
            omomtgtref = OMom[goodtgtref]
            omomtgtnotref = OMom[goodtgtnotref]
            avgtgtresp = avgmom - omomtgt
            avgtgtrespref = avgmomref - omomtgtref
            avgtgtrespnotref = avgmomnotref - omomtgtnotref
            #
            self.HTgtAvgResp.fill(np.array(avgtgtresp))
            self.HTgtAvgRespRef.fill(np.array(avgtgtrespref))
            self.HTgtAvgRespNotRef.fill(np.array(avgtgtrespnotref))
            # find the latest target intersection
            tgtmomlate = tgtsegs[ak.argsort(tgtsegs.time,ascending=False)].mom.magnitude()
            tgtmomlate = ak.flatten(tgtmomlate[:,:1])
            tgtmomlateref = tgtsegsref[ak.argsort(tgtsegsref.time,ascending=False)].mom.magnitude()
            tgtmomlateref = ak.flatten(tgtmomlateref[:,:1])
            tgtmomlatenotref = tgtsegsnotref[ak.argsort(tgtsegsnotref.time,ascending=False)].mom.magnitude()
            tgtmomlatenotref = ak.flatten(tgtmomlatenotref[:,:1])

            latetgtresp = tgtmomlate - omomtgt
            latetgtrespref = tgtmomlateref - omomtgtref
            latetgtrespnotref = tgtmomlatenotref - omomtgtnotref
            self.HTgtLatestResp.fill(np.array(latetgtresp))
            self.HTgtLatestRespRef.fill(np.array(latetgtrespref))
            self.HTgtLatestRespNotRef.fill(np.array(latetgtrespnotref))

            self.HTgtRho.fill(np.array(ak.flatten(tgtrho)))
            self.HTgtRhoRef.fill(np.array(ak.flatten(tgtrhoref)))
            self.HTgtRhoNotRef.fill(np.array(ak.flatten(tgtrhonotref)))
            self.HTgtCosT.fill(np.array(ak.flatten(tgtcost)))
            self.HTgtCosTRef.fill(np.array(ak.flatten(tgtcostref)))
            self.HTgtCosTNotRef.fill(np.array(ak.flatten(tgtcostnotref)))
            self.HTgtFoil.fill(np.array(list(map(TargetFoil,ak.flatten(tgtz)))))
            self.HTgtFoilRef.fill(np.array(list(map(TargetFoil,ak.flatten(tgtzref)))))
            self.HTgtFoilNotRef.fill(np.array(list(map(TargetFoil,ak.flatten(tgtznotref)))))

            tgtsegsmc = SegsMC[(SegsMC.sid == SID.ST_Foils())]
            self.HTgtRhoMC.fill(np.array(ak.flatten(tgtsegsmc.pos.rho())))
            self.HTgtCosTMC.fill(np.array(ak.flatten(tgtsegsmc.mom.cosTheta())))
            self.HTgtFoilMC.fill(np.array(list(map(TargetFoil,ak.flatten(tgtsegsmc.pos.z())))))

            # legacy

            nopa = ak.count_nonzero(gSegs.sid==SID.OPA(),axis=1)

            self.HTDmomall.fill(np.array(ak.flatten(gSegs.mom.Z()/gSegs.mom.rho())))
            self.HTDLHall.fill(np.array(ak.flatten(glhpars.tanDip)))
            self.HTDparall.fill(np.array(ak.flatten(glhpars.lam/glhpars.rad)))
            fsel = gSegs.sid == SID.TT_Front()
            flhpars = glhpars[fsel]
            fsegs = gSegs[fsel]
            self.HTDmom.fill(np.array(ak.flatten(fsegs.mom.Z()/fsegs.mom.rho())))
            self.HTDLH.fill(np.array(ak.flatten(flhpars.lam/flhpars.rad)))
            self.HTDpar.fill(np.array(ak.flatten(flhpars.tanDip)))

            # test new cuts
            self.Hd0.fill(np.array(ak.flatten(flhpars.d0)))
            self.Hd0nstfe0.fill(np.array(ak.flatten(flhpars[nstf==0].d0)))
            self.Hd0nstfg0.fill(np.array(ak.flatten(flhpars[nstf>0].d0)))
            self.Hd0nstce0.fill(np.array(ak.flatten(flhpars[nstc==0].d0)))
            self.Hd0nstcg0.fill(np.array(ak.flatten(flhpars[nstc>0].d0)))

            self.Hrmax.fill(np.array(ak.flatten(flhpars.maxr)))
            self.Hrmaxnopae0.fill(np.array(ak.flatten(flhpars[nopa==0].maxr)))
            self.Hrmaxnopag0.fill(np.array(ak.flatten(flhpars[nopa>0].maxr)))

            cfsegs = fsegs[(nstf>0)&(nopa==0)]
            ccsegs = fsegs[(nstc>0)&(nopa==0)]
            cutsetc = (flhpars.d0<105) & (fsegs.mom.cosTheta() > 0.5) & (fsegs.mom.cosTheta() < 0.7071) & (flhpars.maxr > 450) & (flhpars.maxr < 680)
            ccfsegs = fsegs[cutsetc]

            self.HTD.fill(np.array(ak.flatten(fsegs.mom.Z()/fsegs.mom.rho())))
            self.HTDnf.fill(np.array(ak.flatten(cfsegs.mom.Z()/cfsegs.mom.rho())))
            self.HTDnc.fill(np.array(ak.flatten(ccsegs.mom.Z()/ccsegs.mom.rho())))
            self.HTDcc.fill(np.array(ak.flatten(ccfsegs.mom.Z()/ccfsegs.mom.rho())))

            self.HMom.fill(np.array(ak.flatten(fsegs.mom.magnitude())))
            self.HMomnf.fill(np.array(ak.flatten(cfsegs.mom.magnitude())))
            self.HMomnc.fill(np.array(ak.flatten(ccsegs.mom.magnitude())))
            self.HMomcc.fill(np.array(ak.flatten(ccfsegs.mom.magnitude())))

            self.HCosT.fill(np.array(ak.flatten(fsegs.mom.cosTheta())))
            self.HCosTnf.fill(np.array(ak.flatten(cfsegs.mom.cosTheta())))
            self.HCosTnc.fill(np.array(ak.flatten(ccsegs.mom.cosTheta())))
            self.HCosTcc.fill(np.array(ak.flatten(ccfsegs.mom.cosTheta())))

            # test for missing intersections
            hasent = (Segs.sid == 0) & TD.Match(self.tdir,Segs.mom.z())
            hasmid = (Segs.sid == 1) & TD.Match(self.tdir,Segs.mom.z())
            hasxit = (Segs.sid == 2) & TD.Match(self.tdir,Segs.mom.z())
            hasent = ak.any(hasent,axis=1)
            hasmid = ak.any(hasmid,axis=1)
            hasxit = ak.any(hasxit,axis=1)
            hasall = hasent & hasmid & hasxit
            missing = ak.count(hasall,0) - ak.count_nonzero(hasall)
            if(missing > 0):
                print("Found",missing,"Instfances of missing intersections in",ak.count(hasall,0),"tracks")
                for itrk in range(len(hasall)):
                    if (not hasall[itrk]):
                        print("Missing intersection: ",hasent[itrk],hasmid[itrk],hasxit[itrk]," eid ",runnum[itrk],":",subrun[itrk],":",event[itrk],sep="")
        print()


    def PlotQuality(self):
        fig, (anhits,afitcon,atrkqual) = plt.subplots(1,3,layout='constrained', figsize=(15,5))
        HistUtil.plot(self.HTrkQual,atrkqual)
        HistUtil.plot(self.HNHits,anhits)
        HistUtil.plot(self.HFitCon,afitcon)

    def PlotTrackerMomentum(self):
        fig, (amom,areso,aresp) = plt.subplots(3,3,layout='constrained', figsize=(15,15))
        for isid in range(len(self.TrackerSIDs)) :
            HistUtil.plot(self.HTrkFitMom[isid],amom[isid])
            HistUtil.plot(self.HTrkMCMom[isid],amom[isid])
            HistUtil.plot(self.HTrkResoMom[isid],areso[isid])
            HistUtil.plot(self.HTrkRespMom[isid],aresp[isid])
            HistUtil.plot(self.HTrkRefRespMom[isid],aresp[isid])
            HistUtil.plot(self.HTrkNotRefRespMom[isid],aresp[isid])
        amom[0].legend(loc="upper left")
        aresp[0].legend(loc="upper left")

    def PlotMaterial(self):
        fig, ([aninter,admom],[aninterMC,admomMC]) = plt.subplots(2,2,layout='constrained', figsize=(10,10))
        HistUtil.plot(self.HNIPA,aninter)
        HistUtil.plot(self.HNSTF,aninter)
        HistUtil.plot(self.HNSTC,aninter)
        HistUtil.plot(self.HNTSDA,aninter)
        HistUtil.plot(self.HNOPA,aninter)
        aninter.legend(loc="upper right")

        HistUtil.plot(self.HNIPAMC,aninterMC)
        HistUtil.plot(self.HNSTFMC,aninterMC)
        aninterMC.legend(loc="upper right")

        HistUtil.plot(self.HIPADMom,admom)
        HistUtil.plot(self.HSTDMom,admom)
        HistUtil.plot(self.HAllDMom,admom)
        admom.legend(loc="upper right")

        HistUtil.plot(self.HIPADMomMC,admomMC)
        HistUtil.plot(self.HSTDMomMC,admomMC)
        HistUtil.plot(self.HAllDMomMC,admomMC)
        admomMC.legend(loc="upper right")

    def PlotTarget(self):
        fig, ((arho,afoil,acost),(avgresp,latestresp,nnresp)) = plt.subplots(2,3,layout='constrained', figsize=(15,10))
        # Rho
        HistUtil.plot(self.HTgtRho,arho)
        HistUtil.plot(self.HTgtRhoRef,arho)
        HistUtil.plot(self.HTgtRhoNotRef,arho)
        HistUtil.plot(self.HTgtRhoMC,arho)
        HistUtil.plot(self.HOriginRho,arho)
        arho.legend(loc="upper left")
        # Foil
        HistUtil.plot(self.HTgtFoil,afoil)
        HistUtil.plot(self.HTgtFoilRef,afoil)
        HistUtil.plot(self.HTgtFoilNotRef,afoil)
        HistUtil.plot(self.HTgtFoilMC,afoil)
        HistUtil.plot(self.HOriginFoil,afoil)
        afoil.legend(loc="upper left")
        # Cos(theta)
        HistUtil.plot(self.HTgtCosT,acost)
        HistUtil.plot(self.HTgtCosTRef,acost)
        HistUtil.plot(self.HTgtCosTNotRef,acost)
        HistUtil.plot(self.HTgtCosTMC,acost)
        HistUtil.plot(self.HOriginCosT,acost)
        acost.legend(loc="upper left")
        # Response: Average
        HistUtil.plot(self.HTgtAvgResp,avgresp)
        HistUtil.plot(self.HTgtAvgRespRef,avgresp)
        HistUtil.plot(self.HTgtAvgRespNotRef,avgresp)
        avgresp.legend(loc="upper left")
        # Latest Foil
        HistUtil.plot(self.HTgtLatestResp,latestresp)
        HistUtil.plot(self.HTgtLatestRespRef,latestresp)
        HistUtil.plot(self.HTgtLatestRespNotRef,latestresp)
        latestresp.legend(loc="upper left")

    def PlotLegacy(self):
        fig, ((atd,atdc),(ad0,armax)) = plt.subplots(2,2,layout='constrained', figsize=(10,10))
        HistUtil.plot(self.HTDparall,atd)
        HistUtil.plot(self.HTDLHall,atd)
        HistUtil.plot(self.HTDmomall,atd)
        HistUtil.plot(self.HTDpar,atd)
        HistUtil.plot(self.HTDmom,atd)
        HistUtil.plot(self.HTDLH,atd)
        atd.legend(loc="upper right")
        HistUtil.plot(self.Hd0,ad0)
        HistUtil.plot(self.Hd0nstfe0,ad0)
        HistUtil.plot(self.Hd0nstfg0,ad0)
        HistUtil.plot(self.Hd0nstce0,ad0)
        HistUtil.plot(self.Hd0nstcg0,ad0)
        ad0.legend(loc="upper right")
        HistUtil.plot(self.Hrmax,armax)
        HistUtil.plot(self.Hrmaxnopae0,armax)
        HistUtil.plot(self.Hrmaxnopag0,armax)
        armax.legend(loc="upper right")
        HistUtil.plot(self.HTD,atdc)
        HistUtil.plot(self.HTDcc,atdc)
        HistUtil.plot(self.HTDnc,atdc)
        HistUtil.plot(self.HTDnf,atdc)
        atdc.legend(loc="upper right")
        fig, (amom,acost) = plt.subplots(1,2,layout='constrained', figsize=(10,5))
        HistUtil.plot(self.HMom,amom)
        HistUtil.plot(self.HMomcc,amom)
        HistUtil.plot(self.HMomnf,amom)
        HistUtil.plot(self.HMomnc,amom)
        amom.legend(loc="upper right")
        HistUtil.plot(self.HCosT,acost)
        HistUtil.plot(self.HCosTcc,acost)
        HistUtil.plot(self.HCosTnf,acost)
        HistUtil.plot(self.HCosTnc,acost)
        acost.legend(loc="upper left")
        print("CutSet C Count",np.sum(self.HCosTcc.view()))
        print("Cut on NCyl Count",np.sum(self.HCosTnc.view()))
        print("Cut on NFoil Count",np.sum(self.HCosTnf.view()))

    def Write(self,savefile):
        with h5py.File(savefile, 'w') as hdf5file:
            HistUtil.save_hist(self.HOriginMom,hdf5file)
            for isid in range(len(self.TrackerSIDs)) :
                HistUtil.save_hist(self.HTrkFitMom[isid],hdf5file)
                HistUtil.save_hist(self.HTrkMCMom[isid],hdf5file)
                HistUtil.save_hist(self.HTrkResoMom[isid],hdf5file)
                HistUtil.save_hist(self.HTrkRespMom[isid],hdf5file)
                HistUtil.save_hist(self.HTrkRefRespMom[isid],hdf5file)
                HistUtil.save_hist(self.HTrkNotRefRespMom[isid],hdf5file)
            HistUtil.save_hist(self.HTgtAvgResp,hdf5file)
            HistUtil.save_hist(self.HTgtRho,hdf5file)
            HistUtil.save_hist(self.HTgtRhoMC,hdf5file)
            HistUtil.save_hist(self.HOriginRho,hdf5file)
            HistUtil.save_hist(self.HTgtFoil,hdf5file)
            HistUtil.save_hist(self.HTgtFoilMC,hdf5file)
            HistUtil.save_hist(self.HOriginFoil,hdf5file)
            HistUtil.save_hist(self.HTgtCosT,hdf5file)
            HistUtil.save_hist(self.HTgtCosTMC,hdf5file)
            HistUtil.save_hist(self.HOriginCosT,hdf5file)
