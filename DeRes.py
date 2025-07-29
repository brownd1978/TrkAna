
# class to measure the resolution of high-energy electrons
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
import MyHist
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


class DeRes(object):
    def __init__(self,momrange,costrange,minNActive,minFitCon,minTrkQual):
        self.MomRange = momrange
        self.CosTRange = costrange
        self.minNActive = minNActive
        self.minFitCon = minFitCon
        self.minTrkQual = minTrkQual

        nDeltaMomBins = 200
        nMomBins = 200
        momrange=(self.MomRange[0],107)
        momresorange=(-2.5,2.5)
        momresprange=(-10,5)
        momtitle="Momentum at "
        momxlabel="Momentum (MeV)"

        self.TrkLoc = [None]*3
        self.HOriginMom = MyHist.MyHist(name="OriginMom",label="MC Origin",bins=nMomBins, range=momrange,title="Momentum at Origin",xlabel="Momentum (MeV)")

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
            self.HTrkFitMom[isid] = MyHist.MyHist(name=loc+"Mom",label="Fit",bins=nMomBins, range=momrange,title="Momentum"+loc,xlabel=momxlabel)
            self.HTrkMCMom[isid] = MyHist.MyHist(name=loc+"Mom",label="MC",bins=nMomBins, range=momrange,title="Momentum"+loc,xlabel=momxlabel)
            self.HTrkResoMom[isid] = MyHist.MyHist(name=loc+"Resolution",label="",bins=nDeltaMomBins, range=momresorange,title=momresotitle+loc,xlabel=dmomxlabel)
            self.HTrkRespMom[isid] = MyHist.MyHist(name=loc+"Response",label="All",bins=nDeltaMomBins, range=momresprange,title=momresptitle+loc,xlabel=dmomxlabel)
            self.HTrkRefRespMom[isid] = MyHist.MyHist(name=loc+"Response",label="NTSDA == 0",bins=nDeltaMomBins, range=momresprange,title=momresptitle+loc,xlabel=dmomxlabel)
            self.HTrkNotRefRespMom[isid] = MyHist.MyHist(name=loc+"Response",label="NTSDA > 0",bins=nDeltaMomBins, range=momresprange,title=momresptitle+loc,xlabel=dmomxlabel)
        # passive material
        nNMatBins = 15
        NMatRange = [-0.5,14.5]
        NMatxlabel = "N Intersections"
        NMattitle = "KKFit Intersections"
        NMattitleMC = "MC Intersections"
        self.HNSTF = MyHist.MyHist(bins=nNMatBins,range=NMatRange,name="NInter",label="ST Foil",xlabel=NMatxlabel,title=NMattitle)
        self.HNSTC = MyHist.MyHist(bins=nNMatBins,range=NMatRange,name="NInter",label="ST Cyl.",xlabel=NMatxlabel,title=NMattitle)
        self.HNIPA = MyHist.MyHist(bins=nNMatBins,range=NMatRange,name="NInter",label="IPA",xlabel=NMatxlabel,title=NMattitle)
        self.HNTSDA = MyHist.MyHist(bins=nNMatBins,range=NMatRange,name="NInter",label="TSDA",xlabel=NMatxlabel,title=NMattitle)
        self.HNOPA = MyHist.MyHist(bins=nNMatBins,range=NMatRange,name="NInter",label="OPA",xlabel=NMatxlabel,title=NMattitle)
        self.HNSTFMC = MyHist.MyHist(bins=nNMatBins,range=NMatRange,name="NInterMC",label="ST",xlabel=NMatxlabel,title=NMattitleMC)
        self.HNIPAMC = MyHist.MyHist(bins=nNMatBins,range=NMatRange,name="NInterMC",label="IPA",xlabel=NMatxlabel,title=NMattitleMC)
        # momentum change
        nDMomBins = 50
        dMomRange = [-3.0,0.0]
        dMomxlabel = "$\\Delta$ E (MeV)"
        dMomtitle = "KKFit Energy Loss"
        dMomtitleMC = "MC Energy Loss"
        self.HSTDMom = MyHist.MyHist(bins=nDMomBins,range=dMomRange,name="DMom",label="ST",xlabel=dMomxlabel,title=dMomtitle)
        self.HIPADMom = MyHist.MyHist(bins=nDMomBins,range=dMomRange,name="DMom",label="IPA",xlabel=dMomxlabel,title=dMomtitle)
        self.HAllDMom = MyHist.MyHist(bins=nDMomBins,range=dMomRange,name="DMom",label="All",xlabel=dMomxlabel,title=dMomtitle)
        self.HSTDMomMC = MyHist.MyHist(bins=nDMomBins,range=dMomRange,name="DMomMC",label="ST",xlabel=dMomxlabel,title=dMomtitleMC)
        self.HIPADMomMC = MyHist.MyHist(bins=nDMomBins,range=dMomRange,name="DMomMC",label="IPA",xlabel=dMomxlabel,title=dMomtitleMC)
        self.HAllDMomMC = MyHist.MyHist(bins=nDMomBins,range=dMomRange,name="DMomMC",label="All",xlabel=dMomxlabel,title=dMomtitleMC)



        # target intersections
        # momentum at target intersections

        tgtmomresptitle = "Target Momentum Response"
        self.HTgtAvgResp = MyHist.MyHist(name="AvgTgtResponse",label="Average",bins=nDeltaMomBins, range=momresprange,title=tgtmomresptitle,xlabel=dmomxlabel)
        self.HTgtAvgRespRef = MyHist.MyHist(name="AvgTgtResponseRef",label="Average (NTSDA == 0)",bins=nDeltaMomBins, range=momresprange,title=tgtmomresptitle,xlabel=dmomxlabel)
        self.HTgtAvgRespNotRef = MyHist.MyHist(name="AvgTgtResponseNotRef",label="Average (NTSDA > 0)",bins=nDeltaMomBins, range=momresprange,title=tgtmomresptitle,xlabel=dmomxlabel)

        self.HTgtLatestResp = MyHist.MyHist(name="LatestTgtResponse",label="Latest",bins=nDeltaMomBins, range=momresprange,title=tgtmomresptitle,xlabel=dmomxlabel)
        self.HTgtLatestRespRef = MyHist.MyHist(name="LatestTgtResponseRef",label="Latest (NTSDA == 0)",bins=nDeltaMomBins, range=momresprange,title=tgtmomresptitle,xlabel=dmomxlabel)
        self.HTgtLatestRespNotRef = MyHist.MyHist(name="LatestTgtResponseNotRef",label="Latest (NTSDA > 0)",bins=nDeltaMomBins, range=momresprange,title=tgtmomresptitle,xlabel=dmomxlabel)

        rhorange = [20,80]
        rhotitle ="Target Rho"
        rhoxlabel = "Rho (mm)"
        rhonbins=50
        self.HTgtRho = MyHist.MyHist(name="TgtRho",bins=rhonbins,range=rhorange,label="Fit",title=rhotitle,xlabel=rhoxlabel)
        self.HTgtRhoRef = MyHist.MyHist(name="TgtRho",bins=rhonbins,range=rhorange,label="Fit (NTSDA == 0)",title=rhotitle,xlabel=rhoxlabel)
        self.HTgtRhoNotRef = MyHist.MyHist(name="TgtRho",bins=rhonbins,range=rhorange,label="Fit (NTSDA > 0)",title=rhotitle,xlabel=rhoxlabel)
        self.HTgtRhoMC = MyHist.MyHist(name="TgtRho",bins=rhonbins,range=rhorange,label="MC",title=rhotitle,xlabel=rhoxlabel)
        self.HOriginRho = MyHist.MyHist(name="TgtRho",bins=rhonbins,range=rhorange,label="MC Origin",title=rhotitle,xlabel=rhoxlabel)
        foilrange = [-0.5,36.5]
        foiltitle ="Target Foil"
        foilxlabel="Foil #"
        foilnbins=37
        self.HTgtFoil = MyHist.MyHist(name="TgtFoil",bins=foilnbins,range=foilrange,label="Fit",title=foiltitle,xlabel=foilxlabel)
        self.HTgtFoilRef = MyHist.MyHist(name="TgtFoil",bins=foilnbins,range=foilrange,label="Fit (NTSDA == 0)",title=foiltitle,xlabel=foilxlabel)
        self.HTgtFoilNotRef = MyHist.MyHist(name="TgtFoil",bins=foilnbins,range=foilrange,label="Fit (NTSDA > 0)",title=foiltitle,xlabel=foilxlabel)
        self.HTgtFoilMC = MyHist.MyHist(name="TgtFoil",bins=foilnbins,range=foilrange,label="MC",title=foiltitle,xlabel=foilxlabel)
        self.HOriginFoil = MyHist.MyHist(name="TgtFoil",bins=foilnbins,range=foilrange,label="MC Origin",title=foiltitle,xlabel=foilxlabel)
        costrange = [-0.8,0.8]
        costtitle ="Target Cos($\\Theta$)"
        costxlabel="Cos($\\Theta$)"
        costnbins=50
        self.HTgtCosT = MyHist.MyHist(name="TgtCosT",bins=costnbins,range=costrange,label="Fit",title=costtitle,xlabel=costxlabel)
        self.HTgtCosTRef = MyHist.MyHist(name="TgtCosT",bins=costnbins,range=costrange,label="Fit (NTSDA == 0)",title=costtitle,xlabel=costxlabel)
        self.HTgtCosTNotRef = MyHist.MyHist(name="TgtCosT",bins=costnbins,range=costrange,label="Fit (NTSDA > 0)",title=costtitle,xlabel=costxlabel)
        self.HTgtCosTMC = MyHist.MyHist(name="TgtCosT",bins=costnbins,range=costrange,label="MC",title=costtitle,xlabel=costxlabel)
        self.HOriginCosT = MyHist.MyHist(name="TgtCosT",bins=costnbins,range=costrange,label="MC Origin",title=costtitle,xlabel=costxlabel)
        # fit quality
        self.HTrkQual = MyHist.MyHist(name="TrkQual",bins=100,range=[0.0,1.0],label="TrkQual",title="Track Quality",xlabel="ANN Result")
        self.HFitCon = MyHist.MyHist(name="FitCon",bins=100,range=[0.0,1.0],label="FitCon",title="Fit Consistency",xlabel="")
        cnbins = 150
        crange = [0.5,cnbins+0.5]
        self.HNH = MyHist.MyHist(name="Count",bins=cnbins,range=crange,label="Hits",title="Fit Count",xlabel="Count")
        self.HNHA = MyHist.MyHist(name="Count",bins=cnbins,range=crange,label="Active Hits",title="Fit Count",xlabel="Count")
        self.HNS = MyHist.MyHist(name="Count",bins=cnbins,range=crange,label="Straws",title="Fit Count",xlabel="Count")
        self.HNSA = MyHist.MyHist(name="Count",bins=cnbins,range=crange,label="Active Straws",title="Fit Count",xlabel="Count")
        self.HNMC = MyHist.MyHist(name="Count",bins=cnbins,range=crange,label="MC True Hits",title="Fit Count",xlabel="Count")

        self.HSDOCA = MyHist.MyHist(name="SDOCA",bins=100,range=[0.0,5.0],label="All",title="Straw DOCA",xlabel="DOCA (mm)")
        self.HSADOCA = MyHist.MyHist(name="SDOCA",bins=100,range=[0.0,5.0],label="Active",title="Straw DOCA",xlabel="DOCA (mm)")
        # legacy variables
#        for isid in range(len(self.TrackerSIDs)):
#            loc = "@"+SID.SurfaceName(self.TrackerSIDs[isid])

        TDrange = [-1.0,2.0]
        self.HTDmomall = MyHist.MyHist(name="TD",bins=50,range=TDrange,label="Pz/Pt (all)",title="TanDip",xlabel="Tan($\\lambda$)")
        self.HTDLHall = MyHist.MyHist(name="TD",bins=50,range=TDrange,label="$\\Lambda/R$ (all)",title="TanDip",xlabel="Tan($\\lambda$)")
        self.HTDparall = MyHist.MyHist(name="TD",bins=50,range=TDrange,label="tanDip (all)",title="TanDip",xlabel="Tan($\\lambda$)")
        self.HTDmom = MyHist.MyHist(name="TD",bins=50,range=TDrange,label="Pz/Pt@TT_Front",title="TanDip",xlabel="Tan($\\lambda$)")
        self.HTDLH = MyHist.MyHist(name="TD",bins=50,range=TDrange,label="$\\Lambda/R$@TT_Front",title="TanDip",xlabel="Tan($\\lambda$)")
        self.HTDpar = MyHist.MyHist(name="TD",bins=50,range=TDrange,label="tanDip@TT_Front",title="TanDip",xlabel="Tan($\\lambda$)")
        d0range = [-10,400]
        nd0=100
        self.Hd0 = MyHist.MyHist(name="d0",bins=nd0,range=d0range,label="No Cut",title="d0@TT_Front",xlabel="d$_{0}$ (mm)")
        self.Hd0cc = MyHist.MyHist(name="d0",bins=nd0,range=d0range,label="Cutset C'",title="d0@TT_Front",xlabel="d$_{0}$ (mm)")
        self.Hd0nstfe0 = MyHist.MyHist(name="d0",bins=nd0,range=d0range,label="N$_{ST Foil}$==0",title="d0@TT_Front",xlabel="d$_{0}$ (mm)")
        self.Hd0nstfg0 = MyHist.MyHist(name="d0",bins=nd0,range=d0range,label="N$_{ST Foil}$>0",title="d0@TT_Front",xlabel="d$_{0}$ (mm)")
        self.Hd0nstce0 = MyHist.MyHist(name="d0",bins=nd0,range=d0range,label="N$_{ST Cyl.}$==0",title="d0@TT_Front",xlabel="d$_{0}$ (mm)")
        self.Hd0nstcg0 = MyHist.MyHist(name="d0",bins=nd0,range=d0range,label="N$_{ST Cyl.}$>0",title="d0@TT_Front",xlabel="d$_{0}$ (mm)")
        rmaxrange = [425,725]
        self.Hrmax = MyHist.MyHist(name="rmax",bins=50,range=rmaxrange,label="No Cut",title="R$_{max}$@TT_Front",xlabel="R$_{max}$ (mm)")
        self.Hrmaxcc = MyHist.MyHist(name="rmax",bins=50,range=rmaxrange,label="Cutset C'",title="R$_{max}$@TT_Front",xlabel="R$_{max}$ (mm)")
        self.Hrmaxnopae0 = MyHist.MyHist(name="rmax",bins=50,range=rmaxrange,label="N$_{OPA}$==0",title="R$_{max}$@TT_Front",xlabel="R$_{max}$ (mm)")
        self.Hrmaxnopag0 = MyHist.MyHist(name="rmax",bins=50,range=rmaxrange,label="N$_{OPA}$>0",title="R$_{max}$@TT_Front",xlabel="R$_{max}$ (mm)")
        TDrange = [0.0,2.0]
        self.HTD = MyHist.MyHist(name="TD",bins=50,range=TDrange,label="No Cut",title="TanDip@TT_Front",xlabel="Tan($\\lambda$)")
        self.HTDcc = MyHist.MyHist(name="TD",bins=50,range=TDrange,label="Cutset C",title="TanDip@TT_Front",xlabel="Tan($\\lambda$)")
        self.HTDnf = MyHist.MyHist(name="TD",bins=50,range=TDrange,label="N$_{ST Foil}$>0 & N$_{OPA}$==0",title="TanDip@TT_Front",xlabel="Tan($\\lambda$)")
        self.HTDnc = MyHist.MyHist(name="TD",bins=50,range=TDrange,label="N$_{ST Cyl.}$>0 & N$_{OPA}$==0",title="TanDip@TT_Front",xlabel="Tan($\\lambda$)")
        momrange=[85,125]
        nmom = 100
        self.HMom = MyHist.MyHist(name="Mom",bins=nmom,range=momrange,label="No Cut",title="Momentum@TT_Front",xlabel="Momentum (MeV)")
        self.HMomcc = MyHist.MyHist(name="Mom",bins=nmom,range=momrange,label="Cutset C",title="Momentum@TT_Front",xlabel="Momentum (MeV)")
        self.HMomnf = MyHist.MyHist(name="Mom",bins=nmom,range=momrange,label="N$_{ST Foil}$>0 & N$_{OPA}$==0",title="Momentum@TT_Front",xlabel="Momentum (MeV)")
        self.HMomnc = MyHist.MyHist(name="Mom",bins=nmom,range=momrange,label="N$_{ST Cyl.}$>0 & N$_{OPA}$==0",title="Momentum@TT_Front",xlabel="Momentum (MeV)")
        costrange=[0.0,1.0]
        self.HCosT = MyHist.MyHist(name="CosT",bins=50,range=costrange,label="No Cut",title="Cos($\\Theta$)@TT_Front",xlabel="P$_{z}$/P")
        self.HCosTcc = MyHist.MyHist(name="CosT",bins=50,range=costrange,label="Cutset C",title="Cos($\\Theta$)@TT_Front",xlabel="P$_{z}$/P")
        self.HCosTnf = MyHist.MyHist(name="CosT",bins=50,range=costrange,label="N$_{ST Foil}$>0 & N$_{OPA}$==0",title="Cos($\\Theta$)@TT_Front",xlabel="P$_{z}$/P")
        self.HCosTnc = MyHist.MyHist(name="CosT",bins=50,range=costrange,label="N$_{ST Cyl.}$>0 & N$_{OPA}$==0",title="Cos($\\Theta$)@TT_Front",xlabel="P$_{z}$/P")

    def Loop(self,files):
        elPDG = 11
        ibatch = 0
        np.set_printoptions(precision=5,floatmode='fixed')
        print("Processing batch ",end=' ')
        for batch,rep in uproot.iterate(files,filter_name="/evtinfo|trk|trkmc|trksegs|trkmcsim|trksegsmc|trkqual|trksegpars_lh/i",report=True):
            print(ibatch,end=' ')
            ibatch = ibatch+1
            runnum = batch['run']
            subrun = batch['subrun']
            event = batch['event']
            segs = batch['trksegs'] # track fit samples
            mats = batch['trkmats'] # track passive materials (straws)
            lhpars = batch['trksegpars_lh'] # track fit samples
            nhits = batch['trk.nhits']  # track N hits
            nhactive = batch['trk.nactive']
            nstraws = batch['trk.nmat']
            nsactive = batch['trk.nmatactive']
            fitcon = batch['trk.fitcon']  # track fit consistency
            trkQual = batch['trkqual.result']  # track fit quality
            trkMCSim = batch['trkmcsim']  # MC genealogy of particles
            trkMCndigi = batch['trkmc.ndigigood']  # MC true # of straw hits
            segsMC = batch['trksegsmc'] # SurfaceStep infor for true primary particle
            # should be 1 track/event
            assert(ak.sum(ak.count_nonzero(nhits,axis=1)!=1) == 0)
            segs = segs[:,0]
            lhpars = lhpars[:,0]
#            print("segs len",len(segs),"lhpars len",len(lhpars))
            FitCon = fitcon[:,0]
            NH = nhits[:,0]
            NHA = nhactive[:,0]
            NS = nstraws[:,0]
            NSA = nsactive[:,0]
            TrkQual = trkQual[:,0]
            mats = mats[:,0]

            # define good MC selection first, to allow downstfream comparisons
            segsMC = segsMC[:,0] # segments (of 1st MC match) of 1st track
            trkMCSim = trkMCSim[:,0,0] # primary MC match of 1st track
            trkMCndigi = trkMCndigi[:,0]

            # basic consistency test
            assert((len(runnum) == len( segs)) & (len(segs) == len(segsMC)) & (len(segs) == len(trkMCSim)) & (len(NHA) == len(segs)))
            goodMC = (trkMCSim.pdg == elPDG) & (trkMCSim.trkrel._rel == 0)
            OMom = trkMCSim.mom.magnitude()
            goodMC = goodMC & (OMom>self.MomRange[0]) & (OMom < self.MomRange[1])
            OMom = OMom[goodMC]
            self.HOriginMom.fill(np.array(OMom))
            self.HOriginRho.fill(np.array(trkMCSim[goodMC].pos.rho()))
            self.HOriginCosT.fill(np.array(trkMCSim[goodMC].mom.cosTheta()))
            self.HOriginFoil.fill(np.array(list(map(TargetFoil,trkMCSim[goodMC].pos.z()))))

            # truncate accordingly
            segsMC = segsMC[goodMC]
            segs = segs[goodMC]
            lhpars = lhpars[goodMC]
            NHA = NHA[goodMC]
            FitCon = FitCon[goodMC]
            TrkQual = TrkQual[goodMC]
            mats = mats[goodMC]
            midsegs = segs[(segs.sid == SID.TT_Mid()) & (segs.mom.z() > 0.0) ]
            CosT = ak.flatten(midsegs.mom.cosTheta())
# not all (cosmic) tracks go through TT_Mid
            goodFit = (NHA >= self.minNActive) & (FitCon > self.minFitCon) & (TrkQual > self.minTrkQual)
            TSDASeg = segs[segs.sid == SID.TSDA() ]
            noTSDA = ak.num(TSDASeg)==0

            self.HTrkQual.fill(np.array(TrkQual))
            self.HFitCon.fill(np.array(FitCon))
            self.HNHA.fill(np.array(NHA))
            self.HNH.fill(np.array(NH[goodMC]))
            self.HNS.fill(np.array(NS[goodMC]))
            self.HNSA.fill(np.array(NSA[goodMC]))
            self.HNMC.fill(np.array(trkMCndigi[goodMC]))

            self.HSDOCA.fill(np.array(ak.flatten(mats,axis=1).doca))
            self.HSADOCA.fill(np.array(ak.flatten(mats[mats.active],axis=1).doca))

            # sample the fits at the specified
            for isid in range(len(self.TrackerSIDs)) :
                sid = self.TrackerSIDs[isid]
                ssegs = segs[(segs.sid == sid) & (segs.mom.z() > 0.0) ]
                mom = ssegs.mom.magnitude()
                mom = mom[(mom > self.MomRange[0]) & (mom < self.MomRange[1])]
                hasmom = ak.count_nonzero(mom,axis=1)==1
                ssegsMC = segsMC[(segsMC.sid == sid) & (segsMC.mom.z() > 0.0) ]
                momMC = ssegsMC.mom.magnitude()
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
            gsegs = segs[goodFit]
            glhpars = lhpars[goodFit]
            gstf = gsegs.sid==SID.ST_Foils()
            gstc = np.logical_or((gsegs.sid==SID.ST_Outer()),(gsegs.sid==SID.ST_Front()))
            nstf = ak.count_nonzero(gstf,axis=1)
            nstc = ak.count_nonzero(gstc,axis=1)

            self.HNSTF.fill(np.array(nstf))
            self.HNSTC.fill(np.array(nstc))
            self.HNIPA.fill(np.array(ak.count_nonzero(gsegs.sid==SID.IPA(),axis=1)))
            self.HNTSDA.fill(np.array(ak.count_nonzero(gsegs.sid==SID.TSDA(),axis=1)))
            self.HNOPA.fill(np.array(ak.count_nonzero(gsegs.sid==SID.OPA(),axis=1)))
            foilsegs = segs.sid==SID.ST_Foils()
            ipasegs = segs.sid==SID.IPA()
            stdmom = ak.sum(segs[foilsegs].dmom,axis=1)
            ipadmom = ak.sum(segs[ipasegs].dmom,axis=1)
            stdmom = stdmom[goodFit]
            ipadmom = ipadmom[goodFit]
            self.HSTDMom.fill(np.array(stdmom))
            self.HIPADMom.fill(np.array(ipadmom))
            self.HAllDMom.fill(np.array(stdmom + ipadmom))
            # Also for MC
            self.HNSTFMC.fill(np.array(ak.count_nonzero(segsMC[goodFit].sid==SID.ST_Foils(),axis=1)))
            self.HNIPAMC.fill(np.array(ak.count_nonzero(segsMC[goodFit].sid==SID.IPA(),axis=1)))
            foilsegsMC = segsMC.sid==SID.ST_Foils()
            ipasegsMC = segsMC.sid==SID.IPA()
            stdmomMC = ak.sum(-segsMC[foilsegsMC].edep,axis=1)
            ipadmomMC = ak.sum(-segsMC[ipasegsMC].edep,axis=1)
            stdmomMC = stdmomMC[goodFit]
            ipadmomMC = ipadmomMC[goodFit]
            self.HSTDMomMC.fill(np.array(stdmomMC))
            self.HIPADMomMC.fill(np.array(ipadmomMC))
            self.HAllDMomMC.fill(np.array(stdmomMC + ipadmomMC))

            #foil response
            reflectable = noTSDA
            notreflectable = np.logical_not(noTSDA)
            tgtsegs = segs[(segs.sid == SID.ST_Foils()) & goodFit]
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

            tgtsegsmc = segsMC[(segsMC.sid == SID.ST_Foils())]
            self.HTgtRhoMC.fill(np.array(ak.flatten(tgtsegsmc.pos.rho())))
            self.HTgtCosTMC.fill(np.array(ak.flatten(tgtsegsmc.mom.cosTheta())))
            self.HTgtFoilMC.fill(np.array(list(map(TargetFoil,ak.flatten(tgtsegsmc.pos.z())))))

            # legacy
            nopa = ak.count_nonzero(gsegs.sid==SID.OPA(),axis=1)

            self.HTDmomall.fill(np.array(ak.flatten(gsegs.mom.Z()/gsegs.mom.rho())))
            self.HTDLHall.fill(np.array(ak.flatten(glhpars.tanDip)))
            self.HTDparall.fill(np.array(ak.flatten(glhpars.lam/glhpars.rad)))
            fsel = gsegs.sid == SID.TT_Front()
            flhpars = glhpars[fsel]
            fsegs = gsegs[fsel]
            self.HTDmom.fill(np.array(ak.flatten(fsegs.mom.Z()/fsegs.mom.rho())))
            self.HTDLH.fill(np.array(ak.flatten(flhpars.lam/flhpars.rad)))
            self.HTDpar.fill(np.array(ak.flatten(flhpars.tanDip)))

            # test new cuts
            cutsetcf = (flhpars.d0<100) & (flhpars.maxr < 680) & (flhpars.maxr > 450) & (fsegs.mom.cosTheta()>0.5) &(fsegs.mom.cosTheta()<0.70711)
            cutsetc = (flhpars.d0<100) & (flhpars.maxr < 680) & (flhpars.maxr > 450)
            ccflhpars = flhpars[cutsetc]
            ccfsegs = fsegs[cutsetc]
            ccfflhpars = flhpars[cutsetcf]
            ccffsegs = fsegs[cutsetcf]

            self.Hd0.fill(np.array(ak.flatten(flhpars.d0)))
            self.Hd0cc.fill(np.array(ak.flatten(ccflhpars.d0)))
            self.Hd0nstfe0.fill(np.array(ak.flatten(flhpars[nstf==0].d0)))
            self.Hd0nstfg0.fill(np.array(ak.flatten(flhpars[nstf>0].d0)))
            self.Hd0nstce0.fill(np.array(ak.flatten(flhpars[nstc==0].d0)))
            self.Hd0nstcg0.fill(np.array(ak.flatten(flhpars[nstc>0].d0)))

            self.Hrmax.fill(np.array(ak.flatten(flhpars.maxr)))
            self.Hrmaxcc.fill(np.array(ak.flatten(ccflhpars.maxr)))
            self.Hrmaxnopae0.fill(np.array(ak.flatten(flhpars[nopa==0].maxr)))
            self.Hrmaxnopag0.fill(np.array(ak.flatten(flhpars[nopa>0].maxr)))

            cfsegs = fsegs[(nstf>0)&(nopa==0)]
            ccsegs = fsegs[(nstc>0)&(nopa==0)]
            #cutsetc = (flhpars.d0<105) & (fsegs.mom.cosTheta() > 0.5) & (fsegs.mom.cosTheta() < 0.7071) & (flhpars.maxr > 450) & (flhpars.maxr < 680)

            self.HTD.fill(np.array(ak.flatten(fsegs.mom.Z()/fsegs.mom.rho())))
            self.HTDnf.fill(np.array(ak.flatten(cfsegs.mom.Z()/cfsegs.mom.rho())))
            self.HTDnc.fill(np.array(ak.flatten(ccsegs.mom.Z()/ccsegs.mom.rho())))
            self.HTDcc.fill(np.array(ak.flatten(ccffsegs.mom.Z()/ccffsegs.mom.rho())))

            self.HMom.fill(np.array(ak.flatten(fsegs.mom.magnitude())))
            self.HMomnf.fill(np.array(ak.flatten(cfsegs.mom.magnitude())))
            self.HMomnc.fill(np.array(ak.flatten(ccsegs.mom.magnitude())))
            self.HMomcc.fill(np.array(ak.flatten(ccffsegs.mom.magnitude())))

            self.HCosT.fill(np.array(ak.flatten(fsegs.mom.cosTheta())))
            self.HCosTnf.fill(np.array(ak.flatten(cfsegs.mom.cosTheta())))
            self.HCosTnc.fill(np.array(ak.flatten(ccsegs.mom.cosTheta())))
            self.HCosTcc.fill(np.array(ak.flatten(ccffsegs.mom.cosTheta())))

            # test for missing intersections
            hasent = (segs.sid == 0) & (segs.mom.z() > 0.0)
            hasmid = (segs.sid == 1) & (segs.mom.z() > 0.0)
            hasxit = (segs.sid == 2) & (segs.mom.z() > 0.0)
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
        fig, (acount,smat) = plt.subplots(1,2,layout='constrained', figsize=(10,5))
        self.HNH.plot(acount)
        self.HNHA.plot(acount)
        self.HNS.plot(acount)
        self.HNSA.plot(acount)
        self.HNMC.plot(acount)
        acount.legend(loc="upper right")
        self.HSDOCA.plot(smat)
        self.HSADOCA.plot(smat)
        smat.legend(loc="upper right")
        fig, (afitcon,atrkqual) = plt.subplots(1,2,layout='constrained', figsize=(10,5))
        self.HTrkQual.plot(atrkqual)
        self.HFitCon.plot(afitcon)

    def PlotTrackerMomentum(self):
        fig, (amom,areso,aresp) = plt.subplots(3,3,layout='constrained', figsize=(15,15))
        for isid in range(len(self.TrackerSIDs)) :
            self.HTrkFitMom[isid].plot(amom[isid])
            self.HTrkMCMom[isid].plot(amom[isid])
            self.HTrkResoMom[isid].plot(areso[isid])
            self.HTrkRespMom[isid].plot(aresp[isid])
            self.HTrkRefRespMom[isid].plot(aresp[isid])
            self.HTrkNotRefRespMom[isid].plot(aresp[isid])
        amom[0].legend(loc="upper left")
        aresp[0].legend(loc="upper left")

    def PlotMaterial(self):
        fig, ([aninter,admom],[aninterMC,admomMC]) = plt.subplots(2,2,layout='constrained', figsize=(10,10))
        self.HNIPA.plot(aninter)
        self.HNSTF.plot(aninter)
        self.HNSTC.plot(aninter)
        self.HNTSDA.plot(aninter)
        self.HNOPA.plot(aninter)
        aninter.legend(loc="upper right")

        self.HNIPAMC.plot(aninterMC)
        self.HNSTFMC.plot(aninterMC)
        aninterMC.legend(loc="upper right")

        self.HIPADMom.plot(admom)
        self.HSTDMom.plot(admom)
        self.HAllDMom.plot(admom)
        admom.legend(loc="upper right")

        self.HIPADMomMC.plot(admomMC)
        self.HSTDMomMC.plot(admomMC)
        self.HAllDMomMC.plot(admomMC)
        admomMC.legend(loc="upper right")

    def PlotTarget(self):
        fig, ((arho,afoil,acost),(avgresp,latestresp,nnresp)) = plt.subplots(2,3,layout='constrained', figsize=(15,10))
        # Rho
        self.HTgtRho.plot(arho)
        self.HTgtRhoRef.plot(arho)
        self.HTgtRhoNotRef.plot(arho)
        self.HTgtRhoMC.plot(arho)
        self.HOriginRho.plot(arho)
        arho.legend(loc="upper left")
        # Foil
        self.HTgtFoil.plot(afoil)
        self.HTgtFoilRef.plot(afoil)
        self.HTgtFoilNotRef.plot(afoil)
        self.HTgtFoilMC.plot(afoil)
        self.HOriginFoil.plot(afoil)
        afoil.legend(loc="upper left")
        # Cos(theta)
        self.HTgtCosT.plot(acost)
        self.HTgtCosTRef.plot(acost)
        self.HTgtCosTNotRef.plot(acost)
        self.HTgtCosTMC.plot(acost)
        self.HOriginCosT.plot(acost)
        acost.legend(loc="upper left")
        # Response: Average
        self.HTgtAvgResp.plot(avgresp)
        self.HTgtAvgRespRef.plot(avgresp)
        self.HTgtAvgRespNotRef.plot(avgresp)
        avgresp.legend(loc="upper left")
        # Latest Foil
        self.HTgtLatestResp.plot(latestresp)
        self.HTgtLatestRespRef.plot(latestresp)
        self.HTgtLatestRespNotRef.plot(latestresp)
        latestresp.legend(loc="upper left")

    def PlotLegacy(self):
        fig, atd = plt.subplots(1,1,layout='constrained', figsize=(10,10))
        self.HTDparall.plot(atd)
        self.HTDLHall.plot(atd)
        self.HTDmomall.plot(atd)
        self.HTDpar.plot(atd)
        self.HTDmom.plot(atd)
        self.HTDLH.plot(atd)
        atd.legend(loc="upper right")
        fig, (ad0,armax,atdc) = plt.subplots(1,3,layout='constrained', figsize=(15,5))
        self.Hd0.plot(ad0)
        self.Hd0cc.plot(ad0)
#        self.Hd0nstfe0.plot(ad0)
#        self.Hd0nstfg0.plot(ad0)
        self.Hd0nstce0.plot(ad0)
        self.Hd0nstcg0.plot(ad0)
        ad0.legend(loc="upper right")
        self.Hrmax.plot(armax)
        self.Hrmaxcc.plot(armax)
        self.Hrmaxnopae0.plot(armax)
        self.Hrmaxnopag0.plot(armax)
        armax.legend(loc="upper right")
        self.HTD.plot(atdc)
        self.HTDcc.plot(atdc)
        self.HTDnc.plot(atdc)
#        self.HTDnf.plot(atdc)
        atdc.legend(loc="upper right")
        fig, (amom,acost) = plt.subplots(1,2,layout='constrained', figsize=(10,5))
        self.HMom.plot(amom)
        self.HMomcc.plot(amom)
#        self.HMomnf.plot(amom)
        self.HMomnc.plot(amom)
        amom.legend(loc="upper right")
        self.HCosT.plot(acost)
        self.HCosTcc.plot(acost)
#        self.HCosTnf.plot(acost)
        self.HCosTnc.plot(acost)
        acost.legend(loc="upper left")
        print("Nocut Count",np.sum(self.HCosT.data))
        print("CutSet C Count",np.sum(self.HCosTcc.data))
        print("Cut on NCyl Count",np.sum(self.HCosTnc.data))
        print("Cut on NFoil Count",np.sum(self.HCosTnf.data))

        print("Nocut Count",np.sum(self.Hd0.data))
        print("CutSet C Count",np.sum(self.Hd0cc.data))
        print("Cut on NCyl Count",np.sum(self.Hd0nstcg0.data))
        print("Cut on NFoil Count",np.sum(self.Hd0nstfg0.data))

    def Write(self,savefile):
        with h5py.File(savefile, 'w') as hdf5file:
            self.HOriginMom.save(hdf5file)
            for isid in range(len(self.TrackerSIDs)) :
                self.HTrkFitMom[isid].save(hdf5file)
                self.HTrkMCMom[isid].save(hdf5file)
                self.HTrkResoMom[isid].save(hdf5file)
                self.HTrkRespMom[isid].save(hdf5file)
                self.HTrkRefRespMom[isid].save(hdf5file)
                self.HTrkNotRefRespMom[isid].save(hdf5file)
            self.HTgtAvgResp.save(hdf5file)
            self.HTgtRho.save(hdf5file)
            self.HTgtRhoMC.save(hdf5file)
            self.HOriginRho.save(hdf5file)
            self.HTgtFoil.save(hdf5file)
            self.HTgtFoilMC.save(hdf5file)
            self.HOriginFoil.save(hdf5file)
            self.HTgtCosT.save(hdf5file)
            self.HTgtCosTMC.save(hdf5file)
            self.HOriginCosT.save(hdf5file)
