#
# Test extrapolation
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
from scipy.stats import norm
from scipy.stats import uniform
import SurfaceIds as SID
import MyHist
import h5py
from scipy.stats import crystalball

class Extrap(object):
    def __init__(self):
        self.HInterSID = MyHist.MyHist(name="SIDs",label="All",bins=110, range=[0,109],title="SID of Intersections",xlabel="SID")
        self.HTargetIndex = MyHist.MyHist(name="Tind",label="All",bins=38, range=[0,37],title="Index of Target Intersections",xlabel="Foil Index")
        self.HTargetdE = MyHist.MyHist(name="TFoildE",label="All",bins=100, range=[-1,0],title="dE of Target Foil Intersections",xlabel="dE (MeV)")
        self.HIPAdE = MyHist.MyHist(name="IPAdE",label="All",bins=100, range=[-1,0],title="dE of IPA Intersections",xlabel="dE (MeV)")

        self.HInterMCSID = MyHist.MyHist(name="MCSIDs",label="All",bins=110, range=[0,109],title="SID of MC Intersections",xlabel="SID")
        self.HTargetMCEDep = MyHist.MyHist(name="TFoilMCEDep",label="All",bins=100, range=[0,1],title="MC EDep of Target Foil Intersections",xlabel="MCEDep (MeV)")
        self.HIPAMCEDep = MyHist.MyHist(name="IPAMCEDep",label="All",bins=100, range=[0,1],title="MC EDep of IPA Intersections",xlabel="EDep (MeV)")
    def Loop(self,files):
        ibatch = 0
        np.set_printoptions(precision=5,floatmode='fixed')
        print("Processing batch ",end=' ')
        for batch,rep in uproot.iterate(files,filter_name="/evtinfo|trk.trk|trkmc|trksegs|trkmcsim|trksegsmc|trkqual|trksegpars_lh/i",report=True):
            print(ibatch,end=' ')
            segs = batch['trksegs'] # track fit samples
            segs = segs[:,0] # first track
            segsMC = batch['trksegsmc'] # SurfaceStep infor for true primary particle
            segsMC = segsMC[:,0] # first track

            self.HInterSID.fill(np.array(ak.flatten(segs.sid)))
            # target foils
            foilsegs = segs[segs.sid==SID.ST_Foils()]
            self.HTargetIndex.fill(np.array(ak.flatten(foilsegs.sindex)))
            self.HTargetdE.fill(np.array(ak.flatten(foilsegs.dmom)))
            # IPA intersections
            ipasegs = segs[segs.sid==SID.IPA()]
            self.HIPAdE.fill(np.array(ak.flatten(ipasegs.dmom)))
            #MC truth
            self.HInterMCSID.fill(np.array(ak.flatten(segsMC.sid)))
            foilsegsMC = segsMC[segsMC.sid==SID.ST_Foils()]
            self.HTargetMCEDep.fill(np.array(ak.flatten(foilsegsMC.edep)))
            ipasegsMC = segsMC[segsMC.sid==SID.IPA()]
            self.HIPAMCEDep.fill(np.array(ak.flatten(ipasegsMC.edep)))


    def PlotInters(self):
        fig, ( [sids,tindex],[tdE,ipadE]) = plt.subplots(2,2,layout='constrained', figsize=(10,10))
        self.HInterSID.plot(sids)
        self.HTargetIndex.plot(tindex)
        self.HTargetdE.plot(tdE)
        self.HIPAdE.plot(ipadE)
        print("total # of intersections",self.HInterSID.integral(),"Target Inters",self.HTargetIndex.integral(),"IPA Inters",self.HIPAdE.integral())

        fig, ( [sids,tindex],[tEDep,ipaEDep]) = plt.subplots(2,2,layout='constrained', figsize=(10,10))
        self.HInterMCSID.plot(sids)
#        self.HTargetMCInd.plot(tindex)
        self.HTargetMCEDep.plot(tEDep)
        self.HIPAMCEDep.plot(ipaEDep)

