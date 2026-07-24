#
# Display the branches in an EventNtuple TTree
#
import uproot
import awkward as ak
def Branches(file,tuple="EventNtuple/ntuple"):
    fullfile = file+":"+tuple
    with uproot.open(fullfile) as rfile:
        print(rfile.keys())
