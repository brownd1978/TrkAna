#
# Display the branches in an EventNtuple TTree
#
import uproot
import awkward as ak
def Branches(file):
    fullfile = file+":EventNtuple/ntuple"
    with uproot.open(fullfile) as rfile:
        print(rfile.keys())
