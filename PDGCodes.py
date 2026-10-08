# simple functions to test pdgcodes
def EMinus():
    return 11
def EPlus():
    return -11
def MuMinus():
    return 13
def MuPlus():
    return -13

def Match(pdgcode1,pdgcode2,testcharge):
    if not testcharge:
        return abs(pdgcode1) == abs(pdgcode2)
    else:
        return pdgcode1 == pdgcode2

def isElectron(pdgcode):
    return abs(pdgcode) == 11
def isMuon(pdgcode):
    return abs(pdgcode) == 13
def isEMinus(pdgcode):
    return pdgcode == 11
def isEPlus(pdgcode):
    return pdgcode == -11
def isMuMinus(pdgcode):
    return pdgcode == 13
def isMuPlus(pdgcode):
    return pdgcode == -13

def PDGName(pdgcode,usecharge=True):
    if usecharge:
        match pdgcode:
            case 11:
                return "EMinus"
            case -11:
                return "EPlus"
            case 13:
                return "MuMinus"
            case -13:
                return "MuPlus"
    else:
        match pdgcode:
            case 11 | -11:
                return "Electron"
            case 13 | -13:
                return "Muon"
    return "Unknown"
