#
# Define track direction enum values in python
#
def Downstream():
    return 0
def Upstream():
    return 1
def Unknown():
    return 2
def DirectionName(tdir):
    match tdir:
        case 0:
            return "Downstream"
        case 1:
            return "Upstream"
        case 2:
            return "Unknown"

def Match(tdir,momz):
    match tdir:
        case 0:
            return momz > 0.0
        case 1:
            return momz < 0.0
        case 2:
            return True

