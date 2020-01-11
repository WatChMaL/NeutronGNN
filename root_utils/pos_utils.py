import numpy as np

row_remap=np.flip(np.arange(16))

"""
The index starts at 1 and counts up continuously with no gaps
Each 19 consecutive PMTs belong to one mPMT module, so (index-1)/19 is the module number.
The index%19 gives the position in the module: 1-12 is the outer ring, 13-18 is inner ring, 0 is the centre PMT
The modules are then ordered as follows:
It starts by going round the second highest ring around the barrel, then the third highest ring, fourth highest
ring, all the way down to the lowest ring (i.e. skips the highest ring).
Then does the bottom end-cap, row by row (the first row has 6 modules, the second row has 8, then 10, 10, 10,
10, 10, 10, 8, 6).
Then the highest ring around the barrel that was skipped before, then the top end-cap, row by row.
I'm not sure why it has this somewhat strange order...
WTF: actually it is: 2, 6, 8 10, 10, 12 and down again in the caps
"""

def num_modules():
    """Returns the total number of mPMT modules"""
    return 536

def num_pmts():
    """Returns the total number of PMTs"""
    return num_modules()*19

def module_index(pmt_index):
    """Returns the module number given the 0-indexed pmt number"""
    return pmt_index//19

def pmt_in_module_id(pmt_index):
    """Returns the pmt number within a 
    module given the 0-indexed pmt number"""
    return pmt_index%19

def is_barrel(module_index):
    """Returns True if module is in the Barrel"""
    return ( (module_index<320) | ((module_index>=408)&(module_index<448)) )

def is_bottom(module_index):
    """Returns True if module is in the bottom cap"""
    return ( (module_index>=320)&(module_index<408) )

def is_top(module_index):
    """Returns True if module is in the top cap"""
    return ( (module_index>=448)&(module_index<536) )