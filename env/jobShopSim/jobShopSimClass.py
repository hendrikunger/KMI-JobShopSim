#!/usr/bin/env python3

import os
import random
import logging


import numpy as np
import pandas as pd

from jobShopSim.demo import demo


class jobShopSim:
 #------------------------------------------------------------------------------------------------------------
 # Loading
 #------------------------------------------------------------------------------------------------------------
    def __init__(self, path_to_ifc_file=None, path_to_materialflow_file = None,  randseed = None, randomPos = False, createMachines = False, verboseOutput = 0, maxMF_Elements = None):
        self.DRAWINGORIGIN = (0,0)
        self.MAXMF_ELEMENTS = maxMF_Elements
        
        self.RANDSEED = randseed
        self.pathPolygon = None
        self.MFIntersectionPoints = None
        random.seed(randseed)
            
        self.lasttime = 0        
        self.RatingDict = {}
        self.MachinesFarFromPath = set()
        self.machine_dict = None
        self.wall_dict = None
        self.machineCollisionList = []
        self.wallCollisionList = []
        self.outsiderList = []


 #------------------------------------------------------------------------------------------------------------
 # Do Stuff
 #------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------
def main():
    # Tests


    return

    
    
if __name__ == "__main__":
    main()

    