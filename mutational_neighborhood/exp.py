#!/usr/bin/env python3
import geneticAlgorithm.library as library
import geneticAlgorithm.fitnessFunctions as fitnessFunctions
from classes.Encoding import Encoding
from classes.Trap import sampleRandomBoards
from classes.Trap import Trap
from classes.Board import Board 
import random
import geneticAlgorithm.constants as constants
import numpy as np
import copy
import matplotlib.pyplot as plt
from collections import Counter
import csv 
from mpl_toolkits.mplot3d import Axes3D
import math
import pandas as pd

def getCoherenceAndLethality(encoder, trap):
    """Returns both the coherence and lethality of a trap"""
    coherence = fitnessFunctions.getCoherence(trap,encoder)
    lethality = fitnessFunctions.getLethality(trap,encoder)
    return coherence,lethality

def singleMutationSubstitution(location, encoding: Encoding, trap, mutation):
    """Perform a single mutation at a specific location where the replacement
        cell code is specified"""
    newTrap = copy.deepcopy(trap)
    newTrap[location] = mutation
    return newTrap

def getSingleMutationCL(encoder, trap):
    """Return all possible mutations on a trap at a cellNumber"""
    newTraps, mutantCoherence, mutantLethality = [], [],[]
    for i in range(0, len(trap), 1):
        for k in range(2, len(constants.CELL_ALPHABET)):
            newTrap = singleMutationSubstitution(i, encoder, trap, constants.CELL_ALPHABET[k])
            C, L = getCoherenceAndLethality(encoder, newTrap)
            newTraps.append(newTrap)
            mutantCoherence.append(C)
            mutantLethality.append(L)
    return trap, newTraps,mutantCoherence, mutantLethality

def getDoubleMutationCL(encoder, trap):
    """Return all possible mutations on a trap at a cellNumber"""
    newTraps, mutantCoherence, mutantLethality = [], [],[]
    for i in range(0, len(trap), 1):
        for k in range(2, len(constants.CELL_ALPHABET)):
            for j in range(i+1, len(trap), 1):
                for r in range(k+1, len(constants.CELL_ALPHABET)):
                    newTrap = singleMutationSubstitution(i, encoder, trap, constants.CELL_ALPHABET[k])
                    newTrap = singleMutationSubstitution(j, encoder, newTrap, constants.CELL_ALPHABET[r])
                    C, L = getCoherenceAndLethality(encoder, newTrap)
                    newTraps.append(newTrap)
                    mutantCoherence.append(C)
                    mutantLethality.append(L)
    return trap, newTraps,mutantCoherence, mutantLethality

def load(file):
    csv = pd.read_csv(file)
    return csv 

def save(groupEntry, pd,file):
    pass


