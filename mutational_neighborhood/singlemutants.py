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


def superControlledSubstitution(location, encoding: Encoding, trap, mutation):
    """Perform a single mutation at a specific location where the replacement
        cell code is specified"""

    newTrap = copy.deepcopy(trap)
    newTrap[location] = mutation

    return newTrap

def generateAllMutantsAtCell(encoder, trap):
    """Return all possible mutations on a trap at a cellNumber"""
    listMutants = []
    for i in range(0, len(trap), 1):
        for k in range(2, len(constants.CELL_ALPHABET)):
            newTrap = superControlledSubstitution(i, encoder, trap, constants.CELL_ALPHABET[k])
            listMutants.append(newTrap)
            
    
    return listMutants


def getSingleMutationCL(encoder, trap):
    """Return all possible mutations on a trap at a cellNumber"""
    mutantCoherence, mutantLethality = [],[]
    for i in range(0, len(trap), 1):
        for k in range(2, len(constants.CELL_ALPHABET)):
            newTrap = superControlledSubstitution(i, encoder, trap, constants.CELL_ALPHABET[k])
            C, L = getCoherenceAndLethality(encoder, newTrap)
            mutantCoherence.append(C)
            mutantLethality.append(L)

            
    
    return mutantCoherence, mutantLethality


def getCoherentTraps(encoder):
    while True:
        trap = library.generateTrap()
        coherence, lethality = getCoherenceAndLethality(encoder, trap)
        if(coherence > 0 and lethality > 0):
            print(coherence)
            return trap


def getCoherenceAndLethality(encoder, trap):
    """Returns both the coherence and lethality of a trap"""
    coherence = fitnessFunctions.getCoherence(trap,encoder)
    lethality = fitnessFunctions.getLethality(trap,encoder)
    return coherence,lethality

def generateLethalTrap(encoder):
    best_lethality = 0
    best_trap = None
    num = 0

    while num < 100:
        trap = library.generateTrap()
        listMutants = generateAllMutantsAtCell(encoder, trap)
        for mutant in listMutants:
            coherence, lethality = getCoherenceAndLethality(encoder, mutant)
            if lethality > best_lethality:
                best_lethality = lethality
                best_trap = mutant
        num += 1

    return best_trap

def plotSingleMutants(encoder, trap):
    C,L = getSingleMutationCL(encoder, trap)
    trapC, trapL = getCoherenceAndLethality(encoder, trap)
    print(trapC, trapL)
    scatterplot(trapC, trapL, C, L)

def scatterplot(ogCoherence, ogLethality,lethalityArr, coherenceArr):
    #plt.scatter(lethalityArr, coherenceArr,color='r')
    newCoherence, newLethality, size = convertCohLetArrToStat(lethalityArr, coherenceArr)
    plt.scatter(newCoherence, newLethality,color='r', s = size, label = 'mutations')
    plt.scatter(ogCoherence, ogLethality,color='b',label='original')

    plt.xlabel("coherence")
    plt.ylabel("lethality")
    plt.show()
    plt.savefig('./plot1.png')


def convertCohLetArrToStat(lethalityArr, coherenceArr):
    arrTuples = []
    for lethal, coherence in zip(lethalityArr, coherenceArr):
        arrTuples.append((lethal, coherence))
    distinctTuples = Counter(arrTuples)

    newCoherence = []
    newLethality = []
    size = []
    for (key0,key1),val in list(distinctTuples.items()):
        newCoherence.append(key0)
        newLethality.append(key1)
        size.append(val)
    
    return newCoherence, newLethality, size

def getDoubleMutationCL(encoder, trap):
    """Return all possible mutations on a trap at a cellNumber"""
    newTraps, mutantCoherence, mutantLethality = [], [],[]
    for i in range(0, len(trap), 1):
        for k in range(2, len(constants.CELL_ALPHABET)):
            for j in range(i+1, len(trap), 1):
                for r in range(k+1, len(constants.CELL_ALPHABET)):
                    newTrap = superControlledSubstitution(i, encoder, trap, constants.CELL_ALPHABET[k])
                    newTrap = superControlledSubstitution(j, encoder, newTrap, constants.CELL_ALPHABET[r])
                    C, L = getCoherenceAndLethality(encoder, newTrap)
                    newTraps.append(newTrap)
                    mutantCoherence.append(C)
                    mutantLethality.append(L)
    return mutantCoherence, mutantLethality
def getTripleMutationCL(encoder, trap, size):
    """Return all possible mutations on a trap at a cellNumber"""
    newTraps, mutantCoherence, mutantLethality = [], [],[]
    num = 0
    seen = set()
    while len(seen) < size:
        pos = [] 
        alphabet = []
        for _ in range(3):
            pos.append(random.choice(range(0, len(trap))))
            alphabet.append(random.choice(range(2, len(constants.CELL_ALPHABET))))
        hash = tuple(pos + alphabet)
        if hash in seen:
            continue 
        elif len(set(pos))!=3:
            continue 
        else:
            seen.add(hash)
        newTrap = copy.deepcopy(trap)
        for i, k in zip(pos,alphabet):
            newTrap = superControlledSubstitution(i, encoder, newTrap, constants.CELL_ALPHABET[k])
        
        C, L = getCoherenceAndLethality(encoder, newTrap)
        #newTraps.append(newTrap)
        mutantCoherence.append(C)
        mutantLethality.append(L)
    return mutantCoherence, mutantLethality

def plotDoubleMutation(encoder, trap):
    C,L = getDoubleMutationCL(encoder, trap)
    trapC, trapL = getCoherenceAndLethality(encoder, trap)
    print(trapC, trapL)
    scatterplot(trapC, trapL, C, L)

def plotTripleMutation(encoder, trap):
    C,L = getTripleMutationCL(encoder, trap, 10000)
    trapC, trapL = getCoherenceAndLethality(encoder, trap)
    print(trapC, trapL)
    scatterplot(trapC, trapL, C, L)


def main():    
    encoder = Encoding() 
    trap = getCoherentTraps(encoder)
    #multimutatedtraps = getMultiMutatedDict(encoder, 3, 4)
    #X,Y,V = getPolar(multimutatedtraps,4, 0)
    #X,Y,V = getPolar(sampleDictionary, 1,1)
    #scatterplot(0, 0,X, Y)
    trap = generateLethalTrap(encoder)
    #print(lethal_trap)

    # trap = getCoherentTraps(encoder)
    #C, L = getSingleMutationCL(encoder, trap)
    #plotSingleMutants(encoder, trap)

    plotTripleMutation(encoder, trap)


  

if __name__ == "__main__":
    main()
