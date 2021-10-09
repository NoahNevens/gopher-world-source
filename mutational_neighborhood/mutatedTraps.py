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


def generateMutated(encoder, trap, mutationFunction = library.mutationFunc, numMutants = 2000):
    """Generates a list of possible mutated traps produced from mutating the same trap"""
    listMutants = [None for _ in range(numMutants)]
    for i in range(numMutants):
        listMutants[i] =  mutationFunction(encoder, copy.deepcopy(trap))
    
    return listMutants


def computeChanges(encoder,mutatedTraps):
    """Compares the coherence and lethality of the mutated traps to the original trap"""
    newCoherences = [0 for _ in range(len(mutatedTraps))]
    newLethalities = [0 for _ in range(len(mutatedTraps))]

    for i in range(len(mutatedTraps)):
        newCoherences[i], newLethalities[i] = getCoherenceAndLethality(encoder, mutatedTraps[i])

    return newCoherences, newLethalities

def controlledSubstitution(location, encoding: Encoding, trap, mutations):
    """Perform a substitution mutation at a specified and valid location in the trap encoding"""
    
    listMutants = [None for _ in range(mutations)]

    while location in (encoding.food, encoding.floor, encoding.door):
        location = random.randrange(0, len(trap), 1)
    
    for i in range(mutations):
        newTrap = copy.deepcopy(trap)
        newTrap[location] = constants.CELL_ALPHABET[random.randrange(2, len(constants.CELL_ALPHABET), 1)]
        listMutants[i] = newTrap

    return listMutants

def controlledSubstitution2(location, encoding: Encoding, trap, subList):
    """Perform a substitution at a speicified and valid location with a
       modified list of possible substitutions"""

    listMutants = [None for _ in range(subList)]

    while location in (encoding.food, encoding.floor, encoding.door):
        location = random.randrange(0, len(trap), 1)

    for i in range(subList):
        newTrap = copy.deepcopy(trap)
        newTrap[location] = subList[i]
        listMutants[i] = newTrap

    return listMutants

def getMutationalNeighborhoodStatistics(encoder, trap):
    mutatedTraps = generateMutated(encoder, trap)
    newCoherences, newLethalities = computeChanges(encoder, trap,mutatedTraps)
    return mutatedTraps,newCoherences,newLethalities

def getCoherenceAndLethality(encoder, trap):
    """Returns both the coherence and lethality of a trap"""
    coherence = fitnessFunctions.getCoherence(trap,encoder)
    lethality = fitnessFunctions.getLethality(trap,encoder)
    return coherence,lethality

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

def scatterplot(ogCoherence, ogLethality,lethalityArr, coherenceArr):
    #plt.scatter(lethalityArr, coherenceArr,color='r')
    newCoherence, newLethality, size = convertCohLetArrToStat(lethalityArr, coherenceArr)
    plt.scatter(newCoherence, newLethality,color='r', s = size, label = 'mutations')
    plt.scatter(ogCoherence, ogLethality,color='b',label='original')

    plt.xlabel("lethality")
    plt.ylabel("coherence")
    plt.show()
    plt.savefig('./plot1.png')

def scatterplot(ogCoherence, ogLethality,lethalityArr, coherenceArr):
    #plt.scatter(lethalityArr, coherenceArr,color='r')
    newCoherence, newLethality, size = convertCohLetArrToStat(lethalityArr, coherenceArr)
    plt.scatter(newCoherence, newLethality,color='r', s = size, label = 'mutations')
    plt.scatter(ogCoherence, ogLethality,color='b',label='original')

    plt.xlabel("lethality")
    plt.ylabel("coherence")
    plt.show()
    plt.savefig('./plot1.png')

"""
def randomMutation(encoding: Encoding, trap):
    '''Performs a mutation that is one of: substitution, deletion, insertion'''
    mutationType = "sub"
    rand1 = random.randrange(0, 1) * 3
    if rand1 > 1:
        mutationType = "del"
    elif rand1 > 2:
        mutationType = "ins"

    if mutationType == "sub":
        return library.mutationFunc(encoding, trap)

    else: 
        index = random.randrange(0, len(trap), 1)
        while index in (encoding.food, encoding.floor, encoding.door):
            index = random.randrange(0, len(trap), 1)

        if mutationType == "del":
            trap = trap[:index] + trap[index+1:]

        elif mutationType == "ins":
            trap = trap[:index] + [constants.CELL_ALPHABET[random.randrange(2, len(constants.CELL_ALPHABET), 1)]] + trap[index+1:]

    return np.array(trap)
"""

def getCoherantTraps(encoder):
    while True:
        trap = library.generateTrap()
        coherance, lethality = getCoherenceAndLethality(encoder, trap)
        if(coherance > 0):
            print(coherance)
            return trap
        
def getMultiMutatedTraps(encoder, numMutants):
    trap = getCoherantTraps(encoder)
    coherences, lethalities = [], []
    levels = []
    traps = [trap]
    i = 0
    for  trap in traps:
        coh, let = getCoherenceAndLethality(encoder, trap)
        coherences.append(coh)
        lethalities.append(let)
        levels.append(i)
        mutants = generateMutated(encoder, trap, library.mutationFunc, numMutants=5)
        newTraps.extend(mutants)
        i += 1
        traps = newTraps

    
    return coherences, lethalities,levels

def plotMultiMutatedTraps(coherences, lethalities,levels):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter(coherences, lethalities, levels)
    ax.set_xlabel("coherence")
    ax.set_ylabel("lethality")
    ax.set_zlabel("level")
    plt.show()

sampleDictionary = {(0,0):{(2,2):{(2,3), (2,4)}, (1,1): {(0,3), (0,4)}}}

def getPolar(dictionary, levels, branches):
    items = dictionary.items()
    newDict = {}
    newDict[(0,0)] = recursePolar(dictionary, levels, branches, 0, 2*math.pi)
    return newDict

def recursePolar (dict, levels, branches, startAngle, endAngle):
    if(levels == 0):
         return {}
    
    offset = (endAngle - startAngle)/branches
    i = 0
    newDict = {}
    for item in dict.keys():
        childDict = recursePolar(dict[item], levels-1, branches, i*offset, i+1*offset)
        newy = levels * math.sin(i* 0.5*offset)
        newx = levels * math.cos(i *0.5 *offset)
        newTuple = (newx, newy)
        newDict[newTuple] = childDict
        i += 1
    
    return newDict



        

    
def main():
    
    encoder = Encoding() 
    #trap = library.generateTrap()
    trap = getCoherantTraps(encoder)
    plotMultiMutatedTraps(*getMultiMutatedTraps(encoder, 5))

def getSingleMutation():
    encoder = Encoding() 
    #trap = library.generateTrap()
    trap = getCoherantTraps(encoder)
    
    #trap = generateSmallTrap()
    #getCoherenceAsndLethality
    #print(trap)
    ogCoherence, ogLethality = getCoherenceAndLethality(encoder, trap)
    mutants = generateMutated(encoder, trap, library.mutationFunc)
    #newCoherences, newLethalities = computeChanges(encoder, mutants)
    #scatterplot(ogCoherence, ogLethality,newCoherences, newLethalities)

    #ogCoherence, ogLethality = getCoherenceAndLethality(encoder, trap)
    #mutants = controlledSubstitution(1, encoder, trap, 12)
    #for i in range(2, 12):
    #    mutants.append(controlledSubstitution(i, encoder, trap, 12))
    newCoherences, newLethalities = computeChanges(encoder, mutants)
    scatterplot(ogCoherence, ogLethality,newCoherences, newLethalities)



if __name__ == "__main__":
    main()
