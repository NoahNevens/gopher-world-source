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

def getRobustness(encoder, trap):
    """Returns the robustness of a trap"""
    ogCoherence = fitnessFunctions.getCoherence(trap, encoder)
    ogLethality = fitnessFunctions.getLethality(trap, encoder)
    mutatedTraps = generateMutated(encoder, trap)
    newCoherences, newLethalities = computeChanges(encoder, trap, mutatedTraps)
    totalCoherenceChange = 0
    totalLethalityChange = 0

    for i in range(len(newCoherences)):
        totalCoherenceChange += abs(ogCoherence - newCoherences[i])
        totalLethalityChange += abs(ogLethality - newLethalities[i])
    
    avgCohChange = totalCoherenceChange / len(newCoherences)
    avgLetChange = totalLethalityChange / len(newLethalities)

    return (1 - avgCohChange), (1 - avgLetChange)


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

def getCoherentTraps(encoder):
    while True:
        trap = library.generateTrap()
        coherence, lethality = getCoherenceAndLethality(encoder, trap)
        if(coherence > 0):
            print(coherence)
            return trap
        
def getMultiMutatedTraps(encoder, numMutants):
    trap = getCoherentTraps(encoder)
    coherences, lethalities = [], []
    levels = []
    traps = [trap]
    newTraps = []
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

def getMultiMutatedDict(encoder, numMutants, levels):
    trap = getCoherentTraps(encoder)

    
    dictt = recurseMultiDict(trap, numMutants, levels, {}, encoder)
    return dictt

def recurseMultiDict(trap, numMutants, levels, dictM, encoder):
    if(levels == 0):
        return {}
    coh, let = getCoherenceAndLethality(encoder, trap)
    mutants = generateMutated(encoder, trap, library.mutationFunc, numMutants=numMutants)

    m = []
    l = {}
    i = 0
    for mutant in mutants:
        mutantMutants = {}
        if(levels != 1):
            mutantMutants = recurseMultiDict(mutant, numMutants, levels-1, {}, encoder)
        coh1, let1 = getCoherenceAndLethality(encoder, mutant)
        l[(coh1,let1)] = mutantMutants
        m += [(coh1, let1)]
    
    
    dictM[(coh, let)] = l
    if(levels == 1):
        dictM[coh, let] = m
    return dictM
    
    




def plotMultiMutatedTraps(coherences, lethalities,levels):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter(coherences, lethalities, levels)
    ax.set_xlabel("coherence")
    ax.set_ylabel("lethality")
    ax.set_zlabel("level")
    plt.show()

def plotMultiMutatedDict(X, Y,Values):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter(X, Y, Values)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Coherance or Lethality")
    plt.show()

    plt.savefig('./plot2.png')

sampleDictionary = {(0,0):{(2,2):{(2,3), (2,4)}, (1,1): {(0,3), (0,4)}}}


SCALE = 5


def getPolar(dictionary, levels, index):
    items = dictionary.items()
    newDict = {}
    X,Y,V = recursePolar(dictionary, levels, 0, 2*math.pi, index, levels)

    rootItem = dictionary.keys()
    val = 0
    for item in rootItem:
        val = item[index]
    X += [0]
    Y += [0]
    V += [val]

    return X,Y,V
    

def recursePolar (dict1, levels, startAngle, endAngle, index, totalLevels):
    if(levels == 0):
         return [],[],[]
    
    branches = len(dict1.keys())
    offset = (endAngle - startAngle)/branches
    i = 0
    newDict = {}
    newList = []
    XList = []
    YList = []
    Value = []
    for item in dict1.keys():
        X, Y, V = recursePolar(dict1[item], levels-1,  i*offset, i+1*offset, index, totalLevels)
        newy = SCALE*(totalLevels-levels+1) * math.sin(i* 0.5*offset)
        newx = SCALE*(totalLevels-levels+1) * math.cos(i *0.5 *offset)
        XList += [newx]
        YList += [newy]
        Value += [item[index]]
        if(levels > 1):
            XList += X
            YList += Y
            Value += V
        i += 1
    

    return XList, YList, Value



        

    

def getSingleMutation():
    encoder = Encoding() 
    #trap = library.generateTrap()
    trap = getCoherentTraps(encoder)
    
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


def main():    
    encoder = Encoding() 
    #trap = library.generateTrap()
    trap = getCoherentTraps(encoder)
    multimutatedtraps = getMultiMutatedDict(encoder, 3, 4)
    X,Y,V = getPolar(multimutatedtraps,4, 0)
    #scatterplot(0, 0,X, Y)
    plotMultiMutatedDict(X,Y,V)

    getSingleMutation()


    #plotMultiMutatedTraps(*getMultiMutatedTraps(encoder, 5))


if __name__ == "__main__":
    main()
