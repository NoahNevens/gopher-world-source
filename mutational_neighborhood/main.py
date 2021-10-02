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
    plt.scatter(newCoherence, newLethality,color='r', s = size)
    plt.scatter(ogCoherence, ogLethality,color='b')

    plt.xlabel("lethality")
    plt.ylabel("coherence")
    plt.show()


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
        return libssrary.mutationFunc(encoding, trap)

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


"""Copied from library.generate trap. Instead of a 3 by 4, trap trying to make a 3 by 3 trap"""
def generateTrap(encoder: Encoding = Encoding()):
    member = []
    for i in range(9):
        cellCode = random.randrange(2, len(constants.CELL_ALPHABET), 1)
        
        # Ensuring the board is valid
        if i == encoder.food:
            cellCode = 1 # Food
        elif i == encoder.floor:
            cellCode = 2 # Floor
        elif i == encoder.door:
            cellCode = 0 # Door

        member.append(cellCode)
    return np.array(member)



def main():

    encoder = Encoding() 
    trap = library.generateTrap()
    #getCoherenceAndLethality
    #print(trap)
    #ogCoherence, ogLethality = getCoherenceAndLethality(encoder, trap)
    #mutants = generateMutated(encoder, trap, library.mutationFunc)
    #newCoherences, newLethalities = computeChanges(encoder, mutants)
    #scatterplot(ogCoherence, ogLethality,newCoherences, newLethalities)
    print(controlledSubstitution(3, encoder, trap, 3))



if __name__ == "__main__":
    main()
