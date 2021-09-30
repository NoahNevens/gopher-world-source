#!/usr/bin/env python3
import geneticAlgorithm.library as library
import geneticAlgorithm.fitnessFunctions as fitnessFunctions
from classes.Encoding import Encoding
from classes.Trap import sampleRandomBoards
from classes.Trap import Trap
import random
import geneticAlgorithm.constants as constants
import numpy as np

def generateMutated(encoder, trap, mutationFunction = library.mutationFunc, numMutants = 10):
    """Generates a list of possible mutated traps produced from mutating the same trap"""
    listMutants = [None for _ in range(numMutants)]
    for i in range(numMutants):
        trapArr = encoder.encode(trap)
        listMutants[i] = mutationFunction(encoder, trapArr)
    return listMutants


def computeChanges(encoder,trap,mutatedTraps):
    """Compares the coherence and lethality of the mutated traps to the original trap"""
    newCoherences = [0 for _ in range(len(mutatedTraps))]
    newLethalities = [0 for _ in range(len(mutatedTraps))]

    for i in range(len(mutatedTraps)):
        newCoherences[i], newLethalities[i] = getCoherenceAndLethality(encoder, mutatedTraps[i])

    return newCoherences, newLethalities

def controlledSubstitution(location, encoding: Encoding, trap):
    """Perform a substitution mutation at a specified and valid location in the trap encoding"""
    if location in (encoding.food, encoding.floor, encoding.door):
        raise ValueError("Location must be part of trap encoding")
    trap[location] = constants.CELL_ALPHABET[random.randrange(2, len(constants.CELL_ALPHABET), 1)]

def getMutationalNeighborhoodStatistics(encoder, trap):
    mutatedTraps = generateMutated(encoder, trap)
    newCoherences, newLethalities = computeChanges(encoder, trap,mutatedTraps)
    return mutatedTraps,newCoherences,newLethalities

def getCoherenceAndLethality(encoder, trap):
    """Returns both the coherence and lethality of a trap"""
    encodedTrap = Encoding.encode(trap.randomBoard())
    coherence = fitnessFunctions.getCoherence(encodedTrap,encoder)
    lethality = fitnessFunctions.getLethalty(encodedTrap,encoder)
    return coherence,lethality

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
def main():

    #trap = library.generateTrap()
    trap = Trap(3,4,True)
    encoder = Encoding()
    #mutants = generateMutated(encoder, trap, library.mutationFunc, 10)
    print(getMutationalNeighborhoodStatistics(encoder, trap))


if __name__ == "__main__":
    main()
