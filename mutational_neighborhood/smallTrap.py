from classes.Encoding import Encoding
import random
import numpy as np
import geneticAlgorithm.constants as constants
from mutatedTraps import getCoherenceAndLethality



"""Copied from library.generate trap. Instead of a 3 by 4, trap trying to make a 3 by 3 trap"""
def generateSmallTrap(encoder: Encoding = Encoding()):
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
    trap = generateSmallTrap()
    print(trap)
    getCoherenceAndLethality(encoder, trap)
    #ogCoherence, ogLethality = getCoherenceAndLethality(encoder, trap)
    #mutants = generateMutated(encoder, trap, library.mutationFunc)
    #newCoherences, newLethalities = computeChanges(encoder, mutants)
    #scatterplot(ogCoherence, ogLethality,newCoherences, newLethalities)

if __name__ == "__main__":
    main()