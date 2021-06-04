import numpy as np
from geneticAlgorithm.utils import createTrap
import simulation as sim
from legacy.designedTraps import *
import geneticAlgorithm.analytical as analytical
import time

"""Runs a benchmark comparison between the old functional fitness and the new functional fitness"""

fitnessDic = {}
def functionalFitness(configuration, numSimulations = 5000, printStatistics = False):
    """
    Assigns a fitness based on the function of the given configuration.
    To do so, we run simulations to get a confidence interval on whether the gopher dies or not 
    or compute the the given configuration's probability of killing a gopher
    """

    # Convert list to string to reference in dictionary
    strEncoding = np.array2string(configuration)

    if strEncoding in fitnessDic:
        return fitnessDic[strEncoding]

    # NOTE: Default probability of entering is 0.8 (found in magicVariables.py)
    # Simulate the trap running numSimulations times
    numberAlive = 0
    hungerLevels = 5
    for hunger in range(hungerLevels):
        for _ in range(int(numSimulations / hungerLevels)):
            numberAlive += int(sim.simulateTrap(createTrap(configuration), False, hunger = hunger / 5)[3])

    # Calculate statistics
    proportion = 1 - numberAlive / numSimulations

    fitnessDic[strEncoding] = proportion

    return proportion

with open('performance.txt', 'w+') as out:
    fileLines = []
    for i, trap in enumerate(traps):
        startTime = time.time()
        analyticalSol = analytical.trapLethality(trap)
        midTime = time.time()
        simulatedSol = functionalFitness(np.array(trap))
        endTime = time.time()

        analyticTimeMs = (midTime - startTime) * 1e3
        simulatedTimeMs = (endTime - startTime) * 1e3
        fileLines.append('Trap {}:\n'.format(i))
        fileLines.append('Difference\t\t:\t{}\n'.format(round(analyticalSol - simulatedSol, 6)))
        fileLines.append('Time Analytical (ms)\t:\t{}\n'.format(round(analyticTimeMs, 3)))
        fileLines.append('Time Simulated  (ms)\t:\t{}\n'.format(round(simulatedTimeMs, 3)))
        fileLines.append('Factor Difference\t:\t{}\n\n'.format(round(simulatedTimeMs / analyticTimeMs, 3)))
    out.writelines(fileLines)