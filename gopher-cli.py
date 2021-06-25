#!/usr/bin/env python3
import argparse
import geneticAlgorithm.constants as constants
import geneticAlgorithm.fitnessFunctions as functions 
import geneticAlgorithm.experiment as geneticExperiment
from geneticAlgorithm.main import geneticAlgorithm
import geneticAlgorithm.utils as util
import legacy.experiment as experiment

parser = argparse.ArgumentParser(description="Commands to run the experiment")
subparsers = parser.add_subparsers(help='sub-command help', dest='command')

legacy = subparsers.add_parser('legacy', help='runs experiment or simulates legacy code (gopher\'s gambit)')
legacyParsers = legacy.add_subparsers(help='legacy parsers', dest='legacy')

# legacyParser.runExperiment flags
legacyExperimentParser = legacyParsers.add_parser('runExperiment', help='runs experiment')
legacyExperimentParser.add_argument('output', help='the output file name')
legacyExperimentParser.add_argument('inputToVary', help='independent variable to vary (probReal, nTrapsWithoutFood, maxProjectileStrength, defaultProbEnter)')
legacyExperimentParser.add_argument('numSimulations', help='number of simluations to run per param value', type=int)

# legacyParser.simulate flags
simulateParser = legacyParsers.add_parser('simulate', help='simulates experiment')
simulateParser.add_argument('--intention', '-i', help='turns on intention gopher', action='store_true')
simulateParser.add_argument('--cautious', '-c', help='if gopher is cautious', action='store_true')
simulateParser.add_argument('--defaultProbEnter', '-d', help='probability of gopher entering trap (not for intention)', type=float, default=0.8)
simulateParser.add_argument('--probReal', '-p', help='percentage of traps that are designed as opposed to random', type=float, default=0.2)
simulateParser.add_argument('--nTrapsWithoutFood', '-n', help='the amount of traps a gopher can survive without entering (due to starvation)', type=int, default=4)
simulateParser.add_argument('--maxProjectileStrength', '-m', help='the maximum projectile strength (thickWire strength)', type=float, default=.45)

# genetic algorithm parser
geneticParser = subparsers.add_parser('genetic-algorithm', help='generates a trap using the genetic algorithm')
geneticSubparsers = geneticParser.add_subparsers(help='genetic algorithm subparsers', dest='genetic')

# generate trap flags
generateTrap = geneticSubparsers.add_parser('generate', help='generates a trap')
generateTrap.add_argument('function', help='a choice of {random, coherence, functional, combined}')
generateTrap.add_argument('--threshold', '-t', help='the threshold to use for termination in [0, 1]', type=float, default=0.8)
generateTrap.add_argument('--max-generations', '-g', help='the maximum number of iterations to run', type=int, default=10000)
generateTrap.add_argument('--no-logs', '-nl', help='turns off logs as generations increase', action='store_false')
generateTrap.add_argument('--export', '-e', help='whether or not to export data to file (changed with -o flag)',  action='store_true')
generateTrap.add_argument('--output-file', '-o', help='the output file to which we write', default='geneticAlgorithm.txt')
generateTrap.add_argument('--show', '-s', help='show output in browser', action='store_true')

# run experiment flags
geneticExperimentParser = geneticSubparsers.add_parser('runExperiment', help='runs an experiment')
geneticExperimentParser.add_argument('function', help='a choice of {random, coherence, functional, combined}')
geneticExperimentParser.add_argument('--threshold', '-t', help='the threshold to use for termination in [0, 1]', type=float, default=0.8)
geneticExperimentParser.add_argument('--max-generations', '-g', help='the maximum number of iterations to run', type=int, default=10000)
geneticExperimentParser.add_argument('--no-logs', '-nl', help='turns on logs for generations', action='store_false')
geneticExperimentParser.add_argument('--num-simulations', '-s', help='the number of simulations of the trap to run', type=int, default=10000)
geneticExperimentParser.add_argument('--no-print-stats', '-np', help='turn off statistic printing', action='store_false')

# run batch experiments flags
geneticExperimentParser = geneticSubparsers.add_parser('runBatchExperiments', help='runs an experiment')
geneticExperimentParser.add_argument('function', help='a choice of {random, coherence, functional, combined}')
geneticExperimentParser.add_argument('--num-experiments', '-e', help='number of experiments to run', type=int, default=10)
geneticExperimentParser.add_argument('--threshold', '-t', help='the threshold to use for termination in [0, 1]', type=float, default=0.8)
geneticExperimentParser.add_argument('--max-generations', '-g', help='the maximum number of iterations to run', type=int, default=10000)
geneticExperimentParser.add_argument('--show-logs', '-l', help='turns on logs for generations', action='store_true')
geneticExperimentParser.add_argument('--output-file', '-o', help='the output file to which we write')
geneticExperimentParser.add_argument('--num-simulations', '-s', help='the number of simulations of the trap to run', type=int, default=10000)
geneticExperimentParser.add_argument('--overwrite', '-w', help='overwrites the experiment csv file', action='store_true')

# simulate trap flags
simulateTrap = geneticSubparsers.add_parser('simulate', help='simulates a trap given an input string')
simulateTrap.add_argument('trap', help='the encoded trap as a string (surrounded by \'\'s)')
simulateTrap.add_argument('--hunger', help='set the hunger for the simulated gopher (0, 1)', type=float, default=0)
simulateTrap.add_argument('--intention', '-in', help='give the simulated gopher intention', action='store_true')
simulateTrap.add_argument('--no-animation', '-na', help='turns off animation', action='store_true')
simulateTrap.add_argument('--gopher-state', '-g', help='sets the gopher\'s state as \'[x, y, rotation, state]\'', default='[-1, -1, 0, 1]')
simulateTrap.add_argument('--frame', '-f', help='the frame of the grid to print', type=int, default=0)

# get fitness trap flags
fitnessParser = geneticSubparsers.add_parser('check-fitnesses', help='returns the fitness of the trap')
fitnessParser.add_argument('trap', help='the encoded trap as a string (surrounded by \'\'s)')

args = parser.parse_args()

if args.command == 'legacy' and args.legacy == 'runExperiment':
    experiment.runExperiment(args.output, args.inputToVary, args.numSimulations)

elif args.command == 'legacy' and args.legacy == 'simulate':
    trapInfo = experiment.simulate({
        "intention" : args.intention, #if gopher has intention
        "cautious" : args.cautious, # only used if intention, fakes a FSC test to confirm intention > cautiousness
        "defaultProbEnter" : args.defaultProbEnter, #probability of gopher entering trap (not for intention)
        "probReal" : args.probReal, #percentage of traps that are designed as opposed to random
        "nTrapsWithoutFood" : args.nTrapsWithoutFood, #the amount of traps a gopher can survive without entering (due to starvation)
        "maxProjectileStrength" : args.maxProjectileStrength, #thickWire strength
    })

    print(trapInfo[1])

elif args.command == 'genetic-algorithm' and args.genetic == 'simulate':
    gopherState = util.convertStringToEncoding(args.gopher_state)
    util.simulateTrapInBrowser(util.convertStringToEncoding(args.trap), args.hunger, args.intention, args.no_animation, gopherState, args.frame)

elif args.command == 'genetic-algorithm' and args.genetic == 'check-fitnesses':
        print('Coherence fitness:\t', round(functions.coherentFitness(util.convertStringToEncoding(args.trap)), 3))
        print('Functional fitness:\t', round(functions.functionalFitness(util.convertStringToEncoding(args.trap)), 3))
        print('Combined fitness:\t', round(functions.combinedFitness(util.convertStringToEncoding(args.trap)), 3))

elif args.command == 'genetic-algorithm':
    # Defining the fitness function
    fitnessFunc = lambda x : 0
    freqs = {}
    fof = {}
    if args.function == 'random':
        fitnessFunc = functions.randomFitness
        freqs = functions.randomFreqs
        fof = functions.randomFoF
    elif args.function == 'coherence':
        fitnessFunc = functions.coherentFitness
        freqs = functions.coherentFreqs
        fof = functions.coherentFoF
    elif args.function == 'functional':
        fitnessFunc = functions.functionalFitness
        freqs = functions.functionalFreqs
        fof = functions.functionalFoF
    elif args.function == 'combined':
        fitnessFunc = functions.combinedFitness
        freqs = functions.combinedFreqs
        fof = functions.combinedFoF
    else:
        raise Exception(args.function, ' is not a real fitness function value. Please try again')

    if args.genetic == 'generate':
        # Running the simulation
        bestTrap = []
        bestFitness = 0
        if args.export:
            bestTrap, bestFitness = util.exportGeneticOutput(
                args.output_file,
                constants.CELL_ALPHABET,
                fitnessFunc,
                args.threshold,
                args.max_iterations,
                args.no_logs,
            )
        else:
            finalPopulation, bestTrap, bestFitness = geneticAlgorithm(
                constants.CELL_ALPHABET,
                fitnessFunc,
                args.threshold,
                args.max_iterations,
                args.no_logs,
            )

        print('Trap (encoded):\t\t', bestTrap)
        print('Coherence fitness:\t', round(functions.coherentFitness(bestTrap), 3))
        print('Functional fitness:\t', round(functions.functionalFitness(bestTrap), 3))
        print('Combined fitness:\t', round(functions.combinedFitness(bestTrap), 3))

        print(fof)

        if args.show:
            util.simulateTrapInBrowser(bestTrap)
    
    elif args.genetic == 'runExperiment':
        trap, fitness, prop, stderr, ci, intention = geneticExperiment.runExperiment(
            fitnessFunc, 
            args.threshold,
            maxGenerations=args.max_iterations,
            showLogs=args.no_logs,
            numSimulations=args.num_simulations,
            printStatistics=False,
            keepFreqs=False
        )

        print('Trap (Encoded):\t ', trap)
        print('Fitness\t:\t ', fitness)
        print('Proportion Dead:', prop)
        print('Standard Error:\t', stderr)
        print('Conf. Interval:\t', ci)
        print('Intention?:\t', 'Yes' if intention else 'No')

    elif args.genetic == 'runBatchExperiments':
        geneticExperiment.runBatchExperiments(
            numExperiments=args.num_experiments,
            fitnessFunction=fitnessFunc,
            threshold=args.threshold,
            numSimulations=args.num_simulations,
            maxGenerations=args.max_iterations,
            showLogs=args.show_logs,
            experimentFile=args.output_file,
            overwrite=args.overwrite
        )
