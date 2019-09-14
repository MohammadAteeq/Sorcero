import math
from random import Random
from time import time

import inspyred
from inspyred import ec


class SumarizationProblem:

    def __init__(self, number_of_sentences, coverage_matrix, sentences_similarity, ratio, objectives=1):
        self.dimensions = int(ratio * number_of_sentences)
        self.objectives = objectives
        self.bounder = None
        self.maximize = True
        self.coverage_matrix = coverage_matrix
        self.sentences_similarity = sentences_similarity
        self.ratio = ratio
        self.number_of_sentences = number_of_sentences
        self.bounder = ec.DiscreteBounder([i for i in range(number_of_sentences)])

    def generator(self, random, args):
        result = [0] * self.dimensions

        choice = [x for x in range(self.number_of_sentences)]
        for i in range(self.dimensions):
            value = random.choice(choice)
            choice.remove(value)
            result[i] = value
        return result

    def evaluator(self, candidates, args):
        fitness = []

        for c in candidates:
            f1 = 0
            sentences = []
            redundancy = 0
            for i in range(len(c)):
                f1 += self.coverage_matrix[c[i]]

                for sentence in sentences:
                    if sentences.__contains__(c[i]):
                        f1 = -math.inf
                        break
                    redundancy += self.sentences_similarity[c[i]][sentence]
                sentences.append(c[i])

            f1 -= (1 / len(c)) * redundancy
            fitness.append(f1)
        return fitness


class SummarizationOptimizer:

    def start_optimization(self, number_of_sentences, coverage_matrix, sentences_similarity, ratio):
        prng = Random()
        prng.seed(time())

        problem = SumarizationProblem(number_of_sentences, coverage_matrix, sentences_similarity, ratio, 2)
        ea = inspyred.ec.EvolutionaryComputation(prng)
        ea.variator = [inspyred.ec.variators.uniform_crossover,
                       inspyred.ec.variators.gaussian_mutation]
        ea.terminator = inspyred.ec.terminators.generation_termination
        ea.replacer = inspyred.ec.replacers.generational_replacement

        final_pop = ea.evolve(generator=problem.generator,
                              evaluator=problem.evaluator,

                              pop_size=100,
                              maximize=problem.maximize,
                              bounder=problem.bounder,
                              max_generations=20000)
        result = max(ea.population)
        return sorted(result.candidate)
