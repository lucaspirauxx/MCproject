import random

"""
a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,!
different possible chromosomes for Hello world, defines cell of 1) letter 2) position of letters in the alphabet 3)binary aspect of alphabets
"""

ALLELE_POOL = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ! "
TARGET_SOLUTION = "Hello World !"


class GeneticAlgorithm():
    class Individual():
        def __init__(self, chromosome):
            """
            :param chromosome: A string the same size as TARGET_SOLUTION
            """
            self.chromosome = chromosome
            self.fitness = self.get_fitness()

        def get_fitness(self):
            """
            :return: A numerical value being the fitness of the individual.
            """
            fitness = 0
            for i in range(len(self.chromosome)):
                if self.chromosome[i] == TARGET_SOLUTION[i]:
                    fitness += 1
            return fitness

        def get_chromosome(self):
            """
            :return: A string, the chromosome of the individual.
            """
            return self.chromosome

    def __init__(self, pop_size=500, pm=0.01, elitism=0.05):
        """
        :param pop_size: An integer defining the size of the population
        :param pm: A float defining the mutation rate
        :param elitism: A float definism the elitism rate
        """
        self.pop_size = pop_size
        self.allele_pool = ALLELE_POOL
        self.mutation_rate = pm
        self.elitism = elitism

    def generate_generation_zero(self):
        """
        :return: A list of size self.pop_size
                 containing randomly generated instances
                 of the class Individual
        """
        population = []
        for _ in range(self.pop_size):
            chromosome = ""
            for j in range(len(TARGET_SOLUTION)):
                chromosome += random.choice(self.allele_pool)
            population.append(self.Individual(chromosome))
        return population

    def mutation(self, individual):
        """
        :param chromosome: An instance of the class Individual
                           whose chromosome is to mutate
        :return:  An instance of the class Individual
                  whose chromosome has been mutated
        """
        mutated_chromosome = ""
        # Each gene has a probability pm to undergo a mutation
        for i in individual:
            if random.random() < self.mutation_rate:
                mutated_chromosome += random.choice(self.allele_pool)
            else:
                mutated_chromosome += i
        return self.Individual(mutated_chromosome)

    def selection(self, population):
        """
        :param population : A list of instances of the class Individuals
        :return: The mating pool constructed from
                the 50% fittest individuals in the population
        """
        population_sorted = sorted(population, key=lambda x: x.get_fitness(), reverse=True)
        n = int(len(population_sorted) * 0.5)  # 50% of population is kept
        return population_sorted[:n]

    def create_offspring(self, parent1, parent2):
        """
        :param parent1: An instance of the class Individual
        :param parent2: An instance of the class Individual
        :return: Two chromosomes/strings created by
                single-point crossover of the parents'
                chromosomes
        """

        crossover = random.randint(1, len(TARGET_SOLUTION) - 1)
        offspring1, offspring2 = (parent1.get_chromosome()[0:crossover] + parent2.get_chromosome()[crossover:]
                                  , parent2.get_chromosome()[0:crossover] + parent1.get_chromosome()[crossover:])

        return offspring1, offspring2

    def run_genetic_algorithm(self, seed,
                              tol=0.0,
                              display=True):
        """
        :param seed: An integer to set the random seed
        :param tol: A tolerance on the fitness function
        :param display: A boolean. If True, the fitness
                        of the best performing individual
                        is displayed at the end of each
                        generation
        """

        random.seed(seed)
        generation = 0

        # 1. Random generation of the initial population
        population = self.generate_generation_zero()

        # --- Modify the convergence criteria ---
        while population[0].get_fitness() - len(TARGET_SOLUTION) < tol:

            if display:
                print("Generation {} : {} \n".format(
                    generation,
                    population[0].get_chromosome()))

            # 2. Creation of the mating pool
            mating_pool = self.selection(population)

            # 3. Apply the elistist strategy
            n = int(len(population) * self.elitism)  # Nombre d'élitistes
            new_population = mating_pool[:n]

            # 4. Continuing the breeding process until
            # the population is entirely renewed
            while len(new_population) < self.pop_size:
                # 4.1 Select the parent in the mating pool
                parent1 = random.choice(mating_pool)
                parent2 = random.choice(mating_pool)

                # 4.2 Make them reproduce
                enfant1, enfant2 = self.create_offspring(parent1, parent2)

                # 4.4 Append the new solutions to the new population
                new_population += [self.mutation(enfant1), self.mutation(enfant2)]

            # The (sorted) new population replace the previous one.
            population = sorted(new_population, key=lambda x: x.get_fitness(), reverse=True)
            generation += 1

        if display:
            print("Generation {} : {}, fitness = {} \n".format(
                generation,
                population[0].get_chromosome(),
                population[0].get_fitness()))

        return generation, population[0].fitness, population[0].get_chromosome()


Generation, fitness, chromosome = GeneticAlgorithm().run_genetic_algorithm(21, 0.0, True)
