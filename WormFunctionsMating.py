#!/usr/bin/env python
# coding: utf-8

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import itertools


plt.rcParams['axes.linewidth'] = 0.2 #set the value globally
plt.rc('lines', linewidth=1)
plt.rcParams['xtick.major.size'] = 2
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.size'] = 1
plt.rcParams['xtick.minor.width'] = 1


#plt.rcParams['legend.fonstsize'] = 1
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14) 

plt.rcParams['font.size']=16
#plt.rcParams.keys()


# In[15]:


class Utility():
    #probably for calculation of the curves? need to campture some properties of the population but not all of them
    
    def __init__(self, loci, dominance, cost_vector,benefit_vector, maxconc=10, EC50_wt=1 ):
        self.loci=loci
        self.number_of_genotypes=3**loci
        self.dominance=dominance
        self.genotypes=self.generate_genotypes()
        self.generate_phenotypes()
        #print ('genotypes', self.genotypes)
        #print ('phenotypes', self.phenotypes)
        self.define_fitness_vectors(cost_vector, benefit_vector)
        self.maxconc=maxconc
        self.EC50_wt=EC50_wt
        
    def print_input_data(self, population, experiment, name=''):
        out_file = open(name+'_InputFiles.txt','w')
        out_file.write(
"""
INPUT FILES

POPULATION:
Population size:%s
Loci:%s
Dominance:%s
Cost vector: %s
Benefit vector: %s
Mutation_rate: %s

Initial_male_fraction: %s
Mean_selfing_offspring_num: %s
Mean_sexrep_offspring_num: %s
Sex_asex_reproduction_ratio: %s


EXPERIMENT
cycle length: %s
concentration_gradient: %s
cycle_number: %s

"""% (population.initial_pop_size, population.loci, population.dominance, population.cost_vector,population.benefit_vector,   population.mutation_rate, population.initial_male_fraction, population.mean_selfing_offspring_num, population.mean_sexrep_offspring_num, 
      population.sex_asex_reproduction_ratio, experiment.cycle_length, experiment.concentration_gradient, experiment.cycle_number ))
        out_file.close()
        #print ('Input file written:', name+'_InputFiles.txt')

    def generate_genotypes(self):
        ### generating all possible diploid genotypes (0-1-2 number of alleles) from the number of loci. 
        genotypes = np.empty([0, self.loci])
        for seq in itertools.product("012", repeat=self.loci):
            s = np.array(seq)
            s = list(map(int, s))
            genotypes = np.vstack([genotypes, s])
            
        #print (genotypes)    
        mut_num = np.sum(genotypes, 1)
        sorting_argument = np.argsort(mut_num)

        sorted_genotypes = genotypes[sorting_argument]
        #print (sorted_genotypes)
        return genotypes
    
    
    def generate_phenotypes(self):
        if self.dominance==1:
            #print ('all loci are dominant')
            self.phenotypes=np.ceil(self.genotypes/2.0)
            #print (self.phenotypes)
        elif self.dominance==0: 
            #print ('all loci are recessive')
            self.phenotypes=np.floor(self.genotypes/2.0)
            #print (self.phenotypes)
        else:
            #print ('all is intermediate')
            self.phenotypes=(self.genotypes/2.0)
            #print (self.phenotypes)

    
    def define_fitness_vectors(self, cost_vector, benefit_vector):
        if len(cost_vector)==self.loci:
            self.cost_vector=cost_vector
        elif len(cost_vector)==1:
            self.cost_vector=np.zeros(self.loci)+cost_vector
        else:
            print ('error in cost definition')

            
        if len(benefit_vector)==self.loci:
            self.benefit_vector=benefit_vector
        elif len(benefit_vector)==1:
            self.benefit_vector=np.zeros(self.loci)+benefit_vector
        else:
            print ('error in benefit definition')

        return ()
    
       #print ('this is fitness', self.fitness)


    
    
    def get_fitness(self, concentration):
        kappa=3
        #print ('wt ec50 is', self.EC50_wt)
        EC50=np.zeros(3**self.loci)
        max_fitness=np.zeros(3**self.loci)
        fitness=np.zeros(3**self.loci)
        reduction=np.zeros(3**self.loci)
        for i in range (3**self.loci):
            max_fitness[i]=np.prod(1-np.dot(self.phenotypes[i,:], self.cost_vector))
            EC50[i]=self.EC50_wt+np.sum(np.dot(self.phenotypes[i,:], self.benefit_vector))
            reduction[i]=1-((concentration/EC50[i])**kappa/(1+(concentration/EC50[i])**kappa))
            fitness[i]=max_fitness[i]*reduction[i]
        #if concentration ==0:
            
            #print ('ec50', EC50)
            #print ('fitness at 0:',max_fitness) 
        return fitness
    

            
            
    
    def plot_PD_curve(self, name):
        PD_curves=[]
        #print ('maxconc', self.maxconc) 
        for i in range (0,int(self.maxconc)):
            point=self.get_fitness(i)
            PD_curves.append(point)
        PD_curves=np.array(PD_curves)
        plt.plot(PD_curves)
        plt.ylabel('fitness')
        plt.xlabel('concentration')
        plt.savefig(name+'PD_curves.png',bbox_inches='tight')
        plt.show()
        
        plt.semilogy(PD_curves)
        plt.ylabel('fitness')
        plt.xlabel('concentration')
        plt.savefig(name+'PD_curves_log.png',bbox_inches='tight')
        plt.show()
        
        x=range(0,3**self.loci)
        plt.bar(x, PD_curves[0,:], alpha=0.5)
        plt.xlabel('genotypes')
        plt.ylabel('fitness')
        plt.savefig(name+'fitness0.png',bbox_inches='tight')
        plt.show()
        
        plt.bar(x, PD_curves[-1,:], alpha=0.5, color='red')
        plt.xlabel('genotypes')
        plt.ylabel('fitness')
        plt.savefig(name+'fitness_end.png',bbox_inches='tight')
        plt.show()
        #print ('start fitness:', PD_curves[0,:], 'final fitness',PD_curves[-1,:] )
            
        
    
        


# In[16]:





# In[17]:



class Experiment_stochastic():
    ### definition of an experiment
    def __init__(self, population, concentration_gradient, cycle_length=1, cycle_number=1, increase_determinant=1 ):
        self.population=population
        self.concentration_gradient=concentration_gradient
        self.cycle_length=cycle_length
        self.cycle_number=cycle_number
        self.increase_determinant=increase_determinant

        self.simulated_population_sizes=[]
        self.simulated_males=[]
        self.simulated_hermaphrodites=[]
        self.simulated_male_fraction=[]
        self.simulated_allele_frequencies=[]
        self.dilution_factor_record=[]
        self.concentration_record=[]

        self.simulated_population_sizes.append(population.population_sizes)
        self.simulated_males.append(population.males)
        self.simulated_hermaphrodites.append(population.hermaphrodites)
        self.simulated_male_fraction.append(population.male_fraction)
        self.dilution_factor_record.append(0)
        self.concentration_record.append(0)

        self.simulated_allele_frequencies.append(population.allele_frequencies)
        #self.time=np.arange(self.cycle_number+1)*self.cycle_length
 
    def plot_results(self, population, name=''):
        XX=name
        legend_labels = [str(num) for num in population.genotypes]
        
        #fig1, ax1 = plt.subplots(figsize=(3, 2))

        #plt.plot(self.generation_time, self.simulated_male_fraction, 'k')
        #plt.plot(self.generation_time, self.simulated_male_fraction,'dk')
        #plt.ylim([0,1])
        #plt.title('Fraction of males')
        #plt.xlabel('Generation')
        #plt.ylabel('fraction of males')
        #plt.savefig(XX+'fraction_males.png',bbox_inches='tight')
        #plt.show()
        #fig1, ax1 = plt.subplots(dpi=200, figsize=(7, 5))
        
        #plt.semilogy(self.generation_time,self.simulated_hermaphrodites, 'd')
        #plt.semilogy(self.generation_time,self.simulated_hermaphrodites, '--k')
        #plt.xlabel('time')
        #plt.title('hermaphrodites')
        #plt.legend(legend_labels)
        #plt.ylim([1,10000])
        #plt.ylabel('population size')
        #plt.savefig(XX+'hermaphrodites.png',bbox_inches='tight')
        
        #plt.show()

        #fig1, ax1 = plt.subplots(dpi=200, figsize=(7, 5))

        #plt.semilogy(self.generation_time,self.simulated_males, 'd')
        #plt.semilogy(self.generation_time,self.simulated_males, ':k')
 
        #plt.title('males')
        #plt.ylim([1,10000])
        #plt.xlabel('Generation')
        #plt.ylabel('population size')
        #plt.savefig(XX+'males.png',bbox_inches='tight')
        #plt.legend(legend_labels)
        #plt.show()
        #print (np.shape(self.simulated_population_sizes))
 
        fig1, ax1 = plt.subplots(dpi=200, figsize=(9, 7))
        ax1.set_prop_cycle('color',[plt.cm.jet(i) for i in np.linspace(0, 1, 3**3)])
        print ('gen time', self.generation_time)
        print ('sim sizes', self.simulated_population_sizes)

        plt.semilogy(self.generation_time,self.simulated_population_sizes,'d')
        plt.semilogy(self.generation_time,self.simulated_population_sizes, ':k')
 
        plt.title('whole population')
        plt.ylim([1,10000])
        plt.xlabel('Generation')
        plt.ylabel('population size')
        #plt.legend(legend_labels)
        plt.savefig(XX+'all.png',bbox_inches='tight')
        plt.show()
        
        #fig1, ax1 = plt.subplots(figsize=(3, 2))

        #plt.plot(self.generation_time,self.dilution_factor_record,'y')
        #plt.plot(self.generation_time,self.dilution_factor_record,'yd')
        #plt.xlabel('Generation')
        #plt.ylabel('dilution factor')
        #plt.savefig(XX+'dilution.png',bbox_inches='tight')
        #plt.show()
 

        fig1, ax1 = plt.subplots(dpi=200, figsize=(7, 5))

        x=range(0,3**population.loci)
        plt.bar(x, self.simulated_population_sizes[-1], alpha=0.5)
        plt.xlabel('genotype')
        plt.yscale('log')
        plt.ylim([1,10000])
        plt.ylabel('size')
        plt.savefig(XX+'final_pop.png',bbox_inches='tight')
        plt.show()
        
        #fig1, ax1 = plt.subplots(dpi=200, figsize=(7, 5))
        #x=range(0,3**population.loci)
        #plt.bar(x, self.simulated_population_sizes[-1], alpha=0.5)
        #plt.xlabel('genotype')
        #plt.yscale('log')
        #plt.ylim([1,10000])
        #plt.ylabel('size')
        #plt.savefig(XX+'final_pop.png',bbox_inches='tight')
        #plt.show()
        
        fig1, ax1 = plt.subplots(dpi=200, figsize=(7, 5))
        x=range(0,population.loci)
        plt.bar(x, self.simulated_allele_frequencies[-1], alpha=0.5)
        plt.xlabel('locus')
        #plt.yscale('log')
        plt.ylim([0,1])
        plt.ylabel('mutation frequency')
        plt.savefig(XX+'final_frequency.png',bbox_inches='tight')
        plt.show()
        print (self.simulated_allele_frequencies[-1])

        fig1, ax1 = plt.subplots(dpi=200, figsize=(7, 5))
        
        plt.plot(self.simulated_allele_frequencies)
        plt.ylabel('mutation frequency')
        plt.xlabel('time')
        plt.savefig(XX+'frequency.png',bbox_inches='tight')
        plt.show()
        
    def save_output(self, population, name=''):
        np.save(name+'frequency', self.simulated_allele_frequencies)
        np.save(name+'dilution', self.dilution_factor_record)
        np.save(name+'time', self.generation_time)
        np.save(name+'concentration', self.concentration_record)
        np.save(name+'dynamics_h',self.simulated_hermaphrodites)
        np.save(name+'dynamics_m',self.simulated_males)

        
        #################  START STOCHASTICITY ###############
  
    def dilute_population(self, population, dilution_factor):
        #print ('dilution in', dilution_factor)
        probabilities_hermaphrodites=population.hermaphrodites/np.sum(population.hermaphrodites) # this was changed from jacqueline's experiment, check if it changed anything!!! and whether it makes sense. 
        probabilities_males=population.males/np.sum(population.males)
        male_fraction=np.sum(population.males)/np.sum(population.males+population.hermaphrodites)
        #print ('probabilities', probabilities_hermaphrodites, probabilities_males)
        if np.sum(probabilities_males)>0:
            population.males=np.random.multinomial(population.initial_pop_size*(male_fraction), probabilities_males)
        else: 
            population.males=np.zeros([population.number_of_genotypes])
            #print ('males got extinct')
        population.hermaphrodites=np.random.multinomial(population.initial_pop_size*(1-male_fraction), probabilities_hermaphrodites)
        population.population_sizes=population.hermaphrodites+population.males
    
    def run_neutral_evolution(self, population):
        current_concentration=0
        for i in range (self.cycle_number):
            #print ('running cycle', (i))

            self.run_one_stochastic_cycle(population, current_concentration)
            #print ('pop size before dilution', population.population_sizes)
            
            dilution_factor=np.sum(population.population_sizes)/population.initial_pop_size
            if dilution_factor<=1:
                dilution_factor=1
                print ('population declining, no dilution' )
            self.dilute_population(population, dilution_factor)
            
            #print ('pop size after dilution', population.population_sizes)
            #print ('males', population.males)
            #print ('population.hermaphrodites', population.hermaphrodites)
            self.dilution_factor_record.append(dilution_factor)
            self.simulated_population_sizes.append(population.population_sizes)

            #print ('inside neutral evol')
            self.simulated_males.append(population.males)
            self.simulated_hermaphrodites.append(population.hermaphrodites)
            self.simulated_male_fraction.append(population.male_fraction)
            self.simulated_allele_frequencies.append(population.allele_frequencies)
           
    def run_adaptation_experiment(self, population):
        
        current_concentration=0
        for i in range (self.cycle_number):
            
            ######### concentration #########
            current_concentration=self.concentration_gradient[i]
            self.concentration_record.append(current_concentration)

            #print ('we are starting cycle', i,'at concentration', current_concentration)
            #print ('current situation:H, M, Sum, Freq', population.hermaphrodites, population.males, population.population_sizes, population.allele_frequencies)

            #self.run_one_deterministic_cycle(population, current_concentration)
            #print ('BEGINNING of cycle', i, 'simulated frequencies', self.simulated_allele_frequencies)
            self.run_one_stochastic_cycle(population, current_concentration)
           

            ######### dilution #########


            #print ('pop size before dilution', population.population_sizes)
            dilution_factor=np.sum(population.population_sizes)/population.initial_pop_size
            #print ('offspringSize', np.sum(population.population_sizes))
            #print ('dilution factor',dilution_factor)
            if dilution_factor<=1:
                dilution_factor=1
                print ('population declining, no dilution' )
            self.dilute_population(population, dilution_factor)
            #print ('pop size after dilution', population.population_sizes)
            self.dilution_factor_record.append(dilution_factor)
                        
            self.simulated_population_sizes.append(population.population_sizes)
            self.simulated_males.append(population.males)
            self.simulated_hermaphrodites.append(population.hermaphrodites)
            self.simulated_male_fraction.append(population.male_fraction)
            self.simulated_allele_frequencies.append(population.allele_frequencies)
            #print ('end of cycle', i, 'simulated frequencies', self.simulated_allele_frequencies)
            #print ('end of cycle', i, 'simulated h', self.simulated_hermaphrodites)
        
        
        self.generation_time=np.arange(self.cycle_number)
        print ('cn', self.cycle_number)

          
            
            ######### calculated additional properties #########
            
             
    
    
    def run_Jacquelines_experiment(self, population):
        
        cycle_counter=0
        conc_increase_counter=0
        
        while cycle_counter<500:  # should be 40 in normal simulations
            
                
            ######### concentration #########
            if conc_increase_counter>=len (self.concentration_gradient):
                conc_increase_counter=conc_increase_counter-1
                print ('exceeding maximum concentration')
            current_concentration=self.concentration_gradient[conc_increase_counter]
            self.concentration_record.append(current_concentration)

            #print ('we are in cycle', cycle_counter,'at concentration', current_concentration)
            
            #self.run_one_deterministic_cycle(population, current_concentration)
            self.run_one_stochastic_cycle(population, current_concentration)
           

            ######### dilution #########

            dilution_factor=np.sum(population.population_sizes)/population.initial_pop_size
            #print ('offspringSize', np.sum(population.population_sizes))
            #print ('dilution factor',dilution_factor)
            
            
            if dilution_factor>self.increase_determinant:
                #print ('population adapted, increasing concentration, diluting', dilution_factor)
                conc_increase_counter=conc_increase_counter+1
                #current_concentration=concentration_gradient[conc_increase_counter]
                #print ('concentration will be', current_concentration)
            elif dilution_factor<=1:
                dilution_factor=1
                #print ('population declining, no concentration increase or dilution' )
            
            #else:
            #    print ('population not large enough, no concentration increase' )
            self.dilute_population(population, dilution_factor)
            
            self.dilution_factor_record.append(dilution_factor)
            cycle_counter=cycle_counter+1
            
            ######### calculated additional properties #########
            
            
            
            ######## recording pop sizes

            self.simulated_population_sizes.append(population.population_sizes)  # this is wrong, it is offspring size
            self.simulated_males.append(population.males)
            self.simulated_hermaphrodites.append(population.hermaphrodites)
            self.simulated_male_fraction.append(population.male_fraction)
            self.simulated_allele_frequencies.append(population.allele_frequencies)
        #print ('finished simulation at cycle:', cycle_counter, 'with increase counter', conc_increase_counter, 'at conc', current_concentration)
        self.generation_time=np.arange(cycle_counter+1)#*self.cycle_length
        



       

    def run_one_stochastic_cycle(self, population, current_concentration):
        population.calculate_fitness(current_concentration)  # is this even important here?!
        
        #print ('beginning stoch cycle: H, M, sum, F', population.hermaphrodites, population.males, population.population_sizes, population.allele_frequencies)
            
        ############ SELFING
        population.define_mating_fractions()
        new_hermaphrodites=population.reproduce_by_selfing()
        population.hermaphrodites=new_hermaphrodites
        
        #print ('from selfing', np.sum(new_hermaphrodites))
        ############ SEXUAL REPRODUCTION
        new_males=np.zeros([population.number_of_genotypes])
        new_hermaphrodites=np.zeros([population.number_of_genotypes])
        
        
        if np.sum(population.males)>0: 
            if population.sex_asex_reproduction_ratio>0:
                if np.sum(population.mating)>0:
                    [new_hermaphrodites, new_males]=population.reproduce_sexually()
        
        population.hermaphrodites=population.hermaphrodites+new_hermaphrodites
        population.males=new_males
        #print ('from sex', np.sum(new_hermaphrodites))
        
        #print ('all ', np.sum(population.hermaphrodites))

        
        ############## MUTATIONS
        
        [mutants_males, mutants_hermaphrodites]=population.mutate()
        
        population.hermaphrodites=population.hermaphrodites+mutants_hermaphrodites
        population.males=population.males+mutants_males
        
        ## HERE CHECK THE SUM!
        
        population.hermaphrodites[population.hermaphrodites<0]=0
        population.males[population.males<0]=0
        
        
        ############## PUT EVERYTHING TOGETHER
        population.population_sizes=population.males+population.hermaphrodites
        population.male_fraction=np.sum(population.males)/np.sum(population.population_sizes)
        population.calculate_allele_frequency()

        #print ('end of stoch cycle: H, M, sum, F', population.hermaphrodites, population.males, population.population_sizes, population.allele_frequencies)
        

        
    
        #print ('end of cycle hemaphrodites', population.hermaphrodites)
        #print ('end of cycle males', population.males)
        

class Worm_population_stochastic:
    ### a simple class containing all necessary properties and functions describing a worm population
    def __init__(self, initial_pop_size, loci=1, cost_vector=[0], benefit_vector=[0], dominance=0, mutation_rate=0.001, initial_male_fraction=0.1, 
                 mean_selfing_offspring_num=10, mean_sexrep_offspring_num=100, sex_asex_reproduction_ratio=0.5):
        
        ### define parameters
        self.initial_pop_size=initial_pop_size
        self.loci=loci
        self.mutation_rate=mutation_rate
        self.initial_male_fraction=initial_male_fraction
        self.mean_selfing_offspring_num=mean_selfing_offspring_num
        self.mean_sexrep_offspring_num=mean_sexrep_offspring_num
        self.sex_asex_reproduction_ratio=sex_asex_reproduction_ratio
        self.dominance=dominance  # this specifies dominance for each locus 0 - recessive, 1- dominant
    
        
        ### determine variables, genotypes and dependencies
        self.number_of_genotypes=3**loci
        self.genotypes=self.generate_genotypes()
        
        self.offspring_flow_list=self.define_selfing_dependencies()
        self.mutation_flow_list=self.define_mutation_dependencies()
        #[self.cost_vector, self.benefit_vector]=
        self.generate_phenotypes()
        self.define_fitness_vectors(cost_vector, benefit_vector)
        

        
        ### this is the questionable part that may need to be edited. Enter initial pop sizes
        self.population_sizes=np.zeros(self.number_of_genotypes)
        self.population_sizes[0]=initial_pop_size
        #self.population_sizes[1]=initial_pop_size
        #self.population_sizes[2]=initial_pop_size
        
        #self.allele_frequencies=self.calculate_allele_frequency()

        self.calculate_allele_frequency()
        #print ('this was the initial calculation of frequencies')
        #print ('allele F', self.allele_frequencies)

        ### split population to males and hermaphrodites for all genotypes
        self.hermaphrodites=np.round(self.population_sizes*(1-self.initial_male_fraction))
        self.males=np.round(self.population_sizes*self.initial_male_fraction)
        #self.population_sizes[9]=initial_pop_size
        #self.population_sizes[15]=initial_pop_size
        self.male_fraction=self.initial_male_fraction
        #print ('init complete. Pops:H, M, sum', self.hermaphrodites, self.males, self.population_sizes)
    
    def calculate_allele_frequency(self):
        whole_size=np.sum(self.population_sizes)
        self.allele_frequencies=np.empty([self.loci])
        #print ('pop sizes in F calcu', self.population_sizes)

        for i in range(self.loci):
          
                   
            allele_f=np.sum((self.population_sizes[self.genotypes[:,i]==1]))+2*np.sum((self.population_sizes[self.genotypes[:,i]==2]))
            #allele_f=np.sum((self.population_sizes[[self.genotypes[:,i]==1]]))+2*np.sum((self.population_sizes[[self.genotypes[:,i]==2]]))
            ###not sure why there were double [[]]
            #print ('allele F intermediate', allele_f)

            allele_f=allele_f/(2*whole_size)
            #print ('locus', i, 'has frequency', allele_f)
            self.allele_frequencies[i]=allele_f
        #print ('allele F:',self.allele_frequencies)
    
    
    
    def generate_genotypes(self):
        ### generating all possible diploid genotypes (0-1-2 number of alleles) from the number of loci. 
        genotypes = np.empty([0, self.loci])
        for seq in itertools.product("012", repeat=self.loci):
            s = np.array(seq)
            s = list(map(int, s))
            genotypes = np.vstack([genotypes, s])

        return genotypes.astype(int)
    
    def generate_phenotypes(self):
        if self.dominance==1:
            #print ('all loci are dominant')
            self.phenotypes=np.ceil(self.genotypes/2.0)
            #print (self.phenotypes)
        elif self.dominance==0: 
            #print ('all loci are recessive')
            self.phenotypes=np.floor(self.genotypes/2.0)
            #print (self.phenotypes)
        else:
            #print ('all is intermediate')
            self.phenotypes=(self.genotypes/2.0)
            #print (self.phenotypes)
 
    def define_fitness_vectors(self, cost_vector, benefit_vector):
        if len(cost_vector)==self.loci:
            self.cost_vector=cost_vector
        elif len(cost_vector)==1:
            self.cost_vector=np.zeros(self.loci)+cost_vector
        else:
            print ('error in cost definition')

            
        if len(benefit_vector)==self.loci:
            self.benefit_vector=benefit_vector
        elif len(benefit_vector)==1:
            self.benefit_vector=np.zeros(self.loci)+benefit_vector
        else:
            print ('error in benefit definition')

        return ()
    
    def calculate_fitness(self, concentration):
        ## most of this could be taken away, but as we don't have that many cycles it does not matter
        kappa=3
        EC50=np.zeros(3**self.loci)
        max_fitness=np.zeros(3**self.loci)
        self.fitness=np.zeros(3**self.loci)
        reduction=np.zeros(3**self.loci)
        for i in range (3**self.loci):
            
            max_fitness[i]=np.prod(1-np.dot(self.phenotypes[i,:], self.cost_vector))
            EC50[i]=1+np.sum(np.dot(self.phenotypes[i,:], self.benefit_vector))
            reduction[i]=1-((concentration/EC50[i])**kappa/(1+(concentration/EC50[i])**kappa))
            self.fitness[i]=max_fitness[i]*reduction[i]
        #print ('this is fitness', self.fitness)
            

    def define_selfing_dependencies(self): 
        ### determines fractions of offspring genotypes for each genotype. Returns list of non-zero fractions
        ### genotype A (ancestral) index - genotype B (offspring) index - fraction  
        probability_table=np.array([[1,0,0],[0.25,0.5,0.25],[0,0,1]])
        offspring_flow_list=[]
        for i in range (self.number_of_genotypes):
            for j in range (self.number_of_genotypes):
                probability_i_j=1
                for k in range (self.loci):
                    probability_locus=probability_table[(self.genotypes[i][k]), (self.genotypes[j][k])]
                
                    probability_i_j=probability_i_j*probability_locus
                if probability_i_j>0:
                    offspring_flow_list.append([i,j,probability_i_j])
                    #print ('from', self.genotypes[i], 'to', self.genotypes[j], 'the probability is', probability_i_j)
        return (offspring_flow_list)

 
    def calculate_ratios_when_sex(self, genotype_female, genotype_male):
        p=np.zeros([3,self.loci]) #offspring probabilities
        for i in range(self.loci):
            if genotype_female[i]==0:
                if genotype_male[i]==0:
                    p[0,i]=1
                elif genotype_male[i]==1:
                    p[0,i]=0.5
                    p[1,i]=0.5
                elif genotype_male[i]==2:
                    p[1,i]=1
            elif genotype_female[i]==1:
                if genotype_male[i]==0:
                    p[0,i]=0.5
                    p[1,i]=0.5
                elif genotype_male[i]==1:
                    p[0,i]=0.25
                    p[1,i]=0.5
                    p[2,i]=0.25
                elif genotype_male[i]==2:
                    p[1,i]=0.5
                    p[2,i]=0.5
        
            elif genotype_female[i]==2:
                if genotype_male[i]==0:
                    p[1,i]=1
                elif genotype_male[i]==1:
                    p[1,i]=0.5
                    p[2,i]=0.5
                elif genotype_male[i]==2:
                    p[2,i]=1
        #print (p)            
        offspring_probabilities=np.zeros(3**self.loci)
        for i in range(3**self.loci):
            this_p=1
            for j in range (self.loci):
                x=self.genotypes[i,j]
                this_p=this_p*p[x,j]
            offspring_probabilities[i]=this_p
            #if this_p>0:
                #print ('from ',genotype_female, genotype_male, 'to', self.genotypes[i],'probability is',this_p)
        return (offspring_probabilities)

    def define_mutation_dependencies(self):
        ### determines mutation dependencies - returns a list of genotypes (indices) as pairs that are 1 mutation apart. 
        mutation_dependences=[]
        for i in range(3**self.loci):
            for j in range ((3**self.loci)):
                hamming_distance=sum(abs(c1 - c2) for c1, c2 in zip(self.genotypes[i], self.genotypes[j]))
                if hamming_distance==1:
                    mutation_dependences.append([i,j])
        return (mutation_dependences)

        
    ############################################## STOCHASTIC PART ######################################

    def define_mating_fractions(self):
        #print ('define mating fraction')
        #print (self.hermaphrodites)
        
        # this is situaion, where fraction of sexually reproducing hermaphrodites depends! on the number of males!
        
        #ff=min(self.sex_asex_reproduction_ratio*(np.sum(self.males)/np.sum((self.hermaphrodites+self.males))),1)
        #print (ff)
        #self.mating=self.hermaphrodites*ff
        # this is situaion, where fraction of sexually reproducing hermaphrodites is independent from the number of males
        self.mating=self.sex_asex_reproduction_ratio*self.hermaphrodites
        #print ('Mating fraction:', self.mating)
        for i in range(len(self.mating)):
            self.mating[i]= np.random.poisson(self.mating[i], 1)
        self.mating[self.mating>self.hermaphrodites]=self.hermaphrodites[self.mating>self.hermaphrodites]
        #print ('CORRECTED mating pop', self.mating)
        #print ('CORRECTED hermaphrodites', self.hermaphrodites)
        #print ('hermaphrodites', self.hermaphrodites)
        #print ('mating pop', self.mating)
        
        self.selfing=self.hermaphrodites-self.mating
        #print ('sefling pop', self.selfing)
        self.mating[self.selfing<0]=self.mating[self.selfing<0]+self.selfing[self.selfing<0]
        self.selfing[self.selfing<0]=0
        
          
    def calculate_offspring_number_in_cycle(self,parent_population, concentration):
        self.calculate_fitness(concentration)
        offspring_number=np.zeros([self.number_of_genotypes])
        helping_list=[]
        for trio in self.offspring_flow_list:
            i=trio[0]
            j=trio[1]
            k=trio[2]
            #print (trio)

            #print (self.fitness[i])
            #print ('pp', parent_population)

            #print ('parent pop', parent_population)
            helping_list.append([j,self.fitness[i]* parent_population[i]*self.mean_selfing_offspring_num*k])
        for pair in helping_list:
            #print (pair)
            offspring_number[pair[0]]=offspring_number[pair[0]]+pair[1]
        return (offspring_number)    

    
    def reproduce_by_selfing(self):
        ### calculate new population sizes of all genotypes depending on selfing dependencies and average number of offspring denerated by selfing
        new_pop_sizes=np.zeros(3**self.loci)
        new_hermaphrodites=np.zeros(3**self.loci)
        offspring_number=np.zeros(3**self.loci)

        # this calculates how many offspring each class has.         
        for i in range(self.number_of_genotypes):
            offspring_number[i]=(np.random.poisson(self.fitness[i]*self.mean_selfing_offspring_num*self.selfing[i],1))
        #print ('offspring number', offspring_number)

        this_help=np.array(self.offspring_flow_list)
        for i in range(self.number_of_genotypes):
            which_offspring_list=(this_help[this_help[:,0]==i,1])
            
            probabilities_list=((this_help[this_help[:,0]==i,2]))
            generated_offspring=np.random.multinomial(offspring_number[i],probabilities_list )
            for j in range (len(which_offspring_list)):
                new_pop_sizes[int(which_offspring_list[j])]= new_pop_sizes[int(which_offspring_list[j])]+generated_offspring[j]
            #print ('genotype', i, self.genotypes[i])
            #print ('WOL', which_offspring_list)
            #print ('PL', probabilities_list)
            #print ('off', generated_offspring)
            
        new_hermaphrodites= new_pop_sizes
        #print ('NH in RS', new_hermaphrodites)
        #print ('new_hermaphrodites', new_hermaphrodites)
        
        return (new_hermaphrodites)
    
    
    
    

    def mutate(self):
        
        ### THIS STUFF IS WRONG - DOES NOT ACCOUNT FOR VARIOUS PROBABILITIES OF MUTATIONS - FROM 00 TO 01 
        ##IS LARGER CHANCE THAN FROM 01 TO 02. OR THINK ABOUT IT. 
        ### calculates new population sizes based on the mutation rate, original population sizes and mutation dependencies

        ### for hermaphrodites:
        mutants_hermaphrodites=np.zeros(3**self.loci)
        for i in range(len(self.mutation_flow_list)):
            this_entry=self.mutation_flow_list[i]
            delta=np.random.poisson(self.hermaphrodites[this_entry[0]]*self.mutation_rate, 1)
            #print ('SH', self.hermaphrodites[this_entry[0]])
            #print ('delta', delta)
            mutants_hermaphrodites[this_entry[0]]=mutants_hermaphrodites[this_entry[0]]-delta
            mutants_hermaphrodites[this_entry[1]]=mutants_hermaphrodites[this_entry[1]]+delta
            #if mutants_hermaphrodites[this_entry[0]]<0:
            #    print ('correction made',mutants_hermaphrodites[this_entry[0]] )
            #    mutants_hermaphrodites[this_entry[1]]=mutants_hermaphrodites[this_entry[1]]+mutants_hermaphrodites[this_entry[0]]
            #    mutants_hermaphrodites[this_entry[0]]=0
        #print ('MH', mutants_hermaphrodites)
        ### for males:
        mutants_males=np.zeros(3**self.loci)
        for i in range(len(self.mutation_flow_list)):
            this_entry=self.mutation_flow_list[i]
            delta=np.random.poisson(self.males[this_entry[0]]*self.mutation_rate, 1)
            #print ('delta', delta)
            mutants_males[this_entry[0]]=mutants_males[this_entry[0]]-delta
            mutants_males[this_entry[1]]=mutants_males[this_entry[1]]+delta
            #if mutants_males[this_entry[0]]<0:
            #    mutants_males[this_entry[1]]=mutants_males[this_entry[1]]+mutants_males[this_entry[0]]
            #    mutants_males[this_entry[0]]=0
            
        return (mutants_males, mutants_hermaphrodites)

       
#        self.males = self.males+mutants_males
#        self.hermaphrodites = self.hermaphrodites+mutants_hermaphrodites
#        self.population_sizes=self.males+self.hermaphrodites

    def reproduce_sexually(self):
        new_pop_hermaphrodites=np.zeros([3**self.loci])
        new_pop_males=np.zeros([3**self.loci])
        mating_males=[]
        male_ratios=[]
        for i in range (3**self.loci):
            if self.males[i]>0:
                mating_males.append(i)
                male_ratios.append(self.males[i])
        mating_males=np.array(mating_males)        
        male_ratios=np.array(male_ratios)/sum(male_ratios)
        
        #print ('this is a list of all mating males', mating_males, 'at this ratios', male_ratios)
        

        for i in range (3**self.loci):  #go through all females
            if self.mating[i]>0:
                
                new_offspring=np.random.poisson(self.fitness[i]*self.mating[i]*self.mean_sexrep_offspring_num)
                #print ('females', i, ' are mating, having', new_offspring, 'offspring')
                split_offspring=np.random.multinomial(new_offspring,male_ratios )
                ####here i have to assign offspring to males
                #print (mating_males)
                for j in range(len(mating_males)): 
                    k=mating_males[j]
                    #print ('cycle j', j, 'mating male is', k)
                    
                    offspring_ratios=self.calculate_ratios_when_sex(self.genotypes[i], self.genotypes[k])
                    new_offspring=np.random.multinomial(split_offspring[j], offspring_ratios)
                    #print ('between ',i,'and', j, 'there are',new_offspring, 'offspring' )
                    new_pop_hermaphrodites=new_pop_hermaphrodites+0.5*new_offspring
                    new_pop_males=new_pop_males+0.5*new_offspring
                    #print ('parental',self.hermaphrodites[i],self.males[j] )
                    #print (new_pop_hermaphrodites)
        ####new_pop_hermaphrodites=np.multiply(self.fitness,new_pop_hermaphrodites)*self.mean_sexrep_offspring_num
        ####new_pop_males=np.multiply(self.fitness,new_pop_males)*self.mean_sexrep_offspring_num
        #print ('NH', new_pop_hermaphrodites)
        #print ('NM', new_pop_males)
        return (new_pop_hermaphrodites, new_pop_males)
  

