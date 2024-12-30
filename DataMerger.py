

import numpy as np
import sys

print ('argument list', sys.argv)
index = int(sys.argv[1])
sex_asex_reproduction_ratio=index/10.0


name = 'WormsT100'
N=200  # population size
loci=6   # number of loci
dominance = 0  #0 for recessive, 1 for dominant, in between possible

epistasis=0

sub_name=name+'N'+str(N)+'L'+str(loci)+'E'+str(epistasis)+'X'+str(sex_asex_reproduction_ratio)+'D'+str(dominance)


R=10
all_concentrations = []
all_dilutions = []
all_dynamics_m=[]
all_dynamics_h=[]
all_time=[]
all_frequency=[]

for i in range (R):
    print (i)
    this_name='DataArticle/'+sub_name+'_i'+str(i)

    frequency=np.load(this_name+'frequency.npy')
    dynamics_h=np.load(this_name+'dynamics_h.npy')
    dynamics_m=np.load(this_name+'dynamics_m.npy')
    concentration=np.load(this_name+'concentration.npy')
    dilution=np.load(this_name+'dilution.npy')
    time=np.load(this_name+'time.npy')

    
    all_concentrations.append(concentration) 
    all_dilutions.append(dilution) 
    all_dynamics_m.append(dynamics_m) 
    all_dynamics_h.append(dynamics_h) 
    all_time.append(time) 
    all_frequency.append(frequency) 
all_concentrations = np.array(all_concentrations)
all_dilutions = np.array(all_dilutions)
all_dynamics_m = np.array(all_dynamics_m)
all_dynamics_h = np.array(all_dynamics_h)
all_time = np.array(all_time)
all_frequency = np.array(all_frequency)

np.save('DataArticle/CombinedData/'+sub_name+'all_frequency.npy', all_frequency)
np.save('DataArticle/CombinedData/'+sub_name+'all_dilution.npy', all_dilutions)
np.save('DataArticle/CombinedData/'+sub_name+'all_time.npy', all_time)
np.save('DataArticle/CombinedData/'+sub_name+'all_concentration.npy', all_concentrations)
np.save('DataArticle/CombinedData/'+sub_name+'all_dynamics_h.npy',all_dynamics_h)
np.save('DataArticle/CombinedData/'+sub_name+'all_dynamics_m.npy',all_dynamics_m)

