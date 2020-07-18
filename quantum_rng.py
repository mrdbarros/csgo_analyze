import random
import numpy as np


class ParticleHiddenInfo():
    def __init__(self,plan):
        self.plan=plan
    def get_align(self,axis_index):
        return self.plan[axis_index]

class ParticleRNG():
    def __init__(self,direction):
        self.current_state=random.getstate()
        self.direction=direction
    def get_align(self,axis):
        random.setstate(self.current_state)
        angle=random.uniform(0.,2*np.pi)
        angle =angle +np.pi*self.direction
        align = np.sign(np.cos(angle-axis))
        self.current_state = random.getstate()
        return align




def createRNGParticles(seed_input):
    #random.seed(a=seed_input)
    return ParticleRNG(0),ParticleRNG(1)

def createHiddenInfoParticles(detectors):
    plan_up = np.empty(len(detectors),dtype=np.int)
    plan_down = np.empty(len(detectors), dtype=np.int)

    for i in range(len(plan_down)):
        choice = random.choice([-1,1])
        plan_down[i]=choice
        plan_up[i] = -choice

    return ParticleHiddenInfo(plan_down),ParticleHiddenInfo(plan_up)

detectors = [np.pi/2.,7./6.*np.pi,11./6.*np.pi]

experiment_a = random.choices(range(3),k=1000)
experiment_b = random.choices(range(3),k=1000)



count_equal_rng = 0
count_equal_hidden_info = 0
for i in range(len(experiment_a)):
    seed = i+42
    particle_down_rng, particle_up_rng = createRNGParticles(seed)
    particle_down_hidden_info, particle_up_hidden_info = createHiddenInfoParticles(detectors)
    if particle_down_rng.get_align(detectors[experiment_a[i]]) == particle_up_rng.get_align(detectors[experiment_b[i]]):
        count_equal_rng +=1

    if particle_down_hidden_info.get_align(experiment_a[i]) == particle_up_hidden_info.get_align(experiment_b[i]):
        count_equal_hidden_info +=1

print("rng:",1-count_equal_rng/len(experiment_a),"hiddeninfo:",1-count_equal_hidden_info/len(experiment_a),"bell:",5/9)
