import numpy as np
import matplotlib.pyplot as plt
import pdb
import functions
from functions import *



##
n_initial_pop = 200
target = 'To be or Not to be! that is the problem'
numeric_target = str_to_number(target)
len_target = len(target)
min_ascii = 32
max_ascii = 126
##

child_ratio = 1
n_children = int(child_ratio * n_initial_pop)
cross_over_ratio = 0.60
mutation_ratio = 0.25
mutation_rate = 0
random_direct_transfer_ratio = 0.01
good_gene_transer = True   # until fulling child_pop

iteration = 1000


##

##




init_pop_list = []

#pdb.set_trace()

for elem in range(n_initial_pop):
    rand = np.random.randint(min_ascii, max_ascii, size=len_target)
    init_pop_list.append([rand, fitness(rand, numeric_target),-1])



#pdb.set_trace()

init_pop_list = np.array(init_pop_list)
init_pop_list = init_pop_list[init_pop_list[:,1].argsort()][::-1]
normal = init_pop_list[:,1].copy() / np.sum(init_pop_list[:,1])
init_pop_list[:,2] = functions.accumulator(normal)


#pdb.set_trace()

best_fit_score = []
init_pop_list_gold_repo = init_pop_list.copy()



for iter_ in range(iteration):

    #pdb.set_trace()



    children_list = []

    random_transfer(random_direct_transfer_ratio, n_children, init_pop_list, children_list)
    cross_over(cross_over_ratio, init_pop_list, children_list, n_children, numeric_target)
    mutation(n_children, mutation_ratio, mutation_rate, init_pop_list, children_list, numeric_target, min_ascii, max_ascii)
    good_gen_pass(n_children, children_list, init_pop_list_gold_repo)






    children_list = np.array(children_list)
    #shuff = np.random.randint(len(children_list), size = len(children_list))
    #children_list = children_list[shuff]


    for elem in range(len(children_list)):
        children_list[elem][1] = fitness(children_list[elem][0], numeric_target)


    #pdb.set_trace()



    children_list = children_list[children_list[:,1].argsort()][::-1]
    normal = children_list[:,1].copy() / np.sum(children_list[:,1])
    children_list[:,2] = functions.accumulator(normal)
    #
    init_pop_list = children_list.copy()
    init_pop_list_gold_repo = init_pop_list.copy()







    best_fit_score.append(init_pop_list[0])



    if not iter_%1:
        #pdb.set_trace()
        result = back_to_str(best_fit_score[-1][0])
        print(result)
        if result == target:
            print("Done!")
            break



best_fit_score = np.array(best_fit_score)
plt.plot(best_fit_score[:,1])
plt.show()








##
