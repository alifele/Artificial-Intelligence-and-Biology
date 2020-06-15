import numpy as np




def score_based_selector(rand, scores_accum):
    return np.argmax(scores_accum>rand)


def accumulator(x):
    x_accum = x.copy()
    for i in range(len(x)):
        x_accum[i] = np.sum(x[:i+1])

    return x_accum




##
def cross_over(cross_over_ratio, init_pop_list, children_list, n_children, numeric_target):
    for i in range(int(n_children*cross_over_ratio)//2):
        rand = np.random.random(2)
        selected_index_1 = score_based_selector(rand[0], init_pop_list[:,2])
        selected_index_2 = score_based_selector(rand[1], init_pop_list[:,2])
        selected1 = init_pop_list[selected_index_1][0].copy()
        selected2 = init_pop_list[selected_index_2][0].copy()
        rand_cross = np.random.randint(len(selected1))
        temp1 = selected1[:rand_cross].copy()
        selected1[:rand_cross] = selected2[:rand_cross]
        selected2[:rand_cross] = temp1
        temp2 = selected1[rand_cross:].copy()
        selected1[rand_cross:] = selected2[rand_cross:]
        selected2[rand_cross:] = temp2

        #selected1[1] = fitness(selected1[0], numeric_target)
        #selected2[1] = fitness(selected2[0], numeric_target)

        children_list.append([selected1,-1,-1])
        children_list.append([selected2,-1,-1])





def mutation(n_children,mutation_ratio, mutation_rate, init_pop_list, children_list, numeric_target, min_ascii, max_ascii):
    length = len(init_pop_list[0][0])
    for i in range(int(n_children * mutation_ratio)):
        n_mutation = int(mutation_rate * length ) + 1

        indeces = np.random.randint(length, size = n_mutation)
        rand = np.random.random()
        selected_index = score_based_selector(rand, init_pop_list[:,2])
        selected = init_pop_list[selected_index][0].copy()
        mut_rand = np.random.randint(min_ascii, max_ascii, size=n_mutation)
        selected[indeces] = mut_rand

        #selected[1] = fitness(selected[0], numeric_target)
        children_list.append([selected,-1,-1])




def good_gen_pass(n_children, children_list, init_pop_list):
    n = n_children - len(children_list)
    #a = init_pop_list[:,1]
    #ind = np.argpartition(a, -n)[-n:]
    #for i in ind:
    #    children_list.append(init_pop_list[i])
    for i in range(n):
        item = init_pop_list[i][0].copy()
        children_list.append([item,-1,-1])


def get_best_score(curr_list):
    n=1
    a = curr_list[:,1]
    ind = np.argpartition(a, -n)[-n:]

    return curr_list[ind]



def random_transfer(random_direct_transfer_ratio, n_children, init_pop_list, children_list):
    n = int(n_children * random_direct_transfer_ratio)
    for i in range(n):
        rand = np.random.random()
        selected_index = score_based_selector(rand, init_pop_list[:,2])
        selected = init_pop_list[selected_index][0].copy()
        children_list.append([selected,-1,-1])



def fitness(x, target):
    #return (1000/(np.sum((x-target)**2)**0.5 + 0.01))**2
    #return 50*np.exp(10000/np.sum((x-target)**2))**0.3
    score = np.sum(x==target)
    return np.exp(score)


def back_to_str(x):
    text = ''
    for elem in x:
        text += chr(elem)
    return text

def str_to_number(x):
    val_list = []
    for elem in x:
        val_list.append(ord(elem))
    return np.array(val_list)
