import random
import copy
from termcolor import colored
import networkx as nx
import matplotlib.pyplot as plt


class Dfsa:
    def __init__(self):
        self.states=set()   
        self.alphabet=set()
        self.transitions=set()
        self.accepting_states=set()
        self.start_state=None
    # setters / getters
    def set_states(self,states):
        self.states=set(states)
    def get_states(self):
      return self.states
    def set_start_state(self,start_state):
        self.start_state=start_state
    def get_start_state(self):
        return self.start_state
    def set_accepting_states(self,accepting_states):
        self.accepting_states=set(accepting_states)
    def get_accepting_states(self):
        return self.accepting_states
    def set_alphabet(self,alphabet):
        self.alphabet=set(alphabet)
    def get_alphabet(self):
        return self.alphabet
    
    def set_transitions(self,transitions):
        self.transitions.update(transitions)
    def get_transitions(self):
        return self.transitions
    # add element to set
    def add_transition(self,transition):
        self.transitions.add(transition)
    def add_state(self,state):
        self.states.add(state)
    def add_accepting_state(self,accepting_state):
        self.accepting_states.add(accepting_state)  
    #  remove element from dfsa  
    def remove_state(self,state):
        self.states.remove(state)
        if state in self.accepting_states:
            self.accepting_states.remove(state)
        for transition in list(self.transitions):
            if transition[0][0]==state or transition[1]==state:
                self.transitions.remove(transition)
        # set any state as next state
        if state == self.start_state:
            self.start_state==list(self.states)[0]
  
    # a function to print all the attribures of the Dfsa
    def display_dfsa(self):
        print(colored('Set of states: ','green'),self.get_states())
        print(colored('Set of accepting states: ','green'),self.get_accepting_states())
        print(colored('Starting state: ','green'),self.get_start_state())
        print(colored('Transition function: ','green'),self.get_transitions())
        print(colored('Alphabet: ','green'),self.get_alphabet())
        print(colored('Number of states: ','green'),len(self.get_states()))
        print(colored('Number of accepting states: ','green'),len(self.get_accepting_states()))
        print(colored('Number of transitions: ','green'),len(self.get_transitions()))
  
    
    # a function to plot the Dfsa as a labelled digraph
    def plot_dfsa_as_labelled_digraph(self):
        G = nx.DiGraph()
        G.add_edges_from([(transition[0][0],transition[1]) for transition in self.get_transitions()])
        
        pos = nx.spring_layout(G)
        color_map = ['black' if node in self.get_accepting_states() else 'white' for node in G]        
        nx.draw(
        G, pos, edge_color='black', width=1, linewidths=2,
        node_size=500, node_color=color_map,
        labels={node: node for node in G.nodes()}
        )
        labels = {(transition[0][0],transition[1]): transition[0][1] for transition in self.get_transitions()}
        nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=labels,
        font_color='red'
        )
        nx.draw_networkx_nodes(G, pos,node_color='lightblue')
        plt.show()   
   
    """
    given a state and an input
    there must exist at most one next state
    since Dfsa is deterministic,
    otherwise a random state is returned
    """
    # pass the state and input/symbol/latter
    def get_next_state(self,at,letter):
        # get all possible next states given at state and symbol
        # since DFSA is deterministic there should be only one next state in list
        next_states = [transition[1] for transition in self.get_transitions() if transition[0][0]==at and transition[0][1]==letter]
        if next_states:
            # if non determinstic transitions are added
            # pop of one next_state at random
            return next_states.pop(random.randrange(len(next_states)))
        else:
            # there is no possible next state return none
            return None
   
    # pass in a state to get all next states on all inputs in alphabet
    def get_next_states(self,state):
        # when at_state (given by transition[0][0]) is the given state
        return [transition[1] for transition in self.get_transitions() if transition[0][0]==state]
    
    """
    A function that randomly initialises a Dfsa based on this following 
    recipe:
    """
    def init_random(self):
      # generate random number n from 16-64
      n = random.randint(16, 64)
      # generate set of vertices {1,2,...n}
      self.set_states([i for i in range(1,n+1)])     
      # randomly choose starting vertex from set of states
      self.set_start_state(random.randint(1,n))
      self.set_accepting_states([i for i in range(1,n+1) if random.randint(0,1)==1])      
      #the alphabet consists only of symbols a and b
      self.set_alphabet(['a','b'])
      # traverse all the states
      for s1 in self.states:
          # traverse all the symbols
          for x in self.alphabet:
              # the transition leads to a random state 
              s2 = random.randint(1,n)
              self.transitions.add(((s1,x),s2))
    # method to get unreachable states
    def get_unreachable_states(self):
        # initialise empty list 
        unreachables = []
        # traverse all vertices and start a BFS from starting vertex
        for end_state in self.get_states():
            # get shortest path from start state to end_state
            shortest_path = self.get_shortest_path(self.get_start_state(),end_state)
            if (not shortest_path):
                # if path is empty -> no path from start state to end state
                # end_state is therefore unreachable
                unreachables.append(end_state)
        return unreachables
    def get_shortest_path(self,at_state,to_state):
        # start BFS starting from starting vertex
        prev = self.solve(at_state)
        # return path from s -> e
        return self.reconstruct_path(at_state,to_state,prev)
    def solve(self,at_state):
        # initialise queue
        queue = []
        # enqueue start_state
        #queue.append(self.get_start_state())
        queue.append(at_state)
        # generate list of False values of length of number if states
        visited = [False]*len(self.get_states())
        #  set visited at location of start_state True
        #visited[self.get_start_state()-1] = True
        visited[at_state-1] = True
        # generate Null list of len dfsa.states
        prev = [None]*len(self.get_states())
        while (len(queue)!=0):
            at_state=queue.pop(0)
            next_states = self.get_next_states(at_state) 
            for next in next_states:
                # if not visited 
                if visited[next-1]==False:
                    # append to queue
                    queue.append(next)
                    visited[next-1]=True        
                    prev[next-1]=at_state
        return prev
    def reconstruct_path(self,at_state,to_state,prev):
        # set up empty list
        path = []
        # state the previus state to to state 
        prev_state = to_state
        # traverse prev list until Null value is found
        while (prev_state != None):
            # append the previous depth elements
            path.append(prev_state)
            prev_state = prev[prev_state-1]
        #  reverse the elements of the list
        path.reverse()
        # if the first element of the path is the intended
        # starting state then the to state is reachable
        # else return empty list
        if path[0]==at_state:
            return path
        return []
    
    def get_depth(self):
        max_ = 0
        for to_state in self.get_states():
            # obtain shortest path by BFS algorithm
            # path -> a list containing vertices in order of path
            path = self.get_shortest_path(self.get_start_state(), to_state)
            # length of path = #vertices - 1
            len_path = len(path)-1
            if max_<len_path:
                max_ = len_path
        return max_
class HopcroftsAlgorithm:
    def __init__(self,old_dfsa):
        example_dfsa = Dfsa()
        example_dfsa.set_states([1,2,3,4,5,6])
        example_dfsa.set_accepting_states([5,6])
        example_dfsa.set_alphabet(['a','b'])
        example_dfsa.set_start_state(1)
        example_dfsa.set_transitions([((1,'a'),2),((2,'a'),3),((2,'b'),4),((3,'b'),5),((4,'b'),6)])
        refinement = self.split(frozenset([1,2,3,4]),set([frozenset([1,2,3,4]),frozenset([5,6])]),example_dfsa)

        self.old_dfsa = old_dfsa
        self.optimised_dfsa = self.hopcroft_algorithm()

    def get_optimised_dfsa(self):
        return self.optimised_dfsa

    def split(self,partition,partitions,dfsa):
        
        # transform set of partitions into a list
        partition_list = list(partitions)
        #a new list to store refinements   
        refinement_list = [frozenset() for i in range(len(partition_list))]
        # get index where the partition to iterate over lies in in partition_list
        index_current_partition = partition_list.index(partition)
        # traverse over the letters
        for letter in dfsa.get_alphabet():
            # travers over the state of the partition we want to refine 
            for state in partition:
                # get next state 
                leading_state = dfsa.get_next_state(state, letter)
                '''
                if next state is none 
                than the state remains in the same 
                partition
                '''
                if leading_state is None:
                    # next state does not exist
                    # current state remains in same partition
                    refinement_list[index_current_partition]=refinement_list[index_current_partition] | frozenset([state])

                # next states exists
                else:
                    # if next state exists
                    
                    # find out in which partition leadingstate lies 
                    for p in partition_list:
                        # if next state lies in paritiion
                        if leading_state in p:
                            index = partition_list.index(p)
                            #put the element in the set corresponding to that partition
                            refinement_list[index] = refinement_list[index] | {state} 
            elements_in_refinement = sum([len(elem) for elem in refinement_list])
        
            if elements_in_refinement == len(partition) and len(refinement_list[index_current_partition])<len(partition_list[index_current_partition]):
                return {i for i in refinement_list if i!=frozenset()}
            refinement_list = [frozenset() for i in range(len(partition_list))]
        # no refinement occurs 
        return {partition}
    def hopcroft_algorithm(self):
        #initialise empty DFSA
        old_dfsa_without_unreachables = Dfsa()
        #set the same alphabet as old 
        old_dfsa_without_unreachables.set_alphabet(self.old_dfsa.get_alphabet())
        # set the same start state as old
        old_dfsa_without_unreachables.set_start_state(self.old_dfsa.get_start_state())
        # get unreachables states of old
        unreachables_of_old_dfsa = self.old_dfsa.get_unreachable_states()
        # getting reachables from start state in  old_dfsa
        reachables_of_old_dfsa = [state for state in self.old_dfsa.get_states() if state not in unreachables_of_old_dfsa]
        # set new states reachables
        old_dfsa_without_unreachables.set_states(reachables_of_old_dfsa)
        # getting the accepting reachables from start state in old_dfsa
        accepting_reachables_of_old_dfsa = [state for state in self.old_dfsa.get_states() if state not in unreachables_of_old_dfsa and state in self.old_dfsa.get_accepting_states()]
        # set accepting states
        old_dfsa_without_unreachables.set_accepting_states(accepting_reachables_of_old_dfsa)
        # getting transitions whose states are in new dfsa
        new_transitions = [transition for transition in self.old_dfsa.get_transitions() if transition[0][0] not in unreachables_of_old_dfsa and transition[1] not in unreachables_of_old_dfsa]          
        # set new transitions 
        old_dfsa_without_unreachables.set_transitions(new_transitions)
        # get rejecting states from newly obtained dfsa 
        rejecting_states = set([state for state in old_dfsa_without_unreachables.get_states() if state not in old_dfsa_without_unreachables.get_accepting_states()])
        '''
        Initialising current partition where
        each partition is in form of a frozenset
        which is immutable and allowed to be an element
        of a set object
        '''
        # initialise current partition
        current = set()
        # add accepting states to current parition
        current.add(frozenset(old_dfsa_without_unreachables.get_accepting_states()))
        # add non accepting states to current partition
        current.add(frozenset(rejecting_states))
        # define an empty set 
        partitions = set()
        while current != partitions:
            # copy the current partitions in temp set
            partitions = current.copy()
            current = set()
            # for each of the curent partition 
            # return a possible partition of that partition
            for partition in partitions:
                example_dfsa = Dfsa()
                example_dfsa.set_states([1,2,3,4,5,6])
                example_dfsa.set_accepting_states([5,6])
                example_dfsa.set_alphabet(['a','b'])
                example_dfsa.set_start_state(1)
                example_dfsa.set_transitions([((1,'a'),2),((2,'a'),3),((2,'b'),4),((3,'b'),5),((4,'b'),6)])

                current = current | self.split(partition, partitions, old_dfsa_without_unreachables)
        # get the refinement consisting of equivalent states
        partitions = list(current) 
        # create new dfsa
        optimised_dfsa = Dfsa()
        # set alphabet as previous
        optimised_dfsa.set_alphabet(old_dfsa_without_unreachables.get_alphabet())
        # set states
        optimised_dfsa.set_states([i for i in range(1,len(partitions)+1)])
        # KEEP TRACK OF STATES (INDEX IN PARTITION LIST)
        state = 1
        for partition in partitions:
            # check if an element in partition is accepting
            if list(partition)[0] in old_dfsa_without_unreachables.get_accepting_states() :
                # add the state corresponding to the partition
                optimised_dfsa.add_accepting_state(state)   
            if old_dfsa_without_unreachables.get_start_state() in partition:
                optimised_dfsa.set_start_state(state)  
            # build the new set of transitions
            # based on the partitions in lists 
            for letter in  old_dfsa_without_unreachables.get_alphabet():
                next_state_old = old_dfsa_without_unreachables.get_next_state(list(partition)[0],letter) 
                # find in which partition next state lies
                next_state_new = 1
                for partition in partitions:
                    if next_state_old in partition:
                        optimised_dfsa.add_transition(((state,letter),next_state_new))
                        break
                    next_state_new+=1        
            # move in to the next partition 
            # -> move in to the next state representing that partition
            state+=1
        # constructing the dfsa from the refinements
        return optimised_dfsa

    
#### Finish of Hopcroft's Algorithm###
class TarjansAlgorithm:
    def __init__(self,dfsa):
        self.dfsa = dfsa
        self.index = 0
        self.undefined = -1
        self.list_of_states = list(self.dfsa.get_states())
        self.onStack = [False]*len(self.list_of_states)
        self.indices = [self.undefined]*len(self.list_of_states)
        self.low_link = [self.undefined]*len(self.list_of_states)
        self.S = [] 
        self.strongly_connected_components = list()
        # add sccs
        for state in self.list_of_states:
            if self.indices[self.list_of_states.index(state)]==self.undefined:
                self.strongconnect(state)
        self.number_of_sccs = len(self.strongly_connected_components)
        self.largest_scc = max(self.strongly_connected_components, key=len)
        self.number_of_states_in_largest_scc = len(self.largest_scc)
        self.smallest_scc = min(self.strongly_connected_components, key=len)
        self.number_of_states_in_smallest_scc = len(self.smallest_scc)
    def get_sccs(self):
        return self.strongly_connected_components      
    def get_number_of_sccs(self):
        return self.number_of_sccs
    def get_largest_scc(self):
        return self.largest_scc
    def get_number_of_states_in_largest_scc(self):
        return self.number_of_states_in_largest_scc
    def get_smallest_scc(self):
        return self.smallest_scc
    def get_number_of_states_in_smallest_scc(self):
        return self.number_of_states_in_smallest_scc
    def strongconnect(self,state):
        index_at = self.list_of_states.index(state)
        # Set the depth index for v to the smallest unused index
        self.indices[index_at] = self.index
        self.low_link[index_at] = self.index
        self.index = self.index + 1
        self.S.append(state)
        self.onStack[index_at] = True
        for next_state in self.dfsa.get_next_states(state):
            index_to = self.list_of_states.index(next_state) 
            if self.indices[index_to]==self.undefined:
        # Successor w has not yet been visited; 
        # recurse on it
                self.strongconnect(next_state)
                self.low_link[index_at] = min(self.low_link[index_to],self.low_link[index_at])
            elif self.onStack[index_to]:
                self.low_link[index_at] = min(self.low_link[index_to],self.low_link[index_at])
                
        # If v is a root state, pop the stack and generate an SCC
        if self.low_link[index_at] ==  self.indices[index_at]:
            scc = set()
            # start a new strongly connected component    
            while True:
                w = self.S.pop()
                index_w = self.list_of_states.index(w)
                self.onStack[index_w] = False
                scc.add(w)
                if w == state:
                    break
            self.strongly_connected_components.append(scc)   
class JohnsonsAlgorithm:
    def __init__(self,dfsa):
        self.dfsa = dfsa
        self.simple_cycles = []
        self.stack = []
    
        self.blocked_set = set([])
        self.blocked_map = set([])
        self.johnsons_algorithm()
    def get_simple_cycles(self):
        return self.simple_cycles        
    def johnsons_algorithm(self):
        temp_dfsa = copy.deepcopy(self.dfsa)
        sccs = TarjansAlgorithm(self.dfsa).get_sccs()
        # going through all states 1,2,3,4,5,...,n
        for scc in sccs:
            for state in scc:
                # find cycles in scc  with starting-ending state  = state
                self.find_cycles_in_scc(scc,state,state)
                # clear contents of stack for next iteration
                self.stack = []
                # clear contents of blocked_set for next iteration
                self.blocked_set = set([])
                # keeps a map of states that can be freed if some state is freed
                self.blocked_map = set([])
                # remove state frome dfsa so it would not be included in next cycle 
                self.dfsa.remove_state(state)
        self.dfsa = temp_dfsa
    def find_cycles_in_scc(self,scc,start_state,current_state):
        found_cycle = False
        self.stack.append(current_state)
        self.blocked_set.add(current_state)
        for neighbour in self.dfsa.get_next_states(current_state):
            if neighbour == start_state:
                self.stack.append(start_state)
                cycle =  [state for state in self.stack]
                cycle.reverse()
                if  cycle not in self.simple_cycles:
                    self.simple_cycles.append(cycle)
                self.stack.pop()
                found_cycle = True
            # else if neighbour is not start state and not in block_set
            elif  neighbour not in self.blocked_set:
                # got_cycle is true if neighbour find cycle in its path
                got_cycle = self.find_cycles_in_scc(scc,start_state,neighbour)
                #  if found cycle is true it will be true for current vertex
                found_cycle = found_cycle or got_cycle
        # if found cycle is true we unblock the current vertex
        if found_cycle:
            self.unblock(current_state)
        # no cycle is not found in the path 
        # add all the neighbours of current vertex to blocked-map
        else:
            for neighbour in self.dfsa.get_next_states(current_state):
                self.blocked_map.add((neighbour,current_state))
        self.stack.pop()
        return found_cycle
    def unblock(self,state):
       self.blocked_set.remove(state)   
       list_block_map = [s[1] for s in self.blocked_map if s[0]==state]
       # if list not empty
       if list_block_map:
           # unblock all states that needs to be unblocked recursively
           for state_to_unblock in list_block_map:
               if state_to_unblock in self.blocked_set:
                   self.unblock(self,state_to_unblock)
           for i in self.blocked_map:
               if i[0]==state:
                   self.blocked_map.remove(i)
