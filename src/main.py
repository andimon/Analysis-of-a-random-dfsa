#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 09:50:41 2022

@author: andre
"""
import random
from dfsa import Dfsa
from dfsa import HopcroftsAlgorithm
from dfsa import TarjansAlgorithm
from dfsa import JohnsonsAlgorithm



#random.seed(7)
# fix seed  
random.seed(18)
# create empty Dfsa
A = Dfsa();
# init random Dfsa
A.init_random()
# display random Dfsa 
print('--------------------------------')
print('DFSA A')
print('--------------------------------')
A.display_dfsa()
print('--------------------------------')
# plot dfsa
A.plot_dfsa_as_labelled_digraph()
print('----------------------------------------')
print('Number of states in A and depth of A')
print('----------------------------------------')
# get # states of A
print('# states of A is ',len(A.get_states()))
# print  the shortest path and exclude unreachable states i.e. empty paths
shortest_paths = [A.get_shortest_path(A.get_start_state(), to_state) for to_state in A.get_states() if len(A.get_shortest_path(A.get_start_state(),to_state))!=0]
print('All shortest path from starting state ',shortest_paths)
# get depth of A
print('Depth of A is ',A.get_depth())
print('--------------------------------')
print('DFSA M')
print('--------------------------------')
M = HopcroftsAlgorithm(A).get_optimised_dfsa()
# obtaining info about optimised DFSA
M.display_dfsa()
print('--------------------------------')
# plotting DFSA
M.plot_dfsa_as_labelled_digraph()
print('----------------------------------------')
print('Number of states in M and depth of M')
print('----------------------------------------')
# get # states of M
print('# states of M is ',len(M.get_states()))
# get depth of M
print('Depth of M is ',M.get_depth())
print('------------------------------------------')
print('The strongly connected components of M')
print('------------------------------------------')
print(TarjansAlgorithm(M).get_sccs())
print('------------------------------------------')
print('Number of strongly connected components of M')
print('------------------------------------------')
print(len(TarjansAlgorithm(M).get_sccs()))
print('------------------------------------------')
print('A largest SCC of M')
print('------------------------------------------')
print(TarjansAlgorithm(M).get_largest_scc())
print('------------------------------------------')
print('Size of a largest SCC of M')
print('------------------------------------------')
print(len(TarjansAlgorithm(M).get_largest_scc()))
print('------------------------------------------')
print('A smallest SCC of M')
print('------------------------------------------')
print(TarjansAlgorithm(M).get_smallest_scc())
print('------------------------------------------')
print('Size of a smallest SCC of M')
print('------------------------------------------')
print(len(TarjansAlgorithm(M).get_smallest_scc()))
print('------------------------------------------')
print('Obtaining Simple Cycles of M')
print('------------------------------------------')
johnson_algorithm_M = JohnsonsAlgorithm(M)
print(johnson_algorithm_M.get_simple_cycles())










