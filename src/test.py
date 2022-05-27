import unittest
from dfsa import Dfsa
from dfsa import HopcroftsAlgorithm
from dfsa import TarjansAlgorithm



"""
These test are written to test that 
the algorithms written return the expected code.
"""


class TestRandomDfsaInit(unittest.TestCase):    
    random_dfsa_test = Dfsa()
    # initialise the dfsa randomly
    random_dfsa_test.init_random()
    

    def test_number_of_states_is_between_16_and_64(self):
         # n is number of states
        n = len(self.random_dfsa_test.get_states())
        check = 16<=n and n<=64
        self.assertTrue(check )
    
    
    def test_alphabet_consists_only_of_a_and_b(self):
        self.assertEqual(self.random_dfsa_test.get_alphabet(), set(['a','b']))
    
    
    def test_final_states_subset_set_of_states(self):
        accepting_states = self.random_dfsa_test.get_accepting_states()
        states = self.random_dfsa_test.get_states()
        self.assertTrue(accepting_states.issubset(states))
        
    
    def test_dfsa_complete(self):
        isComplete = True
        """
        traverse each state and letter
        if no next state can be found 
        the DFSA is not complete
        """
        for state in self.random_dfsa_test.get_states():
            for letter in self.random_dfsa_test.get_alphabet():
                if self.random_dfsa_test.get_next_state(state, letter)==None:
                    # not complete
                    isComplete = False
                    break
        self.assertTrue(isComplete)
    def test_start_state_in_state(self):
        self.assertTrue(self.random_dfsa_test.get_states().__contains__(self.random_dfsa_test.get_start_state()))
        
class TestComputeDepthOfDfsa(unittest.TestCase):
    example_dfsa = Dfsa()
    example_dfsa.set_states([1,2,3,4,5])
    example_dfsa.set_accepting_states([3,5])
    example_dfsa.set_start_state(1)
    example_dfsa.set_alphabet(['a','b'])
    example_dfsa.set_transitions([((1,'a'),2),((1,'b'),5),((2,'a'),3),((3,'b'),4),((5,'b'),4)])    
    
    
        
        
    def test_expected_path_is_empty(self):
        self.assertEqual(self.example_dfsa.get_shortest_path(2,5) , [])
    
    def test_expected_path(self):
        self.assertEqual(self.example_dfsa.get_shortest_path(1,4) , [1,5,4])
        
        
    
    def test_number_of_states_is_between_16_and_64(self):

        self.assertEqual(self.example_dfsa.get_depth(), 2)


class TestHopcroftAlgorithm(unittest.TestCase):
    example_dfsa = Dfsa()
    example_dfsa.set_states([1,2,3,4,5,6])
    example_dfsa.set_accepting_states([5,6])
    example_dfsa.set_alphabet(['a','b'])
    example_dfsa.set_start_state(1)
    example_dfsa.set_transitions([((1,'a'),2),((2,'a'),3),((2,'b'),4),((3,'b'),5),((4,'b'),6)])
    
    def test_split_refinement_success(self):
        refinement = HopcroftsAlgorithm(self.example_dfsa).split(frozenset([1,2,3,4]),set([frozenset([1,2,3,4]),frozenset([5,6])]),self.example_dfsa)
        self.assertEqual(refinement,set([frozenset([1,2]),frozenset([3,4])]))
    def test_split_no_refinement_occurs(self):
        refinement = HopcroftsAlgorithm(self.example_dfsa).split(frozenset([5,6]),set([frozenset([1,2,3,4]),frozenset([5,6])]),self.example_dfsa)
        self.assertEqual(refinement,set([frozenset([5,6])]))


class TestTarjanAlgorithm(unittest.TestCase):
    example_dfsa = Dfsa()
    example_dfsa.set_states([1,2,3,4,5,6,7,8,9,10,11,12])
    example_dfsa.set_accepting_states([5,8,9,11])
    example_dfsa.set_alphabet(['a','b'])
    example_dfsa.set_start_state(1)
    example_dfsa.set_transitions([((1,'a'),2),((2,'a'),4),((2,'b'),5),((3,'a'),6),((5,'a'),2),((5,'b'),6),((6,'a'),3),((6,'b'),8),((7,'a'),10),((7,'b'),8),((8,'a'),11),((9,'a'),7),((10,'b'),9),((11,'a'),12),((12,'b'),10)])
    TarjanResults = TarjansAlgorithm(example_dfsa)
    
    def test_expected_sccs(self):
        sccs = self.TarjanResults.get_sccs()
        check = True
        for scc in [set([1]),set([4]),set([2,5]),set([3,6]),set([7,8,9,10,11,12])]:
            if scc not in sccs:
                check = False
                break
        self.assertTrue(check)
    
    def test_get_largest_scc(self):
        self.assertEqual(set([7,8,9,10,11,12]),self.TarjanResults.get_largest_scc())
        
    def test_get_smallest_scc(self):
        self.assertTrue(self.TarjanResults.get_smallest_scc() in [set([1]),set([4])])
    
    def test_size_largest_scc(self):
        self.assertEqual(self.TarjanResults.get_number_of_states_in_largest_scc() , 6)
    
    def test_size_smallest_scc(self):
        self.assertEqual(self.TarjanResults.get_number_of_states_in_smallest_scc(), 1)
if __name__ == '__main__':
    unittest.main()

