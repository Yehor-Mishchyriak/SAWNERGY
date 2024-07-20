from math import log, exp
import numpy as np
import util

class Protein:

    def __init__(self, residues: dict, transition_probabilities_matrix: np.array, epsilon: float = 1e-10) -> None:
        '''
        Initialize a Protein object.
        
        Parameters:
        residues: A dictionary mapping indices to respective amino acids in the protein.
        transition_probabilities: A 2D square matrix whose entries are transition probabilities of an allosteric signal between the residues.
        epsilon: A small value added to transition probabilities to avoid logarithm of zero issues.
        '''
        self.residues = residues
        self.number_residues = len(self.residues)
        self.epsilon = epsilon
        # Ensuring that no transition probability is zero, preventing issues with taking logarithms
        self.transition_probabilities_matrix = transition_probabilities_matrix + self.epsilon
        # Validate and normalize the transition probabilities if necessary
        self._validate_and_normalize_transition_probabilities()

    def _validate_and_normalize_transition_probabilities(self) -> None:
        '''
        Validate and normalize the transition probabilities matrix if necessary.
        '''
        if self.transition_probabilities_matrix.shape != (self.number_residues, self.number_residues):
            raise ValueError("transition_probabilities_matrix does not correspond to the number of residues")
        
        row_sums = np.sum(self.transition_probabilities_matrix, axis=1)
        if not np.allclose(row_sums, 1):
            self.transition_probabilities_matrix = util.normalize_row_vectors(self.transition_probabilities_matrix)
    
    def get_transition_probabilities_from_residue(self, residue_index: int) -> np.array:
        if residue_index not in self.residues:
            raise ValueError(f"Start residue index {residue_index} is not valid.")
        return self.transition_probabilities_matrix[residue_index, :]

    def allosteric_signal_path_generator(self, start: int, number_iterations: int, most_probable_path: bool = True):
        '''
        Generate the path of an allosteric signal starting from a given residue.

        Parameters:
        start: The starting residue index.
        number_iterations: The number of iterations to perform.
        most_probable_path: A boolean indicating whether to choose the most probable path (True) or a path based on the probability distribution (False).

        Yields:
        A tuple (next_residue, transition_probability) for each step in the path.
        '''
        if number_iterations < 0:
            raise ValueError("Number of iterations must be non-negative.")
        if start not in self.residues:
            raise ValueError(f"Start residue index {start} is not valid.")

        current_residue = start
        current_path = list()
        for _ in range(number_iterations):
            # Get transition probabilities from the current residue
            transition_probabilities_from_current_residue = self.get_transition_probabilities_from_residue(current_residue)
            # fixes count to infinity problem (no ping pong between two residues that have high interaction energy)
            # this approach also ensures that the signal has some sort of direction -- it cannot go back to the preceding residue right away
            # however, there's another problem: it can loop if there's a triangle or some other combination of residues that makes it loop
            # I think, the way to solve it can be adding stochasticity in cases of loops:
            # check if the reside continues to show up in the nearest slice of the constructed path
            # if it does, change the path generation to stochastic for the next round(s)
            if len(current_path) > 0:
                last_perturbed_residue = current_path[-1]
                transition_probabilities_from_current_residue[last_perturbed_residue] = 0.0
                transition_probabilities_from_current_residue = util.normalize_vector(transition_probabilities_from_current_residue)
            # Find the next residue given the transition probability distribution
            if most_probable_path:  # Chooses the most likely next residue (deterministic given a static matrix)
                next_residue = np.argmax(transition_probabilities_from_current_residue)
            else:  # Chooses the next residue given the probability distribution (stochastic given a static matrix)
                # Draw a residue from the p=transition_probabilities_from_current_residue distribution
                next_residue = np.random.choice(range(self.number_residues), p=transition_probabilities_from_current_residue)

            # Yield the next residue and its transition probability
            yield next_residue, transition_probabilities_from_current_residue[next_residue]

            # Move to the next residue
            current_residue = next_residue
    
    def extend_path(self, residue: int, transition_probability: float) -> None:
        '''
        Extend the current path with a new residue and its transition probability.

        Parameters:
        residue: The next residue to add to the path.
        transition_probability: The transition probability to the next residue.
        '''
        # Append the residue to the path
        self.allosteric_signal_propagation[0] += self.residues[residue]
        # Add the log of the transition probability to the total log probability
        self.allosteric_signal_propagation[1] += log(transition_probability)

    def clear_path(self) -> None:
        '''
        Clear the current path and reset the probability.
        '''
        self.allosteric_signal_propagation[0] = ""
        self.allosteric_signal_propagation[1] = 0.0

    def show_path(self) -> tuple:
        '''
        Show the current path and its probability.

        Returns:
        A tuple containing the path as a string of residues and the total probability of the path.
        '''
        # Get the path
        path = self.allosteric_signal_propagation[0]
        # Calculate the probability from the log probability
        probability = exp(self.allosteric_signal_propagation[1])
        return path, probability


def main():
    pass


if __name__ == "__main__":
    main()
