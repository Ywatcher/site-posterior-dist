import logging
from cse587Autils.SequenceObjects.SequenceModel import SequenceModel
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

class LikelihoodSequenceModel(SequenceModel):
    def sample_motif(self):
        vectorized_choice = np.vectorize(
            # 4 base 
            pyfunc=lambda prob_base_i: np.random.choice(4, size=1, p=prob_base_i),
            signature="(m) -> ()"
        )
        return vectorized_choice(self.site_base_probs)

    def sample_background(self):
        # print("size",self.motif_length, "p",self.background_base_probs)
        return np.random.choice(
            4, # # mapping: 0 = A, 1 = C, 2 = G, 3 = T
            size= self.motif_length(),
            replace=True, 
            p=self.background_base_probs 
        )

    def likelihood_motif(self, sequence_onehot):
        # sequence_onehot: [seq_len, 4]
        # base_prob: [seq_len, 4]
        # likelihood: prod_i sum_k(p_base_ik * delta_ik)
        return np.prod(np.einsum(
            "lb, lb->l",
            sequence_onehot, self.site_base_probs
            ))

    def likelihood_background(self, sequence_onehot):
        # sequence_onehot: [seq_len, 4]
        # base_prob: [4 ]
        return np.prod(np.einsum(
            "lb, b->l",
            sequence_onehot, self.background_base_probs
            ))
    # @property
    # def prior_np(self) -> NDArray:
        # # [p_site, p_bg]
        # return np.array([self.site_prior, 1-self.site_prior])
    
    @classmethod
    def from_parent(cls, obj:SequenceModel):
        return cls(
            obj.site_prior, obj.site_base_probs, obj.background_base_probs,
            obj._precision, obj._tolerance
        )

def seq2onehot(sequence:list[int],num_classes:int) -> NDArray:
    seq_len = len(sequence)
    onehot = np.eye(num_classes)[sequence]
    return onehot

def site_posterior(sequence: list[int],
                   sequence_model: SequenceModel) -> float:
    """
    Calculate the posterior probability of a bound site versus an unbound site.

    :param sequence: Observed bases represented as integers where 0 = A,
        1 = C, 2 = G and 3 = T
    :type sequence: list[int]
    :param sequence_model: A SequenceModel object which stores the parameters
        of the genome model which produced the sequence
    :type sequence_model: SequenceModel

    :return: Posterior probability of a bound site
    :rtype: float

    :Example:
    >>> site_base_probs = [[0.25, 0.25, 0.25, 0.25],
    ...                    [0.25, 0.25, 0.25, 0.25],
    ...                    [0.25, 0.25, 0.25, 0.25],
    ...                    [0.25, 0.25, 0.25, 0.25]]
    >>> background_base_probs = [0.25, 0.25, 0.25, 0.25]
    >>> sm = SequenceModel(0.01, site_base_probs, background_base_probs)
    >>> site_posterior([0, 1, 2, 3], sm)
    0.01
    """
    # check that the inputs are valid
    if not isinstance(sequence, list):
        raise TypeError("sequence must be a list")
    if not len(sequence) == sequence_model.motif_length():
        raise ValueError(
            "sequence and site_base_probs must be the same length")
    for base in sequence:
        if not isinstance(base, int):
            raise TypeError("sequence must be a list of integers")
        if base < 0 or base > 3:
            raise ValueError("sequence must be a list of integers between 0 "
                             "and 3 (inclusive)")
    sequence_model = LikelihoodSequenceModel.from_parent(sequence_model) 
    # YOUR CODE HERE
    sequence_onehot = seq2onehot(sequence, num_classes=4)
    numerator_site = sequence_model.site_prior * sequence_model.likelihood_motif(sequence_onehot)
    numerator_bg = (1-sequence_model.site_prior) * sequence_model.likelihood_background(sequence_onehot)
    if numerator_site == 0 and numerator_bg == 0:
        raise ZeroDivisionError("got likelihood be 0 for both site and background")
    posterior_prob = numerator_site / (numerator_site + numerator_bg)


    # Normalize the posterior probabilities and store the result in a variable
    # called posterior_prob

    # YOUR CODE HERE

    return posterior_prob
