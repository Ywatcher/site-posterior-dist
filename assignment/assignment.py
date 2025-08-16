import logging
from cse587Autils.SequenceObjects.SequenceModel import SequenceModel
import numpy as np

logger = logging.getLogger(__name__)


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

    # YOUR CODE HERE

    # Normalize the posterior probabilities and store the result in a variable
    # called posterior_prob

    # YOUR CODE HERE

    return posterior_prob
