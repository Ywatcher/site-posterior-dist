import logging
import unittest
from gradescope_utils.autograder_utils.decorators import weight #type: ignore
from cse587Autils.SequenceObjects.SequenceModel import SequenceModel
from cse587Autils.configure_logging import configure_logging

# Handle both VS Code (relative import) and autograder (absolute import) contexts
try:
    from .assignment import site_posterior  # VS Code context
except ImportError:
    from assignment import site_posterior  # Autograder context

configure_logging(logging.WARN)


class Testsite_posterior(unittest.TestCase):

    @weight(4)
    def test_1(self):
        """
        Testing the case where the sequence is equally likely under either
        model
        """
        sequence = [1, 2, 3]
        site_base_probs = [[1/4, 1/4, 1/4, 1/4],
                           [1/4, 1/4, 1/4, 1/4],
                           [1/4, 1/4, 1/4, 1/4]]
        background_base_probs = [1/4, 1/4, 1/4, 1/4]

        sm = SequenceModel(1/2, site_base_probs, background_base_probs)

        actual = site_posterior(sequence, sm)

        self.assertAlmostEqual(actual, 1/2, places=10)

    @weight(2)
    def test_2(self):
        """
        Testing that, when the bases provide no evidence favoring either 
        sequence type, the posterior is equal to the prior.
        Requires exact arithmetic to pass.
        """
        sequence = [2, 1, 3]
        site_base_probs = [[1/4, 1/4, 1/4, 1/4],
                           [1/4, 1/4, 1/4, 1/4],
                           [1/4, 1/4, 1/4, 1/4]]
        background_base_probs = [1/4, 1/4, 1/4, 1/4]

        sm = SequenceModel(1/3, site_base_probs, background_base_probs)

        actual = site_posterior(sequence, sm)

        self.assertAlmostEqual(actual, 1/3, places=10)

    @weight(2)
    def test_3(self):
        """
        Testing that the likelihood can overcome the prior.
        Since some of the inputs are in floating (approximate) arithmetic,
        the output is, too.
        """
        sequence = [1, 2, 0]
        site_base_probs = [[0.1, 0.5, 0.3, 0.1],
                           [0.1, 0.4, 0.4, 0.1],
                           [0.8, 0.1, 0.1, 0.0]]
        background_base_probs = [0.25, 0.25, 0.25, 0.25]

        sm = SequenceModel(1/3, site_base_probs, background_base_probs)

        actual = site_posterior(sequence, sm)

        self.assertAlmostEqual(actual, 0.84, places=2)

    @weight(2)
    def test_4(self):
        """
        Testing the handling of a zero in backgroundProbs.
        """
        sequence = [0, 2, 0]
        site_base_probs = [[0.25, 0.25, 0.25, 0.25],
                           [0.25, 0.25, 0.25, 0.25],
                           [1/8, 1/8, 1/8, 5/8]]
        background_base_probs = [1/8, 2/8, 0/8, 5/8]

        sm = SequenceModel(1/3, site_base_probs, background_base_probs)

        actual = site_posterior(sequence, sm)

        self.assertAlmostEqual(actual, 1, places=2)

    @weight(2)
    def test_5(self):
        """
        Testing the handling of a zero in siteProbs.
        """
        sequence = [0, 2, 0]
        site_base_probs = [[0.25, 0.25, 0.25, 0.25],
                           [0.25, 0.5, 0, 0.25],
                           [1/8, 1/8, 1/8, 5/8]]
        background_base_probs = [1/4, 1/4, 1/4, 1/4]

        sm = SequenceModel(1/3, site_base_probs, background_base_probs)

        actual = site_posterior(sequence, sm)
        
        self.assertAlmostEqual(actual, 0., places=2)

    @weight(2)
    def test_6(self):
        """
        Testing the handling of a zero in siteProbs for an unobserved base.
        """
        sequence = [0, 0, 3]
        site_base_probs = [[0.25, 0.5, 0, 0.25],
                           [0.25, 0, 0.5, 0.25],
                           [1/8, 1/8, 1/8, 5/8]]
        background_base_probs = [1/4, 1/4, 1/4, 1/4]

        sm = SequenceModel(1/3, site_base_probs, background_base_probs)

        actual = site_posterior(sequence, sm)

        self.assertAlmostEqual(actual, 0.56, places=2)

    @weight(2)
    def test_7(self):
        """
        Testing the handling of seeing bases with probability zero under
            both sites and background models.
        """
        sequence = [0, 0, 3]
        site_base_probs = [[0.25, 0.5, 0, 0.25],
                           [0.25, 0, 0.5, 0.25],
                           [1/3, 1/3, 1/3, 0]]
        background_base_probs = [1/3, 1/3, 1/3, 0]

        sm = SequenceModel(1/3, site_base_probs, background_base_probs)

        with self.assertRaises(ZeroDivisionError):
            site_posterior(sequence, sm)

    @weight(2)
    def test_8(self):
        """
        Testing the handling of a zero in the site prior and siteProbs.
        """
        sequence = [1, 2, 3]
        site_base_probs = [[0, 0.8, 0.1, 0.1],
                           [0.1, 0, 0.8, 0.1],
                           [1/3, 1/3, 1/3, 0]]
        background_base_probs = [1/4, 1/4, 1/4, 1/4]

        sm = SequenceModel(0, site_base_probs, background_base_probs)

        actual = site_posterior(sequence, sm)

        self.assertAlmostEqual(actual, 0, places=2)

    @weight(2)
    def test_9(self):
        """
        Testing the handling of a zero in the background prior and siteProbs.
        """
        sequence = [1, 2, 3]
        site_base_probs = [[0, 0.8, 0.1, 0.1],
                           [0.1, 0, 0.8, 0.1],
                           [1/3, 1/3, 1/3, 0]]
        background_base_probs = [1/4, 1/4, 1/4, 1/4]

        sm = SequenceModel(1, site_base_probs, background_base_probs)

        with self.assertRaises(ZeroDivisionError):
            site_posterior(sequence, sm)