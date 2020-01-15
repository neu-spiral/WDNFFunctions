from abc import ABCMeta, abstractmethod #ABCMeta works with Python 2, use ABC for Python 3


class Problem(object): #For Python 3, replace object with ABCMeta
    """Abstract class to parent classes of different problem instances.
    """
    __metaclass__ = ABCMeta #Comment out this line for Python 3


    @abstractmethod
    def __init__(self):
        """
        """
        pass


class DiversityReward(Problem):
    """
    """


    def __init__(self, rewards, givenPartitions, constraintPartitions):
        """ rewards is a dictionary containing {word: reward} pairs,
        givenPartitions is a dictionary containing {partition: word tuples},
        constraintPartitions is a dictionary containing {word: type} pairs.
        """

        return partitionMatroid, wdnf_list
