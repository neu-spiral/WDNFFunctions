from abc import ABCMeta, abstractmethod #ABCMeta works with Python 2, use ABC for Python 3
from wdnf import wdnf
#from ContinuousGreedy import LinearSolver, PartitionMatroidSolver


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


    def __init__(self, rewards, givenPartitions, types):
        """ rewards is a dictionary containing {word: reward} pairs,
        givenPartitions is a dictionary containing {partition: word tuples},
        types is a dictionary containing {word: type} pairs.
        """
        wdnf_list = []
        partitionedSet = {}
        for i in givenPartitions:
            coefficients = {}
            for j in givenPartitions[i]:
                coefficients[j] = rewards[j]
                if partitionedSet.has_key(types[j]):
                    partitionedSet[types[j]].add(j)
                else:
                    partitionedSet[types[j]] = {j}
            new_wdnf = wdnf(coefficients, 1)
            wdnf_list.append(new_wdnf)
        self.wdnf_list = wdnf_list
        self.partitionedSet = partitionedSet
        self.problemSize = len(rewards)


class QueueSize(Problem):
    """
    """


    def __init__(self):
        pass


class InfluenceMaximization(Problem):
    """
    """


    def __init__(self):
        pass


class FacilityLocation(Problem):
    """
    """


    def __init__(self):
        pass
