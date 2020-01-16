from abc import ABCMeta, abstractmethod #ABCMeta works with Python 2, use ABC for Python 3
from wdnf import wdnf
from ContinuousGreedy import LinearSolver, PartitionMatroidSolver


def log(x):
    """
    """
    output = 0.0
    for wdnf_object in wdnf_list:
        output += wdnf_object.evaluate(x, np.log1p)
    return output


def qs(x):
    return x / (1.0 - x)


def queueSize(x):
    """
    """
    output = 0.0
    for wdnf_object in wdnf_list:
        output += wdnf_object.evaluate(x, qs)
    return output


def derive(type, x, degree):
    if type == 'ln':
        if degree == 0:
            return np.log1p(x) #log1p(x) is ln(x+1)
        else:
            return (((-1.0)**degree) * math.factorial(degree)) / ((1.0 + x)**(degree + 1))
    if type == 'queueSize':
        if degree == 0:
            return 1.0 / (1.0 - x)
        else:
            return math.factorial(degree) / ((1.0 - x)**(degree + 1))


def findDerivatives(type, center, degree):
    """Type is either 'ln' or 'queueSize'.
    """
    derivatives = []
    for i in range(degree + 1):
        derivatives.append(derive(type, center, i))
    return derivatives


def evaluateAll(taylor_instance):
    my_wdnf = wdnf(dict(), wdnf_list[0].sign)
    #print(my_wdnf.coefficients)
    for wdnf_instance in wdnf_list:
        #print(wdnf_instance.coefficients)
        #print(taylor_instance.compose(wdnf_instance).coefficients)
        my_wdnf += taylor_instance.compose(wdnf_instance)
        #print(my_wdnf.coefficients)
    return my_wdnf


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
        self.y = dict.fromkeys(rewards.iterkeys(), 0.0)


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



if __name__ == "__main__":
    rewards = {1: 0.3, 2: 0.2, 3: 0.1, 4: 0.7, 5: 0.05, 6: 0.4}
    givenPartitions = {'fruits': (1, 5), 'things': (2, 3), 'actions': (4, 6)}
    types = {1: 'noun', 2: 'noun', 3: 'noun', 4: 'verb', 5: 'noun', 6: 'verb'}
    newProblem = DiversityReward(rewards, givenPartitions, types)
    #for item in newProblem.wdnf_list:
        #print item.coefficients
        #print item.sign
    #print(newProblem.partitionedSet)
    wdnf_list = newProblem.wdnf_list
    estimator1 = SamplerEstimator(log, 10)
    linearSolver = PartitionMatroidSolver(newProblem.partitionedSet, {'verb': 1, 'noun': 2})
    cg1 = ContinuousGreedy(linearSolver, estimator1)
    Y1 = cg1.FW(3)
    print(Y1)

    derivatives = findDerivatives('ln', 0, 1)
    myTaylor = taylor(1, derivatives, 0)
    print(myTaylor.poly_coef)
    my_wdnf = evaluateAll(myTaylor)
    print(my_wdnf.coefficients)
    estimator2 = PolynomialEstimator(my_wdnf)
    cg2 = ContinuousGreedy(linearSolver, estimator2)
    Y2 = cg2.FW(3)
    print(Y2)


    # actors = {'act1', 'act2', 'act3', 'act4', 'act5'}
    # actors_gradient = {'act1': 1000, 'act2': 300, 'act3': 400, 'act4': 500, 'act5': 700}
    # NewUniSolver = UniformMatroidSolver(actors, 3)
    # print(NewUniSolver.solve(actors_gradient))
    # print(isinstance(NewUniSolver, UniformMatroidSolver))
    #
    # directors = {'dir1', 'dir2', 'dir3'}
    # directors_gradient = {'dir1': 1500, 'dir2': 1200, 'dir3': 250}
    # figurants = {'fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'fig6', 'fig7'}
    # figurants_gradient = {'fig1': 10, 'fig2': 20, 'fig3': 35, 'fig4': 5, 'fig5': 6, 'fig6': 2, 'fig7': 13}
    # candidates = {'actors': actors, 'directors': directors, 'figurants': figurants}
    # candidates_gradient = {}
    # candidates_gradient.update(actors_gradient)
    # candidates_gradient.update(directors_gradient)
    # candidates_gradient.update(figurants_gradient)
    # k_list = {'actors': 2, 'directors': 1, 'figurants': 5}
    # NewPartSolver = PartitionMatroidSolver(candidates, k_list)
    # print(NewPartSolver.solve(candidates_gradient))
    # print(findDerivatives('queueSize', 3, 5))
    #
    # kids = {'kid1': {'goalkeeper': 100, 'defender': 50, 'forward': 25},
    #         'kid2': {'goalkeeper': 80, 'defender': 150, 'forward': 30},
    #         'kid3': {'goalkeeper': 10, 'defender': 35, 'forward': 250},
    #         'kid4': {'goalkeeper': 300, 'defender': 5, 'forward': 125},
    #         'kid5': {'goalkeeper': 20, 'defender': 50, 'forward': 75},
    #         'kid6': {'goalkeeper': 50, 'defender': 28, 'forward': 36},
    #         'kid7': {'goalkeeper': 60, 'defender': 90, 'forward': 12},
    #         'kid8': {'goalkeeper': 70, 'defender': 90, 'forward': 450},
    #         'kid9': {'goalkeeper': 45, 'defender': 350, 'forward': 30},
    #         'kid10': {'goalkeeper': 40, 'defender': 48, 'forward': 45},
    #         'kid11': {'goalkeeper': 175, 'defender': 12, 'forward': 120}}
    #
    # def findPartition(items, partition):
    #     result = {}
    #     for item in items:
    #         result[item] = items[item][partition]
    #     return result
    #
    # goalkeepers = findPartition(kids, 'goalkeeper')
    # defenders = findPartition(kids, 'defender')
    # forwards = findPartition(kids, 'forward')
    #
    # players = {'goalkeepers': goalkeepers, 'defenders': defenders, 'forwards': forwards}
    # player_list = {'goalkeepers': 1, 'defenders': 5, 'forwards': 5}
    # print(NewPartSolver.solve(players, player_list))
