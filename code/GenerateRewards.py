import argparse
import numpy as np
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a random rewards dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--size', default=20, help='Total size of the ground set')
    parser.add_argument('--partitions', default = 5, help = 'Number of partitions in the ground set')
    parser.add_argument('--types', default = 2, help = 'Number of targeted partitions of the ground set')
    parser.add_argument('--rewardsOutput', default = 'rewards.txt', help = 'File in which output rewards are stored')
    parser.add_argument('--givenPartitionsOutput', default = 'givenPartitions.txt', help = 'File in which given partitions are stored')
    parser.add_argument('--targetedPartitionsOutput', default = 'types.txt', help = 'File in which targeted partitions are stored')
    args = parser.parse_args()

    rewards = {1: 0.3, 2: 0.2, 3: 0.1, 4: 0.6, 5: 0.5, 6: 0.4}  # {x_i: r_i} pairs
    with open("datasets/DR_rewards0", "w") as f:
        pickle.dump(rewards, f)
    givenPartitions = {1: (1, 5), 2: (2, 3), 3: (4, 6)}  # {P_i: (x_j)} pairs where x_j in P_i
    with open("datasets/DR_givenPartitions0", "w") as f:
        pickle.dump(givenPartitions, f)
    types = {1: 2, 2: 2, 3: 2, 4: 1, 5: 2, 6: 1}  # {x_i: type} pairs
    with open("datasets/DR_types0", "w") as f:
        pickle.dump(types, f)
    k_list = {1: 1, 2: 2}
    with open("datasets/DR_k_list0", "w") as f:
        pickle.dump(k_list, f)


    # #creates random {x_i: r_i} pairs
    # rewards = dict()
    # size = args.size #size of rewards
    # for i in range(1, size + 1):
    #     rewards[i] = np.random.rand()
    # f = open(args.rewardsOutput, "w")
    # f.write(str(rewards))
    # f.close()
    #
    # #randomly partition
    # N = args.partitions #number of partitions
    # givenPartitions = dict()
    # for i in range(1, N + 1):
    #     givenPartitions[i] = ()
    #
    # for i in range(1, size + 1):
    #     dice = np.random.rand()
    #     for j in range(1, N + 1):
    #         if dice < (j * 1.0) / N:
    #             givenPartitions[j] += (i,)
    #             break
    #
    # f = open(args.givenPartitionsOutput, "w")
    # f.write(str(givenPartitions))
    # f.close()
    #
    # T = args.types #number of types
    # types = dict()
    # for i in range(1, size + 1):
    #     dice = np.random.rand()
    #     for j in range(1, T + 1):
    #         if dice < (j * 1.0) / T:
    #             types[i] = j
    #             break
    #
    # f = open(args.targetedPartitionsOutput, "w")
    # f.write(str(types))
    # f.close()
