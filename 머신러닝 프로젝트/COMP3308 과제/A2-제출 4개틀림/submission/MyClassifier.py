import sys
import numpy
from lib.Instance import Instance
from lib.KNearestNeighbour import k_nearest_neighbour
from lib.NaiveBayes import naive_bayes


def parse_input_file(filename):
    """
    parses input file and returns array of instances.
    :param filename: name of file to be parsed
    :return: array of instances
    """
    instances = []
    with open(filename, "r") as infile:
        for line in infile:
            attributes = line.split(',')
            # convert from str to float
            for i in range(len(attributes)):
                if attributes[i] != 'yes\n' and attributes[i] != 'no\n':
                    attributes[i] = numpy.float(attributes[i])
                else:
                    attributes[i] = attributes[i].replace('\n', '')
            # If this is a training set
            if attributes[-1] == 'yes' or attributes[-1] == 'no':
                instances.append(Instance(attributes=attributes[0:len(attributes)-1], class_variable=attributes[-1]))
            # If this is a testing set
            else:
                instances.append(Instance(attributes=attributes))

    return instances


if __name__ in '__main__':
    training_data = sys.argv[1]
    testing_data = sys.argv[2]
    algorithm = sys.argv[3]

    training_instances = parse_input_file(training_data)
    testing_instances = parse_input_file(testing_data)

    if algorithm == 'NB':
        # Do Naive Bayesian classification
        results = naive_bayes(training_instances, testing_instances)
    else:
        k = int(algorithm.replace('NN', ''))
        # Do k-nearest neighbour classification
        results = k_nearest_neighbour(training_instances, testing_instances, k)
    for result in results:
        print(result)