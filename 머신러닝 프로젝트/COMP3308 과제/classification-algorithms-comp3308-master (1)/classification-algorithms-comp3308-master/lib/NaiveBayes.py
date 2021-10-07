import numpy


def naive_bayes(training_set, testing_set):
    set_length = len(training_set)
    num_of_attributes = len(training_set[0].attributes)
    yes_count = 0
    no_count = 0

    # Split attributes into yes/no and types
    dict_of_attributes = dict()
    for i in range(num_of_attributes):
        dict_of_attributes[i] = {'yes': [],
                                 'no': []
                                 }

    for instance in training_set:
        if instance.class_variable == 'yes':
            yes_count += 1
        else:
            no_count += 1
        for i in range(num_of_attributes):
            dict_of_attributes[i][instance.class_variable].append(instance.attributes[i])

    # Get probability of yes/no
    p_yes = yes_count/set_length
    p_no = no_count/set_length

    # Obtain standard deviation and mean for each attribute/class
    stats_dict = dict()
    for i in range(num_of_attributes):
        stats_dict[i] = {'s_dev': {'yes': numpy.std(dict_of_attributes[i]['yes']),
                                   'no': numpy.std(dict_of_attributes[i]['no'])},
                         'mean': {'yes': numpy.mean(dict_of_attributes[i]['yes']),
                                  'no': numpy.mean(dict_of_attributes[i]['no'])}
                         }

    # for x in stats_dict:
    #     print(x)
    #     for y in stats_dict[x]:
    #         print(y, ':', stats_dict[x][y])
    # Calculate results
    results = []
    for instance in testing_set:
        nb_yes = p_yes
        nb_no = p_no
        for i in range(num_of_attributes):
            nb_yes *= probability_density_function(instance.attributes[i],
                                                   stats_dict[i]['mean']['yes'],
                                                   stats_dict[i]['s_dev']['yes'])
            nb_no *= probability_density_function(instance.attributes[i],
                                                   stats_dict[i]['mean']['no'],
                                                   stats_dict[i]['s_dev']['no'])
        if nb_no > nb_yes:
            results.append('no')
        else:
            results.append('yes')
    return results


def probability_density_function(x, mean, s_dev):
    return (1/(s_dev*numpy.sqrt(2*numpy.pi)))*numpy.e**((-1)*((x-mean)**2/(2*s_dev**2)))
