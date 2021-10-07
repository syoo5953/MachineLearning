def k_nearest_neighbour(training_set, testing_set, k):
    # sort neighbours by distance and slice list of neighbours to be of length k
    results = []
    for test_instance in testing_set:
        neighbours = []
        for training_instance in training_set:
            training_instance.set_distance(test_instance)
            insert_instance(training_instance, neighbours)
        neighbours = neighbours[0:k]
        # print([neighbour.class_variable for neighbour in neighbours])
        yes_count = 0
        no_count = 0
        # count the votes
        for neighbour in neighbours:
            if neighbour.class_variable == 'yes':
                yes_count += 1
            elif neighbour.class_variable == 'no':
                no_count += 1
        # append result to array
        if no_count > yes_count:
            results.append('no')
        else:
            results.append('yes')
    return results


def insert_instance(instance, neighbours):
    if len(neighbours) != 0:
        for i in range(len(neighbours)):
            if neighbours[i].distance >= instance.distance:
                neighbours.insert(i, instance)
                break
    else:
        neighbours.append(instance)
