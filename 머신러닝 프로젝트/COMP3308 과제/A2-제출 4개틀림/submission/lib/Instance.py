import numpy


class Instance(object):
    def __init__(self, attributes, class_variable=None):
        self.attributes = numpy.asarray(attributes)
        self.class_variable = class_variable
        self.distance = None

    def set_distance(self, other_instance):
        # Return Euclidean distance between two instances.
        self.distance = numpy.sqrt(numpy.sum((self.attributes-other_instance.attributes)**2))
