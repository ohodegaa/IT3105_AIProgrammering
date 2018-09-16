import numpy as np
import random


# *********** CASE MANAGER ********
# This is a simple class for organizing the cases (training, validation and test) for a
# a machine-learning system

class Caseman:

    def __init__(self, cfunc, vfrac=0, tfrac=0, cfrac=1.0):
        self.casefunc = cfunc
        self.validation_fraction = vfrac
        self.test_fraction = tfrac
        self.training_fraction = 1 - (vfrac + tfrac)
        self.cfrac = cfrac
        self.cases = None
        self.training_cases = None
        self.validation_cases = None
        self.testing_cases = None
        self.generate_cases()
        self.organize_cases()

    def generate_cases(self):
        ca = self.casefunc()
        self.cases = random.sample(ca, round(
            len(ca) * self.cfrac))  # Run the case generator.  Case = [input-vector, target-vector]

    def organize_cases(self):
        ca = np.array(self.cases)
        np.random.shuffle(ca)  # Randomly shuffle all cases
        separator1 = round(len(self.cases) * self.training_fraction)
        separator2 = separator1 + round(len(self.cases) * self.validation_fraction)
        self.training_cases = ca[0:separator1]
        self.validation_cases = ca[separator1:separator2]
        self.testing_cases = ca[separator2:]

    def get_training_cases(self): return self.training_cases

    def get_validation_cases(self): return self.validation_cases

    def get_testing_cases(self): return self.testing_cases
