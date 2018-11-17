import numpy as np
import random


class ReplayBuffer:

    def __init__(self, vfrac=0, tfrac=0, cfrac=1.0, size=20):
        self.validation_fraction = vfrac
        self.test_fraction = tfrac
        self.training_fraction = 1 - (vfrac + tfrac)

        self.cfrac = cfrac

        self.size = size

        self.buffer = []
        self.cases = []

        self.training_cases = None
        self.validation_cases = None
        self.testing_cases = None

    def push(self, case):
        self.buffer.append(case)
        if len(self.buffer) > self.size:
            self.pop()

    def pop(self):
        popped = self.buffer.pop(-1)
        return popped

    def update(self):
        self.generate_cases()
        self.organize_cases()

    def generate_cases(self):
        # Run the case generator.  Case = [input-vector, target-vector]
        ca = self.buffer[:]
        self.cases = random.sample(ca, round(len(ca) * self.cfrac))

    def organize_cases(self):
        ca = np.array(self.cases)
        np.random.shuffle(ca)  # Randomly shuffle all cases
        separator1 = round(len(self.cases) * self.training_fraction)
        separator2 = separator1 + round(len(self.cases) * self.validation_fraction)
        self.training_cases = ca[0:separator1]
        self.validation_cases = ca[separator1:separator2]
        self.testing_cases = ca[separator2:]

    def get_training_cases(self):
        self.update()
        return self.training_cases

    def get_validation_cases(self):
        self.update()
        return self.validation_cases

    def get_testing_cases(self):
        self.update()
        return self.testing_cases

    def __str__(self):
        out = ""
        for case in self.buffer:
            out += str(case) + "\n"

        return out

    def __len__(self):
        return len(self.buffer)
