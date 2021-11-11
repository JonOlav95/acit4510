
class Normalizer:

    def __init__(self):
        self.mean_dict = {}
        self.std_dict = {}

    def normalize(self, feature, column):
        if feature not in self.mean_dict:
            self.mean_dict[feature] = column.mean()

        if feature not in self.std_dict:
            self.std_dict[feature] = column.std()

        column = (column - self.mean_dict[feature]) / (self.std_dict[feature])

        return column

    def unormalize(self, feature, column):
        column = (column * (self.std_dict[feature])) + self.mean_dict[feature]

        return column
