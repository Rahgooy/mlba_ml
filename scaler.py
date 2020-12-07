from sklearn.preprocessing import StandardScaler, Normalizer

class DummyScaler:
    def transform(self, x):
        return x

    def fit_transform(self, x):
        return x


class CustomScaler:
    def __init__(self):
        self.norm = Normalizer(norm='max')
        self.st = StandardScaler()

    def transform(self, x):
        x = self.norm.transform(x)
        return self.st.transform(x)

    def fit_transform(self, x):
        x = self.norm.fit_transform(x)
        return self.st.fit_transform(x)
