from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfInterpreter:
    def __init__(self, vectorizer: TfidfVectorizer) -> None:
        self.vectorizer = vectorizer
