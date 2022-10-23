from sklearn.feature_extraction.text import TfidfVectorizer

TOKEN = str
TFIDF = float
TOKEN_TFIDF_PAIR = tuple[TOKEN, TFIDF]


class TfidfInterpreter:
    def __init__(self, vectorizer: TfidfVectorizer) -> None:
        self.vectorizer = vectorizer

    def interpret(self, document: str) -> list[TOKEN_TFIDF_PAIR]:
        tfidf_vector = self.vectorizer.transform([document])[0]
        tfidf_array = tfidf_vector.toarray()[0]
        tokens = self.vectorizer.inverse_transform(tfidf_vector)[0]
        pairs = [
            (token, tfidf_array[self.vectorizer.vocabulary_[token]])
            for token in tokens
        ]
        return sorted(pairs, key=lambda pair: pair[1], reverse=True)
