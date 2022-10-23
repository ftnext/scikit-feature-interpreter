from skinterpret.feature_extraction.text import TfidfInterpreter
from sklearn.feature_extraction.text import TfidfVectorizer

CORPUS = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]


def test_can_create_from_fitted_vectorizer():
    vectorizer = TfidfVectorizer()
    vectorizer.fit(CORPUS)

    interpreter = TfidfInterpreter(vectorizer)

    assert interpreter.vectorizer is vectorizer
