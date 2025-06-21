
"""
Юнит-тест функции clean_text класса SentimentAnalyzer.
"""
import pytest
from Start_AVG import SentimentAnalyzer

analyzer = SentimentAnalyzer("model.keras", "tokenizer.pickle")

@pytest.mark.parametrize("raw,expected", [
    ("Это ужасно!!! 😡 123", "это ужасно 123"),
    ("   Привет   мир  ", "привет мир"),
    ("Hello, WORLD!!!", "hello world"),
    ("123 *** ###", "123"),
])
def test_clean_text(raw, expected):
    assert analyzer.clean_text(raw) == expected
