
"""
Юнит-тест структуры ответа метода analyze_sentiment.
"""
from Start_AVG import SentimentAnalyzer

analyzer = SentimentAnalyzer("model.keras", "tokenizer.pickle")
analyzer.model = None  # mock-загрузка не требуется
analyzer.tokenizer = None

def test_structure_on_none_model():
    try:
        analyzer.analyze_sentiment("Привет")
        assert False, "Ожидалось исключение из-за незагруженной модели"
    except RuntimeError:
        assert True
