
# -*- coding: utf-8 -*-
"""
Functional test (TOP-3 sub sentiment matching) with auto Word report.

Запуск:
    pytest FunctionalTesting_SubSentiment_TOP3.py
"""
import os, pytest
from Start_AVG import SentimentAnalyzer

MODEL_PATH = os.getenv("MOODIX_MODEL", "model.keras")
TOKENIZER_PATH = os.getenv("MOODIX_TOKENIZER", "tokenizer.pickle")

@pytest.fixture(scope="session")
def analyzer():
    sa = SentimentAnalyzer(MODEL_PATH, TOKENIZER_PATH)
    sa.load_model()
    return sa

# Тестовые примеры и ожидаемые суб-настроения
cases = [
    ("Очень удобный в пользовании. Камера и экран супер. Батарея по сравнению с СЕ 2020 кажется не садится.",
     ["joy", "admiration"]),
    ("Приобрела айфон 16 буквально пару дней назад... Телефон мощный, камеры усовершенствовали...",
     ["joy", "admiration", "love", "optimism"]),
    ("Ужасный телефон не стоит своих денег. Достоинства: их в принципе нет. Недостатки: очень быстро нагревается...",
     ["disappointment", "anger", "annoyance"]),
    ("Сильно разочарован этим устройством... теперь белый уходит в синий... Очень обидно наблюдать...",
     ["disappointment", "sadness", "anger"]),
    ("Ужасно! Только зарегистрировался и телефон получил удалённую блокировку...",
     ["anger", "disappointment"]),
    ("Отличный аппарат. Очень компактный, удобный. Оболочка лучшая. Камера, звук супер.",
     ["joy", "admiration"]),
    ("Перешла с Samsung S10plus... Не советую, потратьте деньги на чо-нить другое",
     ["disappointment", "sadness"]),
    ("К огромному сожалению, для моих глаз обзор новых смартфонов... заканчивается. :(",
     ["sadness"]),
]

@pytest.mark.parametrize("text,expected_subs", cases)
def test_sub_sentiment_top3(analyzer, text, expected_subs):
    res = analyzer.analyze_sentiment(text)
    pred_subs = res["sub"]
    top3 = sorted(pred_subs.items(), key=lambda x: x[1], reverse=True)[:3]
    top_labels = {k for k, _ in top3}
    matched = set(expected_subs) & top_labels
    assert matched, f"Ожидались суб-настроения: {expected_subs}, но в top-3 только: {top_labels}"
