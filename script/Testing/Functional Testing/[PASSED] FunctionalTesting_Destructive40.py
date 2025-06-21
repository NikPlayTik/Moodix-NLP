
"""
Функциональное тестирование компонента 'destructive_flags' модели Moodix.

Перед запуском укажите пути к модели и токенизатору в переменных окружения:
MOODIX_MODEL и MOODIX_TOKENIZER
"""

import os
import pytest
from Start_AVG import SentimentAnalyzer

MODEL_PATH = os.getenv("MOODIX_MODEL", "model.keras")
TOKENIZER_PATH = os.getenv("MOODIX_TOKENIZER", "tokenizer.pickle")

@pytest.fixture(scope="session")
def analyzer():
    sa = SentimentAnalyzer(MODEL_PATH, TOKENIZER_PATH)
    sa.load_model()
    return sa


# Тест-кейсы: текст → ожидаемая активация флага (destructive_flag → True)
cases = [
    # obscene
    ("Ты чё, **б твою мать, охренел?", "obscene"),
    ("Какого хрена вы творите?!", "obscene"),

    # insult
    ("Ты тупой урод и не заслуживаешь уважения", "insult"),
    ("Этот человек — настоящая мразь", "insult"),

    # hate
    ("Я ненавижу всех геев и черных", "hate"),
    ("Нацисты были правы — одни евреи кругом", "hate"),

    # extremist
    ("Смерть государству! Взрывайте здания!", "extremist"),
    ("Мы должны объединиться и взять оружие против системы", "extremist"),

    # threat
    ("Я тебя найду и убью", "threat"),
    ("Если ты снова так скажешь — сломаю тебе ноги", "threat"),

    # destructive (общая)
    ("Разрежу себе вены, больше не хочу жить", "destructive"),
    ("Хочу умереть. Всё бессмысленно", "destructive"),

    # ложноположительные сценарии (не должны активировать флаги)
    ("Этот фильм ужасен", None),
    ("Мне не понравился новый закон", None),
    ("У него плохая дикция", None),
    ("Что за фигня с этой системой?", None),
    ("Никакого удовольствия, только разочарование", None),
    ("Печально, что сервис так и не улучшился", None),

    # граница по вероятности (слабые признаки)
    ("Может, я и не прав, но хочется всё разрушить", "destructive"),
    ("Ненавижу вставать рано", "hate"),
    ("Вы меня прям бесите", "insult"),

    # технические фразы (должны быть нейтральны)
    ("Установите драйвер для вашего устройства", None),
    ("Введите номер карты для оплаты", None),
    ("Ваш заказ подтверждён", None),

    # метафоры и фразы с потенциалом ошибки
    ("Этот баг — как ядерная бомба", None),
    ("Взорвало мозг от такой новости", None),
    ("Это просто убийственный дизайн", None),

    # нейтральные сильные эмоции (но не деструктивность)
    ("С ума сойти, какая красота!", None),
    ("Разрыдался от счастья", None),

    # пограничные угрозы
    ("Если не исправите — пожалеете", "threat"),
    ("Буду жаловаться во все инстанции!", None),

    # политический контекст
    ("Эти чиновники доведут страну до развала", "destructive"),
    ("Революция — единственный выход", "extremist")
]


@pytest.mark.parametrize("text,expected_flag", cases)
def test_destructive_flags(analyzer, text, expected_flag):
    result = analyzer.analyze_sentiment(text)
    flags = result["destructive"]["flags"]
    if expected_flag is None:
        assert not any(flags.values()), f"Не ожидалось деструктивных флагов: {flags}"
    else:
        assert flags.get(expected_flag, False), f"Ожидался флаг {expected_flag}, но получено: {flags}"
