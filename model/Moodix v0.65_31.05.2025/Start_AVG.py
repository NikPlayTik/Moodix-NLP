import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import re
import os

main_sentiment_ru = {
    "positive": "Позитивное",
    "neutrally": "Нейтральное",
    "negative": "Негативное"
}

sub_sentiment_ru = {
    "admiration": "восхищение",
    "excitement": "волнение",
    "gratitude": "благодарность",
    "inspiration": "вдохновение",
    "joy": "радость",
    "love": "любовь",
    "optimism": "оптимизм",
    "curiosity": "любопытство",
    "informative": "информативность",
    "realization": "осознание",
    "anger": "гнев",
    "annoyance": "раздражение",
    "disappointment": "разочарование",
    "disgust": "отвращение",
    "fear": "страх",
    "sadness": "грусть"
}

destructive_ru = {
    "destructive": "деструктивность",
    "extremist": "экстремизм",
    "threat": "угроза",
    "hate": "ненависть",
    "obscene": "непристойность",
    "insult": "оскорбление"
}

def split_into_paragraphs(text):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    out = []
    for p in paragraphs:
        if len(p) > 300:
            sentences = re.split(r'(?<=[.!?…])\s+', p)
            buf = ""
            for s in sentences:
                if len(buf) + len(s) < 300:
                    buf += (s + " ")
                else:
                    out.append(buf.strip())
                    buf = s + " "
            if buf.strip():
                out.append(buf.strip())
        else:
            out.append(p)
    return out

class SentimentAnalyzer:
    def __init__(self, model_path, tokenizer_path, maxlen=50):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.maxlen = maxlen
        self.sentiment_mapping = {0: 'negative', 1: 'neutrally', 2: 'positive'}
        self.sub_categories = list(sub_sentiment_ru.keys())
        self.destructive_categories = list(destructive_ru.keys())

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(f"Токенизатор не найден: {self.tokenizer_path}")
        self.model = load_model(self.model_path)
        with open(self.tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        print("Модель и токенизатор успешно загружены")

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^а-яёa-z0-9\s!?]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def analyze_sentiment(self, text):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Модель или токенизатор не загружены")
        cleaned = self.clean_text(text)
        seq = self.tokenizer.texts_to_sequences([cleaned])
        if not seq or not seq[0]:
            return None
        padded = pad_sequences(seq, maxlen=self.maxlen, padding='post')
        pred_main, pred_sub, pred_destr = self.model.predict(padded, verbose=0)
        idx = int(np.argmax(pred_main[0]))
        main_label = self.sentiment_mapping[idx]
        confidence = float(pred_main[0][idx])
        sub_probs = {
            cat: round(float(prob) * 100, 2)
            for cat, prob in zip(self.sub_categories, pred_sub[0])
        }
        destr_probs = {
            cat: round(float(prob) * 100, 2)
            for cat, prob in zip(self.destructive_categories, pred_destr[0])
        }
        destr_flags = {
            cat: (p > 50) for cat, p in destr_probs.items()
        }
        return {
            'main': {
                'label': main_label,
                'label_ru': main_sentiment_ru[main_label],
                'confidence': round(confidence * 100, 2),
                'probs': {
                    'negative': round(float(pred_main[0][0]) * 100, 2),
                    'neutrally': round(float(pred_main[0][1]) * 100, 2),
                    'positive': round(float(pred_main[0][2]) * 100, 2),
                }
            },
            'sub': sub_probs,
            'destructive': {
                'probs': destr_probs,
                'flags': destr_flags
            }
        }

    def print_model_summary(self):
        if self.model:
            self.model.summary()
        else:
            print("Модель не загружена.")

def aggregate_results(paragraph_results):
    main_labels = [r['main']['label'] for r in paragraph_results]
    main_count = {l: main_labels.count(l) for l in main_sentiment_ru.keys()}
    agg_main = max(main_count, key=main_count.get)
    if list(main_count.values()).count(max(main_count.values())) > 1:
        if main_count['neutrally'] == max(main_count.values()):
            agg_main = 'neutrally'
        elif main_count['positive'] == max(main_count.values()):
            agg_main = 'positive'
        else:
            agg_main = 'negative'
    probs_agg = {k: np.mean([r['main']['probs'][k] for r in paragraph_results]) for k in main_sentiment_ru.keys()}
    confidence = probs_agg[agg_main]
    sub_agg = {}
    for cat in sub_sentiment_ru:
        vals = [r['sub'][cat] for r in paragraph_results]
        sub_agg[cat] = round(float(np.mean(vals)), 2)
    top3_sub = sorted(sub_agg.items(), key=lambda x: x[1], reverse=True)[:3]
    destr_agg = {}
    destr_flag = {}
    for cat in destructive_ru:
        vals = [r['destructive']['probs'][cat] for r in paragraph_results]
        destr_agg[cat] = round(float(np.mean(vals)), 2)
        destr_flag[cat] = destr_agg[cat] > 50
    top3_destr = sorted(destr_agg.items(), key=lambda x: x[1], reverse=True)[:3]
    return {
        'main': {
            'label': agg_main,
            'label_ru': main_sentiment_ru[agg_main],
            'confidence': round(confidence, 2),
            'probs': {main_sentiment_ru[k]: round(v, 2) for k, v in probs_agg.items()}
        },
        'sub': sub_agg,
        'top3_sub': [sub_sentiment_ru[c] for c, _ in top3_sub],
        'destructive': {
            'probs': destr_agg,
            'flags': destr_flag,
            'top3': [destructive_ru[c] for c, _ in top3_destr]
        }
    }

def main():
    model_path = 'D:/Laboratary and practical/!Диплом/Model/15 Moodix v0.65_31.05.2025/model.keras'
    tokenizer_path = 'D:/Laboratary and practical/!Диплом/Model/15 Moodix v0.65_31.05.2025/tokenizer.pickle'

    analyzer = SentimentAnalyzer(model_path, tokenizer_path, maxlen=50)
    try:
        analyzer.load_model()
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    analyzer.print_model_summary()

    print("\nАнализатор запущен. Для выхода введите 'exit'.\n")
    print("Основные настроения:")
    for v in main_sentiment_ru.values():
        print(f"- {v}")
    print("\nСуб-настроения:")
    for cat in analyzer.sub_categories:
        print(f"- {sub_sentiment_ru[cat]}")
    print("\nДеструктивные метки:")
    for cat in analyzer.destructive_categories:
        print(f"- {destructive_ru[cat]}")
    print("\n")

    while True:
        user_input = input("Введите текст для анализа: ").strip()
        if user_input.lower() in ('exit', 'quit', 'e'):
            print("Выход.")
            break
        if not user_input:
            print("Пустой ввод, повторите попытку.")
            continue

        paragraphs = split_into_paragraphs(user_input)
        if len(paragraphs) == 1:
            result = analyzer.analyze_sentiment(user_input)
            paragraph_results = [result]
        else:
            paragraph_results = []
            for p in paragraphs:
                r = analyzer.analyze_sentiment(p)
                paragraph_results.append(r)

        agg = aggregate_results(paragraph_results)
        main = agg['main']
        print(f"\nОсновное настроение: {main['label_ru']} ({main['confidence']}%)")
        print("Вероятности основного:")
        for k, v in main['probs'].items():
            print(f"  {k}: {v}%")
        print("\nСуб-настроения:")
        for cat, prob in agg['sub'].items():
            print(f"  {sub_sentiment_ru[cat]}: {prob}%")
        print("Топ-3 суб-настроения:", ", ".join(agg['top3_sub']))
        print("\nДеструктивность:")
        for cat, prob in agg['destructive']['probs'].items():
            flag = agg['destructive']['flags'][cat]
            print(f"  {destructive_ru[cat]}: {prob}% — {'Да' if flag else 'Нет'}")
        print("Топ-3 деструктивных метки:", ", ".join(agg['destructive']['top3']))
        print("\n" + "-"*40 + "\n")

if __name__ == "__main__":
    main()