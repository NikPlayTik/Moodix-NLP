# Moodix v0.65_31.05.2025

# 📌 1. Описание проекта

**Moodix** — это ***интеллектуальный модуль анализа текста***, разработанный с целью автоматического выявления ***эмоциональной окраски*** и ***признаков деструктивного поведения*** в русскоязычных сообщениях. 

Модуль представляет собой ***серверную систему***, основанную на нейросетевых технологиях, которая в режиме реального времени классифицирует тексты по:
1. основному настроению: ***позитивное, нейтральное, негативное***;
2. определение одной или нескольких из 16 категорий суб-настроений: ***восхищение, радость, гнев, волнение, благодарность, вдохновение, любовь, оптимизм, любопытство, информативность, осознание, раздражение, разочарование, отвращение, страх, грусть***;
3. определение наличия до 6 деструктивных маркеров: ***деструктивность, экстремизм, угроза, ненависть, непристойность, оскорбление***;

---
## ✅ 1.1 Проект актуален в условиях цифровой перегрузки и угроз информационной безопасности.
Современный пользовательский и государственный сектор сталкиваются с миллионами сообщений в мессенджерах, соцсетях и на форумах. Вручную отслеживать, фильтровать и оценивать такие потоки невозможно — и здесь на помощь приходит автоматизированная модель, как **Moodix**.

При этом **Moodix находит применение не только в сфере цифровой безопасности**. Он полезен:

- **Маркетологам и PR-отделам** — для анализа отзывов, публичных реакций, клиентской лояльности и быстрого реагирования на всплески негатива;
    
- **Аналитикам и UX-исследователям** — для выявления эмоционального восприятия сервисов, продуктов и интерфейсов;
    
- **Разработчикам NLP-систем и дата-сайентистам** — для автоматической разметки текстов и построения своих датасетов с многометочной классификацией;
    
- **Образовательным учреждениям** — как инструмент выявления агрессивных высказываний в школьной и студенческой среде.

---

## 🔴 1.2 Главные преимущества модуля заключаются в следующем: 
1. Полностью локальное развертывание (отсутствие отправки данных на внешние серверы, как в случае с IBM Watson или Google NLP);
2. Поддержка русского языка на уровне токенизации, лемматизации и эмбеддингов;
3. Расширенный спектр анализа: от базовой тональности до эмоций и маркеров агрессии;
4. Гибкость: Moodix может дообучаться на новых выборках, адаптируясь под специфические задачи;
5. Удобство встраивания: реализован как FastAPI-плагин, подходящий для интеграции в ИТ-системы органов власти, школ, модераторов и чат-ботов.

---
Система протестирована в рамках ***прикладных стресс-тестов на базе реальных сообщений***, собранных из открытых источников, в том числе с использованием эвристик и экспертной оценки. Moodix демонстрирует точность определения деструктивных признаков в 9 случаях из 10 и эмоциональной окраски — в 8 из 10.

Модель построена на архитектуре BiLSTM (двунаправленная долгосрочная память), с использованием технологий векторного кодирования, эмбеддингов, многоклассовой и многозадачной классификации.

---
# 📊 2. Как работает модель на практике

**Ниже будет расписан сам алгоритм обработки/предобработки текста:**
1. Предобработка текста (очистка, лемматизация);
2. Токенизация и преобразование в числовую форму;
3. Обработка в Embedding и BiLSTM-слоях;
4. Классификация и генерация JSON-ответа;
5. Визуализация CLI-результата пользователю.

---
## 🧪 2.1 Примеры работы модели

### ▶️ Пример входного текста: ***Свергнуть власть и убить всех***

**Классификация:** 
 - основное настроение: негативное - 100%;

**Суб-настроения** *(нулевые категории из-за недоразмеченных суб-настроений, некоторые текста в наборе данных были пустые): 
- восхищение, информативность, волнение;

**Деструктивные метки:** 
- деструктивность - 98.9%, экстремизм - 44.81%, ненависть - 33.24%, оскорбление - 69.31%.

Ниже будет представлен результат классификации входного текста в локальном-консольном (CLI) виде.
![Пример 1 — негатив, призыв к насилию](doc/Pasted%20image%2020250530150050.png)

---
### ▶️ Пример входного текста: ***Спасибо за помощь, вы лучшие***

**Классификация:** 
 - основное настроение: нейтральное - 99.97%;

**Суб-настроения**: 
- восхищение - 97.81%, радость - 0.99%, волнение - 0.39%, вдохновение - 0.37%;

**Деструктивные метки:** 
- все метки будут стремится к нулю.

Ниже будет представлен результат классификации входного текста в локальном-консольном (CLI) виде.
![Пример 2 — позитивное сообщение](doc/Pasted%20image%2020250530151511.png)

---
### ▶️ Пример входного текста: ***Средний сервис, но цены приемлемые***

**Классификация:** 
 - основное настроение: нейтральное - 100%;

**Суб-настроения**: 
- восхищение - 98.84%, раздражение - 1.81%, радость - 0.45%;

**Деструктивные метки:** 
- все метки будут стремится к нулю.

Ниже будет представлен результат классификации входного текста в локальном-консольном (CLI) виде.
![Пример 3 — нейтральное, с позитивным подтекстом](doc/Pasted%20image%2020250530151949.png)

---
### ▶️ Пример входного текста: ***Эти идиоты ничего не понимают***

**Классификация:** 
 - основное настроение: негативное - 100%;

**Суб-настроения** *(нулевые категории из-за недоразмеченных суб-настроений, некоторые текста в наборе данных были пустые): 
- раздражение - 0.91%;

**Деструктивные метки:** 
- деструктивность - 99.91%, ненависть - 80.28%, непристойность - 24.24%, оскорбление - 97.89%.

Ниже будет представлен результат классификации входного текста в локальном-консольном (CLI) виде.
![Пример 4 — оскорбительное сообщение](doc/Pasted%20image%2020250530152253.png)

---
### ▶️ Пример входного текста: ***Смартфон Samsung Galaxy A55 5G: переоцененный представитель А-серии***

**Классификация:** 
 - основное настроение: нейтральное - 98.12%;

**Суб-настроения**: 
- восхищение - 99.54%, радость - 14.46%;

**Деструктивные метки:** 
- все метки будут стремится к нулю.

Ниже будет представлен результат классификации входного текста в локальном-консольном (CLI) виде.
![Пример 5 — обзор, нейтральный стиль](doc/Pasted%20image%2020250530152645.png)

---
### ▶️ Пример входного текста: ***К огромному сожалению, для моих глаз обзор новых смартфонов на строке «сенсорный дисплей AMOLED» заканчивается. :(***

**Классификация:** 
 - основное настроение: нейтральное - 100%;

**Суб-настроения**: 
- восхищение - 12.94%, радость - 90.18%, раздражение - 3.86%, разочарование - 2.54%, оптимизм - 2.01;

**Деструктивные метки:** 
- все метки будут стремится к нулю.

Ниже будет представлен результат классификации входного текста в локальном-консольном (CLI) виде.
![Пример 6 — сожаление, эмоции](doc/Pasted%20image%2020250530152938.png)

---
# ⚙️ 3. Минимальные технические характеристики

1. **Процессор (CPU)** – одноядерный, с тактовой частотой не ниже 2.0 ГГц.
2. **ОЗУ (RAM)** – не менее 1 ГБ (рекомендуется от 2 ГБ для обеспечения стабильности и многократной обработки).
3. **ПЗУ (ROM)** – не менее 10 ГБ (включая модель, зависимости и виртуальную память).
4. **Swap-память (Swap-memory)** – минимум 2 ГБ, используемая в случае превышения объёма ОЗУ.

> 💡 В реальных тестах использовалась машина с процессором **Intel Core i5‑1135G7**, **8 ГБ ОЗУ**, **SSD 512 ГБ** и видеокартой **NVIDIA MX350 (2 ГБ)**, что обеспечило стабильную работу при параллельной обработке 1000 запросов с 20 одновременными запросами.


---
# 📦 4. Локальная установка и запуск модели

## 📥 4.1  Загрузка необходимых файлов

Скачайте следующие файлы и разместите их в одной директории (например, `~Moodix/Model/`): 
- `model.keras` — обученная нейросетевая модель;
- `tokenizer.pickle` — сериализованный токенизатор (словарь);
- `Start_AVG.py` — основной скрипт запуска (код см. ниже или скачайте готовый файл у разработчика).

---
## ⚙️ 4.2  Установка зависимостей

**Создайте и активируйте виртуальное окружение (опционально):**
```bash

python -m venv venv

source venv/bin/activate  # для Windows: venv\Scripts\activate

```

**Установите также библиотеки/зависимости:**
```bash

pip install tensorflow==2.15.0

pip install numpy scikit-learn nltk

pip install transformers==4.40.2

pip install torch==2.2.2

pip install langdetect==1.0.9

pip install sentencepiece

```

---
## 🚀 4.3.  Запуск анализатора

Создайте файл `run_moodix.py` с содержимым, данный скрипт уже имеет внутренний алгоритм присвоения оценки тональности большому по контексту текста (Например: негатив + нейтральное + негативное = негативное):
```python

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
    probs_agg = {k: np.mean([r['main']['probs'][k] for r in paragraph_results]) 
    
for k in main_sentiment_ru.keys()}
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
    model_path = ''
    tokenizer_path = ''
    
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
            print(f"  {k}: {v}%")
        
        print("\nСуб-настроения:")
        for cat, prob in agg['sub'].items():
            print(f"  {sub_sentiment_ru[cat]}: {prob}%")
        print("Топ-3 суб-настроения:", ", ".join(agg['top3_sub']))

        print("\nДеструктивность:")
        for cat, prob in agg['destructive']['probs'].items():
            flag = agg['destructive']['flags'][cat]
            print(f"  {destructive_ru[cat]}: {prob}% — {'Да' if flag else 'Нет'}")
        print("Топ-3 деструктивных метки:", ", ".join(agg['destructive']['top3']))

        print("\n" + "-"*40 + "\n")

if __name__ == "__main__":
    main()

```

**Предварительно в скрипте измените путь к модели и токенизатору:**
```python

model_path = 'путь/к/model.keras'

tokenizer_path = 'путь/к/tokenizer.pickle'

```
