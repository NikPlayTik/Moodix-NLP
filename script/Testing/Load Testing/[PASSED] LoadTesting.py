import argparse
import random
import string
import time
import csv
import os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from Start import SentimentAnalyzer


def generate_random_text(min_words=3, max_words=15):
    word_count = random.randint(min_words, max_words)
    words = []
    for _ in range(word_count):
        word_length = random.randint(3, 10)
        word = ''.join(random.choices(string.ascii_letters + 'абвгдеёжзиклмнопрстуфхцчшщэюя', k=word_length))
        words.append(word)
    return ' '.join(words)


def run_single_test(analyzer, text):
    start_time = time.time()
    try:
        result = analyzer.analyze_sentiment(text)
        duration = time.time() - start_time
        return duration, result is not None, text
    except Exception:
        return None, False, text


def load_test(model_path, tokenizer_path, total_requests, concurrent_threads, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.csv")
    analyzer = SentimentAnalyzer(model_path, tokenizer_path)
    analyzer.load_model()

    durations = []
    errors = 0

    with open(log_path, mode="w", newline='', encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["№", "Успешно", "Время_ответа_сек", "Текст"])

        with ThreadPoolExecutor(max_workers=concurrent_threads) as executor:
            futures = []
            for i in range(total_requests):
                text = generate_random_text()
                futures.append(executor.submit(run_single_test, analyzer, text))

            for i, future in enumerate(as_completed(futures), 1):
                duration, success, text = future.result()
                if not success or duration is None:
                    errors += 1
                    writer.writerow([i, 0, "", text])
                else:
                    durations.append(duration)
                    writer.writerow([i, 1, f"{duration:.4f}", text])

    total_time = sum(durations)
    avg_time = total_time / len(durations) if durations else 0
    rps = len(durations) / total_time if total_time > 0 else 0

    print(f"\n=== Отчёт нагрузочного тестирования ===")
    print(f"Всего запросов: {total_requests}")
    print(f"Успешных: {len(durations)} | Ошибок: {errors}")
    print(f"Среднее время отклика: {avg_time:.4f} сек")
    print(f"RPS (запросов в секунду): {rps:.2f}")
    print(f"Общее время выполнения: {total_time:.2f} сек")
    print(f"Лог сохранён в: {log_path}")

    if durations:
        plot_path = os.path.join(log_dir, "response_times.png")
        plt.figure(figsize=(12, 5))
        plt.plot(durations, marker='o', linestyle='-', markersize=3)
        plt.title("Время отклика по каждому успешному запросу")
        plt.xlabel("№ запроса")
        plt.ylabel("Время (сек)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"График сохранён в: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Нагрузочное тестирование локального модуля Moodix")
    parser.add_argument("--model", type=str, required=True, help="Путь к .keras модели")
    parser.add_argument("--tokenizer", type=str, required=True, help="Путь к .pickle токенизатору")
    parser.add_argument("--requests", type=int, default=1000, help="Общее количество запросов")
    parser.add_argument("--threads", type=int, default=10, help="Количество параллельных потоков")
    parser.add_argument("--log_dir", type=str, default="./load_test_logs", help="Каталог для логов и графиков")

    args = parser.parse_args()

    load_test(args.model, args.tokenizer, args.requests, args.threads, args.log_dir)