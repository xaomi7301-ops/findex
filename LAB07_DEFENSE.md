# Лабораторна №7 — захист (5 хвилин)

Тема: пошук у вебі: FastAPI, Pydantic, ASGI. Версія 0.7.0.

## Демо
1. Відкрити https://findex-search-1n7g.onrender.com (після простою дочекатися пробудження).
2. Запитати python AND (asyncio OR thread), показати score, snippet,
   пагінацію та документ. Показати /docs, /stats, /health.
3. Показати README з реальною таблицею oha, пояснити p50/p95/p99.
4. Показати Render Logs та health check, Dockerfile non-root.

## Reflection
- Uvicorn приймає TCP/HTTP, передає ASGI scope/receive/send; Starlette
  виконує middleware/маршрутизацію, FastAPI Depends і Pydantic
  валідують параметри, відповідь серіалізується і йде до клієнта.
- ASGI підтримує async-обмін подіями й WebSocket; WSGI — синхронний
  контракт request/response. Async не прискорює CPU сам по собі.
- Query з k: int і ge/le створює Pydantic-схему. Некоректний k
  викликає 422 до виконання пошуку. Dataclass лишається всередині.
- Depends(get_index) повертає завантажений Index; dependency_overrides
  замінює його тестовим об'єктом, не читаючи справжній файл.
- BM25 та читання сніпетів синхронні; def FastAPI переносить в пул
  потоків. GIL може обмежувати CPU, але event loop лишається доступним.
  Чотири workers — процеси, кожен має свій індекс і GIL.
- Settings читає змінні оточення, Docker працює з тим самим кодом
  у dev/prod. Lifespan завантажує один раз на процес і закриває
  при SIGTERM. Health 503 означає, що індекс не готовий.
- p50 — медіана, p95/p99 — хвіст затримок; RPS — пропускна здатність.
  Висновки про різницю режимів роблю з фактичного oha, не вигадую.

## Локальні вимірювання (12.11.2026)

514 документів Python, AMD Ryzen 5 7430U, oha 1.16.0.
За 500 запитів concurrency=50:
- async def, 1 worker: 333.79 RPS, p99 448.29 мс.
- def, 1 worker: 320.78 RPS, p99 287.40 мс.
- def, 4 workers (повтор): 838.72 RPS, p99 155.12 мс,
  500/500 HTTP 200. Пропускна здатність у 2.61 раза більша.
- Початковий 4-worker тест мав аномальну затримку ~251 с;
  повтор із таймаутом 10 с завершився за 0.596 с.
  Причину аномалії не встановлено.

Залишаю def у production, оскільки синхронний пошук
працює в пулі потоків і не займає безпосередньо event loop.
## Docker та публічний сервіс

Локально Docker Engine 29.9.3: hello-world успішно, збірка
19/19 за 26.2 с, контейнер findex:0.7.0 запущено з INDEX_PATH,
health check 200. Render Frankfurt Free: перший Deploy succeeded
за 1 хв 5 с, /health 200. Публічна адреса:
https://findex-search-1n7g.onrender.com

Публічний oha: 200/200 HTTP 200, concurrency=20,
21.76 RPS, p50 901.72 мс, p95 1308.47 мс, p99 1497.41 мс,
wall time 9.193 с. Це не чисте порівняння CPU: мережа,
ресурси Render та інша конкурентність (локально 50).
У браузері запит python AND (asyncio OR thread) повернув
128 документів, перший Developing with asyncio, 12.566 мс
внутрішнього часу пошуку. Цей час не є end-to-end latency oha.
Безкоштовний інстанс засинає після простою; перед захистом
відкрити сторінку і перевірити /health.
