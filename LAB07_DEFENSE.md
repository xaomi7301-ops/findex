# Лабораторна №7 — захист (5 хвилин)

Тема: пошук у вебі: FastAPI, Pydantic, ASGI. Версія 0.7.0.

## Демо
1. Відкрити власну публічну адресу (додати після деплою).
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

Числа та URL доповнюються після реальних запусків.
