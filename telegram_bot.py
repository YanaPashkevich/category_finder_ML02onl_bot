#!/usr/bin/env python3
from __future__ import annotations

"""
Telegram-бот для категоризации товаров по названиям.
Отправьте CSV-файл с колонкой "title" — получите CSV с предсказанными категориями.
Токен задаётся переменной окружения TELEGRAM_BOT_TOKEN.
"""
import io
import os
import csv
import logging
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# Модуль инференса из этого же проекта
from inference import load_artifacts, predict_smart

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Лимиты
MAX_FILE_SIZE_MB = 15
MAX_ROWS = 5000
MODELS_DIR = Path(__file__).resolve().parent / "models"

# Глобально загружаем модель один раз при старте
_model = None
_id2label = None
_seen_titles = None

# Пул потоков для выполнения синхронных операций
_executor = ThreadPoolExecutor(max_workers=1)


def get_model():
    global _model, _id2label, _seen_titles
    if _model is None:
        _model, _id2label, _, _seen_titles = load_artifacts(str(MODELS_DIR))
    return _model, _id2label, _seen_titles


def parse_csv_titles(content: bytes, filename: str) -> tuple[list[str], str | None]:
    """
    Парсит CSV, ищет колонку 'title' или берёт первую колонку.
    Возвращает (список названий, ошибка или None).
    """
    for encoding in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        return [], "Кодировка файла не распознана (нужен UTF-8 или CP1251)."

    try:
        # Пробуем разделитель ; или ,
        for sep in (";", ","):
            reader = csv.DictReader(io.StringIO(text), delimiter=sep)
            if not reader.fieldnames:
                continue
            fieldnames = [f.strip() for f in reader.fieldnames]
            # Ищем колонку title (без учёта регистра)
            title_col = None
            for col in fieldnames:
                if col.lower() == "title" or col.lower() == "название":
                    title_col = col
                    break
            if title_col is None:
                title_col = fieldnames[0]

            rows = list(reader)
            titles = []
            for row in rows:
                val = row.get(title_col) or row.get(fieldnames[0])
                if val is not None and str(val).strip():
                    titles.append(str(val).strip())
            if titles:
                return titles, None
    except Exception as e:
        return [], f"Ошибка разбора CSV: {e}"

    return [], "В файле не найдена колонка 'title' или первая колонка пуста."


def build_result_csv(results: list[dict]) -> bytes:
    """Собирает CSV из результатов предсказания. Возвращает байты в UTF-8."""
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    writer.writerow(["title", "pred_status", "pred_category", "pred_message"])
    for r in results:
        writer.writerow([
            r["title"],
            r["status"],
            r.get("category") or "",
            r.get("message") or "",
        ])
    # Возвращаем байты в UTF-8 без BOM для быстрого открытия
    return buf.getvalue().encode('utf-8')


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"Received /start command from user {update.effective_user.id}")
    await update.message.reply_text(
        "Привет. Я категоризатор товаров (раздел «Электрика»).\n\n"
        "Отправьте CSV-файл с колонкой «title» (или «название»). "
        "Я верну CSV с колонками: title, pred_status, pred_category, pred_message.\n\n"
        f"Ограничения: размер файла до {MAX_FILE_SIZE_MB} МБ, до {MAX_ROWS} строк."
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"Received document from user {update.effective_user.id}")
    doc = update.message.document
    if not doc:
        logger.warning("No document in update.message")
        return

    file_size_mb = (doc.file_size or 0) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        await update.message.reply_text(
            f"Файл слишком большой ({file_size_mb:.1f} МБ). Максимум {MAX_FILE_SIZE_MB} МБ."
        )
        return

    name = (doc.file_name or "").lower()
    if not name.endswith(".csv"):
        await update.message.reply_text(
            "Нужен файл с расширением .csv. Отправьте CSV с колонкой «title»."
        )
        return

    await update.message.reply_text("Скачиваю файл…")

    try:
        file = await context.bot.get_file(doc.file_id)
        content = await file.download_as_bytearray()
    except Exception as e:
        logger.exception("Download failed")
        await update.message.reply_text(f"Не удалось скачать файл: {e}")
        return

    titles, err = parse_csv_titles(bytes(content), doc.file_name or "")
    if err:
        await update.message.reply_text(err)
        return

    if not titles:
        await update.message.reply_text("В файле нет строк с названиями.")
        return

    if len(titles) > MAX_ROWS:
        await update.message.reply_text(
            f"Слишком много строк ({len(titles)}). Максимум {MAX_ROWS}. Обрежьте файл."
        )
        return

    await update.message.reply_text(
        f"Обрабатываю {len(titles)} названий…"
    )

    try:
        logger.info(f"Starting prediction for {len(titles)} titles")
        model, id2label, seen_titles = get_model()
        
        # Выполняем предсказание в отдельном потоке, чтобы не блокировать event loop
        def run_prediction():
            return predict_smart(
                titles,
                model,
                id2label,
                margin_threshold=0.25,
                proba_threshold=0.55,
                seen_titles=seen_titles,
            )
        
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(_executor, run_prediction)
        logger.info(f"Prediction completed: {len(results)} results")
        csv_out = build_result_csv(results)
        logger.info("CSV built successfully")
    except Exception as e:
        logger.exception("Prediction failed")
        await update.message.reply_text(f"Ошибка при категоризации: {e}")
        return

    out_name = "categorized_" + (doc.file_name or "result.csv")
    if not out_name.endswith(".csv"):
        out_name += ".csv"

    try:
        logger.info(f"Sending result document: {out_name} ({len(csv_out)} bytes)")
        # Используем уже закодированные байты (UTF-8 без BOM для быстрого открытия)
        await update.message.reply_document(
            document=io.BytesIO(csv_out),
            filename=out_name,
            caption=f"Готово: {len(results)} строк. OK: {sum(1 for r in results if r['status'] == 'OK')}, без категории: {sum(1 for r in results if r['status'] != 'OK')}.",
        )
        logger.info("Document sent successfully")
    except Exception as e:
        logger.exception("Failed to send document")
        await update.message.reply_text(f"Ошибка при отправке результата: {e}")


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "Задайте переменную окружения TELEGRAM_BOT_TOKEN.\n"
            "Пример: export TELEGRAM_BOT_TOKEN='8160250779:AAF...'"
        )

    if not MODELS_DIR.is_dir() or not (MODELS_DIR / "model.joblib").exists():
        raise SystemExit(
            f"Папка с моделью не найдена: {MODELS_DIR}\n"
            "Сначала обучите модель в ноутбуке и сохраните артефакты в models/."
        )

    # Предзагрузка модели
    get_model()
    logger.info("Model loaded.")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    logger.info("Bot started. Waiting for updates...")
    logger.info("Send /start to your bot in Telegram to test it.")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
