
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

TOKEN = "your_token"

async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Halo! Aku bot sederhana. Ketik sesuatu untuk mulai ngobrol!")

async def echo(update: Update, context: CallbackContext):
    text_received = update.message.text
    await update.message.reply_text(f"Kamu berkata: {text_received}")

app = Application.builder().token(TOKEN).build()

app.add_handler(CommandHandler("start", start))

app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

print("Bot berjalan...")
app.run_polling()
