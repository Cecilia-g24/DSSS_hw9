import asyncio
import nest_asyncio

import torch
from transformers import pipeline

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
nest_asyncio.apply()

from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the model
model_name = "prithivMLmods/Llama-Deepsync-1B" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Hello! Let's chat！")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    inputs = tokenizer(user_message, return_tensors="pt", padding=True)

    # Generate a response
    outputs = model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Display the response
    #print(f"Bot: {response}")
   
    await update.message.reply_text(f"respond: {response}")

# model with ai
def main():
    TOKEN = "7601158234:AAFJtpitbJRuIkHY1uUfZzk7S1KO_kRvYUk"
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo)
    application.run_polling()
    print("Bot 已启动！按 Ctrl+C 停止。")

if __name__ == "__main__":
    main()