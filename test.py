from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

tokenizer = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def chat_with_bot():
    while True:
        input_text = input("You")

        if input_text.lower() in ['quit', 'exit','bye']:
            print('Bye')
            break

        inputs = token