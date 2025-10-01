from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = 'google/flan-t5-small'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def chat_bot():
    while True:
        input_text = input('you:')
        if input_text.lower() in ['bye','exit','stop']:
            print('Bye!')
            break

        inputs = tokenizer.encode(input_text, return_tensors='pt')
        outputs = model.generate(inputs, max_new_tokens=150)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print('chatbot response:', response)


chat_bot()