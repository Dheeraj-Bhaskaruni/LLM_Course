from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = 'facebook/blenderbot-400M-distill'

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def chat_with_bot():
    while True:

        input_user = input('User:')

        if input_user.lower() in ['break', 'quit', 'exit']:
            print('Goodbye user!')
            break

        inputs = tokenizer.encode(input_user, return_tensors='pt')
        outputs = model.generate(inputs, max_new_tokens = 150)

        response = tokenizer.decode(outputs[0], skip_special_tokens = True).strip()

        print('Chatbot', response)


chat_with_bot()



