# ## Spacy
#
# # import spacy
# # text = 'I am Dheeraj. My name is Dheeraj'
# #
# # nlp = spacy.load('en_core_web_sm')
# # doc = nlp(text)
# #
# # token_list = [token.text for token in doc]
# # print('Tokens:', token_list)
#
#
# ## NLTK
#
# # import nltk
# # # nltk.download('punkt')
# # # nltk.download('punkt_tab')
# # from nltk.tokenize import word_tokenize
# #
# # text = 'Unicorns are real. I saw a unicorn yesterday. I couldnt see it today'
# #
# # token = word_tokenize(text)
# # print(token)
#
#
# ### Tokenizers Learn
#
# # import nltk
#
# # from nltk.tokenize import word_tokenize
# #
# text = 'Dhoni is goat'
# # print(word_tokenize(text))
#
# # import spacy
# # nlp = spacy.load('en_core_web_sm')
# # doc = nlp(text)
# # token_list = [token.text for token in doc]
# # print("Tokens:", token_list)
# #
# # # Showing token details
# # for token in doc:
# #     print(token.text, token.pos_, token.dep_)
#
# # import spacy
# # nlp = spacy.load('en_core_web_sm')
# # doc = nlp(text)
# # token_list = [token.text for token in doc]
# # print('Tokens:', token_list)
# #
# # for token in doc:
# #     print(token.text, token.pos_, token.dep_)
#
#
# from transformers import BertTokenizer
#
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# print(tokenizer.tokenize(text))
#
#
# from transformers import XLNetTokenizer
#
# tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
# tokenizer.tokenize("IBM taught me tokenization.")
#
#
# for torchtext.data.utils import get tokenizer

text = 'Hi everyone, my name is Dheeraj. Everyone here can call me Dheeraj or Dhee'

# from nltk import word_tokenize
#
# print(word_tokenize(line))

# import spacy
#
# nlp = spacy.load('en_core_web_sm')
# doc = nlp(line)
#
# token_list = [token.text for token in doc]
# print('Tokens:', token_list)
#
# for token in doc:
#     print(token.text, token.pos_, token.dep_)

# from transformers import BertTokenizer
# from transformers import XLNetTokenizer
#
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# print(tokenizer.tokenize(text))
#
# tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
# print(tokenizer.tokenize(text))

dataset = [
    (1,"Introduction to NLP"),
    (2,"Basics of PyTorch"),
    (1,"NLP Techniques for Text Classification"),
    (3,"Named Entity Recognition with PyTorch"),
    (3,"Sentiment Analysis using PyTorch"),
    (3,"Machine Translation with PyTorch"),
    (1," NLP Named Entity,Sentiment Analysis,Machine Translation "),
    (1," Machine Translation with NLP "),
    (1," Named Entity vs Sentiment Analysis  NLP ")]

from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer('basic_english')

from torchtext.vocab import build_vocab_from_iterator
def yield_tokens(data_iter):
    for _,text in data_iter:
        yield tokenizer(text)

my_iterator = yield_tokens(dataset)

print(next(my_iterator))
print(next(my_iterator))

vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

def get_tokenized_sentence_and_indices(iterator):
    tokenized_sentence = next(iterator)
    token_indices = [vocab[token] for token in tokenized_sentence]

    return tokenized_sentence, token_indices

tokenized_sentence, token_indices = get_tokenized_sentence_and_indices(my_iterator)

print(tokenized_sentence)
print(token_indices)


lines = ["IBM taught me tokenization",
         "Special tokenizers are ready and they will blow your mind",
         "just saying hi!"]


special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

tokenizer_en = get_tokenizer('spacy', language = 'en_core_web_sm')

tokens = []

max_length = 0

for line in lines:
    tokenized_line = tokenizer_en(line)
    tokenized_line = ['<bos>'] + tokenized_line + ['<eos>']
    tokens.append(tokenized_line)
    max_length = max(max_length, len(tokenized_line))

for i in range(len(tokens)):
    tokens[i] = tokens[i] + ['<pad>'] * (max_length-len(tokens[i]))


print(tokens)

# Build vocabulary without unk_init
vocab = build_vocab_from_iterator(tokens, specials=['<unk>'])
vocab.set_default_index(vocab["<unk>"])

# Vocabulary and Token Ids
print("Vocabulary:", vocab.get_itos())
print("Token IDs for 'tokenization':", vocab.get_stoi())


