# %%

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

# %% load tokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer

# %% explore vocabulary terms

vocab = tokenizer.get_vocab()
print(f"Total number of tokens in vocabulary: {len(vocab)} \n---------")

# "G" is an encoded whitespace
items = list(vocab.items())
for word, idx in random.sample(items, 10):
    print(word, idx)
    
# %% tokenize text inputs

text = "stephen is nice"

encoded_input1 = tokenizer(text, return_tensors='pt')

print("Tokens:")
temp_tokens = encoded_input1["input_ids"][0]
print(tokenizer.convert_ids_to_tokens(temp_tokens))
print("\n------------------------------------------\n")
print("Token IDs:")
print(temp_tokens)

# %% download pretrained model

model = AutoModelForCausalLM.from_pretrained('gpt2')
print(model.config)

# %% examine input text encoding

text = "After a season of positive corporate earnings announcements driven by AI adoption, NVIDIA's share price hit an all-time"

input_ids = tokenizer(text, return_tensors='pt')["input_ids"]

with torch.no_grad():
    output = model(input_ids, output_hidden_states=True)

print(f"number of network layers: {len(output.hidden_states)}")

last_hidden_state = output.hidden_states[-1]

print(f"shape of last hidden layer: {last_hidden_state.shape}")

# %% examine distribution over next word

print(f"shape of logits in output: {output.logits.shape}")

next_token_logits = output.logits[0, -1, :] # logits at last token
probs = torch.softmax(next_token_logits, dim=-1) # convert to distribution

top_probs, top_ids = torch.topk(probs, 10)

print(f"Prompt: {text!r}\n")
print("Top 10 next-token candidates:")
for prob, idx in zip(top_probs, top_ids):
    token = tokenizer.decode(idx)
    print(f"  {token!r:15} {prob.item():.4f}")

# %% more diffuse distribution

text = "I am"

input_ids = tokenizer(text, return_tensors='pt')["input_ids"]

with torch.no_grad():
    output = model(input_ids, output_hidden_states=True)
    
next_token_logits = output.logits[0, -1, :] # logits at last token
probs = torch.softmax(next_token_logits, dim=-1) # convert to distribution

top_probs, top_ids = torch.topk(probs, 10)

print(f"Prompt: {text!r}\n")
print("Top 10 next-token candidates:")
for prob, idx in zip(top_probs, top_ids):
    token = tokenizer.decode(idx)
    print(f"  {token!r:15} {prob.item():.4f}")

# %% text generation (run multiple times for different continuations)

text = "I am"
input_ids = tokenizer(text, return_tensors='pt')["input_ids"]

max_new_tokens = 40

with torch.no_grad():
    for _ in range(max_new_tokens):
        output = model(input_ids)
        next_token_logits = output.logits[0, -1, :]
        temp = 1.0 # temperature
        probs = torch.softmax(next_token_logits / temp, dim=-1)

        # sample token from the multinomial distribution
        next_id = torch.multinomial(probs, num_samples=1)

        # stop if we drew end-of-text
        if next_id.item() == tokenizer.eos_token_id:
            break

        # append and feed the longer sequence back in
        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)

print(tokenizer.decode(input_ids[0]))

# %% bias in raw word prediction

def next_token_prob(prompt, word):
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        logits = model(ids).logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    tok_ids = tokenizer.encode(" " + word)   # leading space: follows "a "
    return probs[tok_ids[0]].item()           # first sub-token proxy

occupations = ["nurse", "engineer", "secretary", "doctor",
               "teacher", "scientist", "cleaner", "programmer"]

man_prompt   = "The man worked as a"
woman_prompt = "The woman worked as a"

print(f"{'occupation':12} {'P(man)':>10} {'P(woman)':>10} {'man/woman':>11}")
results = []
for w in occupations:
    pm = next_token_prob(man_prompt, w)
    pw = next_token_prob(woman_prompt, w)
    results.append((w, pm, pw, pm / pw))

# sort most male-skewed at top, most female-skewed at bottom
for w, pm, pw, ratio in sorted(results, key=lambda r: -r[3]):
    print(f"{w:12} {pm:10.5f} {pw:10.5f} {ratio:11.2f}")

# %%
