from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import random
import json
from tqdm.auto import tqdm


training_dataset = load_dataset('stanfordnlp/imdb', split='train')
testing_dataset = load_dataset('stanfordnlp/imdb', split='test')

our_model = AutoModelForCausalLM.from_pretrained('./fine-tuned-gpt2-large')
our_tokenizer = AutoTokenizer.from_pretrained('./fine-tuned-gpt2-large')


sentiment_tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")

print("Models loaded.")

NUM_PROMPTS = 1000
NUM_SAMPLES_PER_PROMPT = 4
OUTPUT_FILE = "generations.json"

generate_kwargs = {
    "max_new_tokens": 200,
    "do_sample": True,
    "temperature": 0.2,
}


print("Loading dataset...")
prefix_dataset = load_dataset('stanfordnlp/imdb', split='train')

prompt_data = []
print("Tokenizing prompts...")
for index in range(NUM_PROMPTS): 
    data = prefix_dataset[index]
    prefix_length = random.randint(2, 8)
    prompt_text = " ".join(data['text'].split()[:prefix_length])
    
    tokens = our_tokenizer(prompt_text, return_tensors="pt")
    
    prompt_data.append({
        "prompt_text": prompt_text,
        "tokenized_inputs": tokens,
        "original_label": data['label']
    })

generation_params = generate_kwargs.copy()
generation_params['num_return_sequences'] = NUM_SAMPLES_PER_PROMPT

if "pad_token_id" not in generation_params:
    generation_params['pad_token_id'] = our_tokenizer.eos_token_id

json_output_data = []

try:
    print("Generating samples...")
    for data in tqdm(prompt_data):
        inputs = data["tokenized_inputs"]
        prompt_text = data["prompt_text"]
        
        
        try:
            generated_sequences = our_model.generate(
                **inputs,
                **generation_params
            )
            
            decoded_samples = []
            input_length = inputs["input_ids"].shape[1]
            
            for seq in generated_sequences:
                generated_tokens_only = seq[input_length:]
                
                decoded_text = our_tokenizer.decode(
                    generated_tokens_only,
                    skip_special_tokens=True
                )
                decoded_samples.append(decoded_text.strip())
            
            json_output_data.append({
                "prompt": prompt_text,
                "generations": decoded_samples,
            })
    
        except Exception as e:
            print(f"Error generating for prompt: '{prompt_text}'. Error: {e}")

except KeyboardInterrupt:
    print("\n--- Generation interrupted by user ---")

finally:
    print("\n--- Generation Complete or Interrupted ---")
    print(f"Total prompts processed: {len(json_output_data)}")
    
    if not json_output_data:
        print("No results to save.")
    else:
        print(f"Saving {len(json_output_data)} results to {OUTPUT_FILE}...")
        try:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(json_output_data, f, indent=4)
            print("Successfully saved to JSON.")
            
            if json_output_data:
                print("\nExample of first item saved:")
                print(json.dumps(json_output_data[0], indent=2))
        
        except Exception as e:
            print(f"Error saving to JSON file: {e}")
