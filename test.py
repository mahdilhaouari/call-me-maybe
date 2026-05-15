"""Test the 4 SDK functions."""

from llm_sdk.llm_sdk import Small_LLM_Model

def main():
    print("=" * 50)
    print("Creating model (first time will download 1.2GB)...")
    print("This may take 5-15 minutes")
    print("=" * 50)
    
    # This will download the Qwen model
    model = Small_LLM_Model()
    
    print("\n✅ Model loaded successfully!\n")
    
    # TEST 1: encode
    print("--- TEST 1: encode() ---")
    text = "what is the mum of 2 plus 2 "
    result = model.encode(text)
    print(f"Input: '{text}'")
    print(f"Output shape: {result.shape}")
    token_ids = result[0].tolist()
    print(f"Token IDs: {token_ids}\n")
    
    # TEST 2: decode
    print("--- TEST 2: decode() ---")
    decoded = model.decode(token_ids)
    print(f"Input token IDs: {token_ids}")
    print(f"Decoded text: '{decoded}'\n")
    
    # TEST 3: get_logits_from_input_ids
    print("--- TEST 3: get_logits_from_input_ids() ---")
    input_text = "what is the sum of 2 plus 2 "
    input_ids = model.encode(input_text)[0].tolist()
    print(f"Input text: '{input_text}'")
    
    logits = model.get_logits_from_input_ids(input_ids)
    print(f"Number of possible next tokens: {len(logits)}")
    
    # Find most likely next token
    import numpy as np
    top_token_id = np.argmax(logits)
    top_word = model.decode([top_token_id])
    print(f"AI predicts next word: '{top_word}'\n")
    
    # TEST 4: get_path_to_vocab_file
    print("--- TEST 4: get_path_to_vocab_file() ---")
    vocab_path = model.get_path_to_vocab_file()
    print(f"Vocabulary file location: {vocab_path}\n")
    
    print("=" * 50)
    print("✅ All tests passed!")
    print("=" * 50)

if __name__ == "__main__":
    main()