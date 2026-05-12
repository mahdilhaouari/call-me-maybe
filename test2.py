# test_sdk.py
import sys
sys.path.insert(0, '.')

try:
    from llm_sdk.llm_sdk import Small_LLM_Model
    print("✓ SDK imported successfully")
    
    # Initialize the model
    model = Small_LLM_Model()
    print("✓ Model initialized")
    
    # Test encode function
    tokens = model.encode("Hello world")
    print(f"✓ Encode test: 'Hello world' → {tokens[:5]}...")
    
    # Test decode function (if available)
    try:
        text = model.decode(tokens[:10])
        print(f"✓ Decode test: {tokens[:10]} → '{text}'")
    except:
        print("⚠ Decode method not available (optional)")
    
    # Test vocabulary path
    vocab_path = model.get_path_to_vocab_file()
    print(f"✓ Vocabulary path: {vocab_path}")
    
except Exception as e:
    print(f"✗ Error: {e}")
