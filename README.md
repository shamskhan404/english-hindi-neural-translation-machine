# English–Hindi Neural Translation Model

A custom **Transformer-based Neural Machine Translation system** for translating English sentences into Hindi.

This project implements a **from-scratch Transformer architecture** with advanced attention mechanisms and is deployed as an interactive translation demo.

---

## Demo

Try the live translator here:

English → Hindi Translation  
https://huggingface.co/spaces/ShamsKhan404/English-Hindi_Neural_Translation_Model

---

## Model Architecture

The translation model is built from scratch using PyTorch and includes:

- Transformer encoder–decoder architecture
- Multi-head coupled attention
- Rotary positional embeddings (RoPE)
- SentencePiece tokenizer
- Beam search decoding
- Language embeddings

### Encoder
- 8 transformer layers
- Coupled attention mechanism
- Feed-forward GLU layers

### Decoder
- 6 transformer layers
- Cross-attention with encoder outputs
- Beam search generation

---

## Key Features

- Custom transformer implementation
- Beam search decoding
- English → Hindi translation
- Interactive Gradio UI
- Batch translation support
- Translation history
- Hugging Face Spaces deployment

---

## Example Translation

Input:

Hello, how are you?

Output:

नमस्ते, आप कैसे हैं?

---

## Technologies Used

- Python
- PyTorch
- SentencePiece
- Gradio
- Hugging Face Spaces

---

## Future Improvements

- English ↔ Hindi bidirectional translation
- Attention visualization
- Faster decoding
- Larger multilingual training dataset
- Transformer scaling

---

## License

MIT License
