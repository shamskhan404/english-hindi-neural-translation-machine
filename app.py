import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import gradio as gr
import sentencepiece as spm
import os

# ===================== CONFIGURATION =====================
CHECKPOINT_PATH = "model.pth"
TOKENIZER_MODEL_PATH = "tokenizer.model"

# Model architecture (must match training)
D_MODEL = 768
D_K = 768
FF_DIM = 4096
NUM_ENCODER_LAYERS = 8
NUM_DECODER_LAYERS = 6
SEQ_LEN = 256
NUM_HEADS = 12
NUM_LANGUAGES = 2
LANG_EMBED_DIM = 768

# Generation parameters
MAX_GENERATION_LENGTH = 150
NUM_BEAMS = 4
LENGTH_PENALTY = 1.0
TEMPERATURE = 0.8
REPETITION_PENALTY = 1.1

# Special tokens (must match training)
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
EN_ID = 4
HI_ID = 5
lang_map = {"en": 0, "hi": 1}

# ===================== TOKENIZER =====================
class SentencePieceUnigramTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.pad_id = PAD_ID
        self.unk_id = UNK_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID

    def encode(self, text, add_bos=True, add_eos=True):
        return self.sp.encode_as_ids(text, add_bos=add_bos, add_eos=add_eos)

    def decode(self, ids):
        return self.sp.decode_ids(ids)

    def get_vocab_size(self):
        return self.sp.get_piece_size()

# ===================== MODEL ARCHITECTURE =====================
class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding"""
    def __init__(self, dim, max_seq_len=5000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len):
        t = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotate half of the dimensions"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(q, k, cos, sin):
    """Apply rotary embeddings to queries and keys"""
    q_embed = (q * cos.unsqueeze(0)) + (rotate_half(q) * sin.unsqueeze(0))
    k_embed = (k * cos.unsqueeze(0)) + (rotate_half(k) * sin.unsqueeze(0))
    return q_embed, k_embed

class MultiHeadCoupledAttention(nn.Module):
    """Multi-Head Coupled Attention (without V projection)"""
    def __init__(self, d_model, d_k, num_heads=12, is_cross_attention=False):
        super().__init__()
        assert d_k % num_heads == 0, "d_k must be divisible by num_heads"

        self.d_model = d_model
        self.d_k = d_k
        self.num_heads = num_heads
        self.head_dim = d_k // num_heads
        self.is_cross_attention = is_cross_attention

        self.qw_proj = nn.Linear(d_model, d_k, bias=False)
        self.kw_proj = nn.Linear(d_model, d_k, bias=False)
        self.qp_proj = nn.Linear(d_model, d_k, bias=False)
        self.kp_proj = nn.Linear(d_model, d_k, bias=False)

        self.out_proj = nn.Linear(d_k, d_model, bias=False)

        self.attn_dropout = nn.Dropout(0.1)

        if not is_cross_attention:
            self.rotary = RotaryEmbedding(self.head_dim)

    def split_heads(self, x):
        batch_size, seq_len = x.size()[:2]
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        batch_size, num_heads, seq_len, head_dim = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, num_heads * head_dim)

    def forward(self, query, key=None, value=None, mask=None, encoder_output=None):
        if encoder_output is not None:
            key = key if key is not None else encoder_output
            value = encoder_output
        elif key is None or value is None:
            key = query
            value = query

        B, L, _ = query.size()
        B, S, _ = key.size()

        qw = self.qw_proj(query)
        kw = self.kw_proj(key)
        qp = self.qp_proj(query)
        kp = self.kp_proj(key)

        qw = self.split_heads(qw)
        kw = self.split_heads(kw)
        qp = self.split_heads(qp)
        kp = self.split_heads(kp)

        value_heads = self.split_heads(value)

        if not self.is_cross_attention:
            cos, sin = self.rotary(qw, seq_len=L)
            qw, kw = apply_rotary_emb(qw, kw, cos, sin)
            qp, kp = apply_rotary_emb(qp, kp, cos, sin)

        w2p_scores = torch.matmul(qw, kp.transpose(-2, -1)) / math.sqrt(self.head_dim)
        p2w_scores = torch.matmul(qp, kw.transpose(-2, -1)) / math.sqrt(self.head_dim)

        w2p_activated = F.silu(w2p_scores)
        p2w_activated = F.silu(p2w_scores)
        attn_scores = (w2p_activated + p2w_activated) / math.sqrt(2)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_heads)
        attn_output = self.combine_heads(attn_output)

        output = self.out_proj(attn_output)

        return output, attn_weights

class EncoderBlock(nn.Module):
    """Encoder Block with Coupled Attention"""
    def __init__(self, d_model, d_k, ff_dim, num_heads=12):
        super().__init__()
        self.self_attn = MultiHeadCoupledAttention(d_model, d_k, num_heads, is_cross_attention=False)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim * 2, bias=False),
            nn.GLU(dim=-1),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, d_model, bias=False),
        )

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x, src_mask):
        residual = x
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.self_attn(x_norm, mask=src_mask)
        x = residual + self.dropout1(attn_out)

        residual = x
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = residual + self.dropout2(ffn_out)

        return x, attn_weights

class DecoderBlock(nn.Module):
    """Decoder Block with Coupled Attention"""
    def __init__(self, d_model, d_k, ff_dim, num_heads=12):
        super().__init__()
        self.self_attn = MultiHeadCoupledAttention(d_model, d_k, num_heads, is_cross_attention=False)
        self.cross_attn = MultiHeadCoupledAttention(d_model, d_k, num_heads, is_cross_attention=True)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim * 2, bias=False),
            nn.GLU(dim=-1),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, d_model, bias=False),
        )

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        residual = x
        x_norm = self.norm1(x)
        attn_out, self_attn_weights = self.self_attn(x_norm, mask=tgt_mask)
        x = residual + self.dropout1(attn_out)

        residual = x
        x_norm = self.norm2(x)
        cross_attn_out, cross_attn_weights = self.cross_attn(
            x_norm, encoder_output=encoder_output, mask=src_mask
        )
        x = residual + self.dropout2(cross_attn_out)

        residual = x
        x_norm = self.norm3(x)
        ffn_out = self.ffn(x_norm)
        x = residual + self.dropout3(ffn_out)

        return x, self_attn_weights, cross_attn_weights

class Encoder(nn.Module):
    """Encoder with Language Embeddings"""
    def __init__(self, num_layers, d_model, d_k, ff_dim, vocab_size, num_heads=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.language_embedding = nn.Embedding(NUM_LANGUAGES, LANG_EMBED_DIM)
        self.lang_proj = nn.Linear(LANG_EMBED_DIM, d_model, bias=False)

        self.layers = nn.ModuleList([
            EncoderBlock(d_model, d_k, ff_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_tokens, src_lang_ids, src_mask):
        word_embed = self.embedding(src_tokens)
        lang_embed = self.language_embedding(src_lang_ids)
        lang_embed = self.lang_proj(lang_embed)
        x = word_embed + lang_embed

        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, src_mask)
            all_attn_weights.append(attn_weights)

        x = self.norm(x)
        return x, all_attn_weights

class Decoder(nn.Module):
    """Decoder with Language Embeddings"""
    def __init__(self, num_layers, d_model, d_k, ff_dim, vocab_size, num_heads=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.language_embedding = nn.Embedding(NUM_LANGUAGES, LANG_EMBED_DIM)
        self.lang_proj = nn.Linear(LANG_EMBED_DIM, d_model, bias=False)

        self.layers = nn.ModuleList([
            DecoderBlock(d_model, d_k, ff_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tgt_tokens, tgt_lang_ids, encoder_output, src_mask, tgt_mask):
        word_embed = self.embedding(tgt_tokens)
        lang_embed = self.language_embedding(tgt_lang_ids)
        lang_embed = self.lang_proj(lang_embed)
        x = word_embed + lang_embed

        all_self_attn_weights = []
        all_cross_attn_weights = []

        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, encoder_output, src_mask, tgt_mask)
            all_self_attn_weights.append(self_attn)
            all_cross_attn_weights.append(cross_attn)

        x = self.norm(x)
        logits = self.output_proj(x)

        return logits, all_self_attn_weights, all_cross_attn_weights

class TranslationTransformer(nn.Module):
    """Translation Transformer (Encoder-Decoder)"""
    def __init__(self, num_encoder_layers, num_decoder_layers, d_model, d_k, ff_dim, vocab_size, num_heads=12):
        super().__init__()
        self.encoder = Encoder(num_encoder_layers, d_model, d_k, ff_dim, vocab_size, num_heads)
        self.decoder = Decoder(num_decoder_layers, d_model, d_k, ff_dim, vocab_size, num_heads)

    def forward(self, src_tokens, src_lang_ids, tgt_tokens, tgt_lang_ids, src_mask, tgt_mask):
        encoder_output, encoder_attn_weights = self.encoder(src_tokens, src_lang_ids, src_mask)
        logits, decoder_self_attn_weights, decoder_cross_attn_weights = self.decoder(
            tgt_tokens, tgt_lang_ids, encoder_output, src_mask, tgt_mask
        )
        return logits, encoder_attn_weights, decoder_self_attn_weights, decoder_cross_attn_weights

# ===================== UTILITY FUNCTIONS =====================
def create_padding_mask(tokens, pad_id, device):
    return (tokens != pad_id).to(device)

def create_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask

def clean_text(text):
    if not text or not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text).strip()[:500]

# ===================== LOAD MODEL =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load tokenizer
tokenizer = SentencePieceUnigramTokenizer(TOKENIZER_MODEL_PATH)
vocab_size = tokenizer.get_vocab_size()
print(f"Tokenizer vocab size: {vocab_size}")

# Initialize model
model = TranslationTransformer(
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    d_model=D_MODEL,
    d_k=D_K,
    ff_dim=FF_DIM,
    vocab_size=vocab_size,
    num_heads=NUM_HEADS
).to(device)

# Load checkpoint
if os.path.exists(CHECKPOINT_PATH):
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    print("Model loaded successfully.")
else:
    print("Checkpoint not found!")
model.eval()

# ===================== TRANSLATION FUNCTION =====================
def translate_english_to_hindi(english_text):
    """Gradio-compatible translation function"""
    if not english_text.strip():
        return "Please enter a sentence."

    src_tokens = tokenizer.encode(english_text, add_bos=True, add_eos=True)

    # Pad/truncate
    if len(src_tokens) < SEQ_LEN:
        src_tokens = src_tokens + [PAD_ID] * (SEQ_LEN - len(src_tokens))
    else:
        src_tokens = src_tokens[:SEQ_LEN]
        src_tokens[-1] = EOS_ID

    src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)
    src_lang_tensor = torch.tensor([[lang_map["en"]] * SEQ_LEN], dtype=torch.long).to(device)
    src_mask = create_padding_mask(src_tensor, PAD_ID, device).unsqueeze(1).unsqueeze(2)

    with torch.no_grad():
        encoder_output, _ = model.encoder(src_tensor, src_lang_tensor, src_mask)

    # Beam search
    beams = [(torch.tensor([BOS_ID], device=device), 0.0)]
    for step in range(MAX_GENERATION_LENGTH):
        candidates = []
        for seq, score in beams:
            if seq[-1] == EOS_ID:
                candidates.append((seq, score))
                continue

            decoder_input = seq.unsqueeze(0)
            if decoder_input.size(1) < SEQ_LEN:
                padding = torch.full((1, SEQ_LEN - decoder_input.size(1)), PAD_ID,
                                   device=device, dtype=torch.long)
                decoder_input = torch.cat([decoder_input, padding], dim=1)
            decoder_input = decoder_input[:, :SEQ_LEN]

            tgt_lang_tensor = torch.tensor([[lang_map["hi"]] * decoder_input.size(1)],
                                          dtype=torch.long).to(device)

            tgt_padding_mask = create_padding_mask(decoder_input, PAD_ID, device)
            causal_mask = create_causal_mask(decoder_input.size(1), device)
            tgt_mask = causal_mask.unsqueeze(0) & tgt_padding_mask.unsqueeze(1)

            with torch.no_grad():
                logits, _, _ = model.decoder(
                    decoder_input, tgt_lang_tensor, encoder_output, src_mask, tgt_mask
                )

            next_logits = logits[0, seq.size(0)-1, :] / TEMPERATURE

            # Repetition penalty
            if REPETITION_PENALTY != 1.0:
                for token_id in seq:
                    if next_logits[token_id] < 0:
                        next_logits[token_id] *= REPETITION_PENALTY
                    else:
                        next_logits[token_id] /= REPETITION_PENALTY

            probs = F.softmax(next_logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, NUM_BEAMS)

            for token_id, token_prob in zip(topk_indices, topk_probs):
                new_seq = torch.cat([seq, token_id.unsqueeze(0)])
                new_score = score + math.log(token_prob.item())
                # Length penalty
                new_score = new_score / ((5 + len(new_seq)) / 6) ** LENGTH_PENALTY
                candidates.append((new_seq, new_score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:NUM_BEAMS]

        if all(beam[0][-1] == EOS_ID for beam in beams):
            break

    best_seq = beams[0][0]
    if best_seq[0] == BOS_ID:
        best_seq = best_seq[1:]
    if EOS_ID in best_seq:
        best_seq = best_seq[:torch.where(best_seq == EOS_ID)[0][0]]

    translation = tokenizer.decode(best_seq.tolist())
    return translation

# ===================== GRADIO INTERFACE =====================
# ===================== TRANSLATION UI =====================

def translate_request(user_input, history, mode, prev_en, prev_hi):

    if history is None:
        history = []

    # Save previous translation
    if prev_en and prev_hi:
        history.append((prev_en, prev_hi))

    if not user_input.strip():
        return user_input, history, "", "", history, "", ""

    # SINGLE MODE
    if mode == "Single Sentence":

        new_en = user_input
        new_hi = translate_english_to_hindi(user_input)

    # BATCH MODE
    else:

        lines = [l.strip() for l in user_input.split("\n") if l.strip()]

        translations = []
        for line in lines:
            translations.append(translate_english_to_hindi(line))

        new_en = "\n".join(lines)
        new_hi = "\n".join(translations)

        # Store each sentence separately in history
        for en, hi in zip(lines, translations):
            history.append((en, hi))

        # Prevent duplicate storage
        prev_en = ""
        prev_hi = ""

        return "", history, new_en, new_hi, history, prev_en, prev_hi

    return "", history, new_en, new_hi, history, new_en, new_hi


def clear_history(prev_en, prev_hi):
    return [], prev_en, prev_hi


def restart_app():
    return "", "", [], [], "", "", "Single Sentence"


with gr.Blocks(title="English → Hindi Translator") as demo:

    gr.Markdown("# English → Hindi Translation")

    mode = gr.Radio(
        ["Single Sentence", "Batch (Multiple Lines)"],
        value="Single Sentence",
        label="Translation Mode"
    )

    # ================= CURRENT TRANSLATION =================

    gr.Markdown("## Current Translation")

    with gr.Group():
        with gr.Row():

            current_en = gr.Textbox(
                label="English (Input)",
                lines=6,
                placeholder="Type English text here...",
                interactive=True
            )

            current_hi = gr.Textbox(
                label="Hindi (Output)",
                lines=6,
                interactive=False
            )

    # Buttons
    with gr.Row():
        translate_btn = gr.Button("Translate")
        restart_btn = gr.Button("Restart")

    # ================= HISTORY =================

    state = gr.State([])
    prev_en_state = gr.State("")
    prev_hi_state = gr.State("")

    with gr.Accordion("Translation History", open=False):

        history_table = gr.Dataframe(
            headers=["English", "Hindi"],
            datatype=["str", "str"],
            interactive=False
        )

        clear_btn = gr.Button("Clear History")

    # ===== TRANSLATE =====

    translate_btn.click(
        translate_request,
        inputs=[current_en, state, mode, prev_en_state, prev_hi_state],
        outputs=[
            current_en,
            state,
            current_en,
            current_hi,
            history_table,
            prev_en_state,
            prev_hi_state
        ]
    )

    current_en.submit(
        translate_request,
        inputs=[current_en, state, mode, prev_en_state, prev_hi_state],
        outputs=[
            current_en,
            state,
            current_en,
            current_hi,
            history_table,
            prev_en_state,
            prev_hi_state
        ]
    )

    # ===== CLEAR HISTORY =====

    clear_btn.click(
        lambda: [],
        None,
        history_table
    ).then(
        lambda: [],
        None,
        state
    )

    # ===== RESTART =====

    restart_btn.click(
        restart_app,
        outputs=[
            current_en,
            current_hi,
            history_table,
            state,
            prev_en_state,
            prev_hi_state,
            mode
        ]
    )


if __name__ == "__main__":
    demo.launch()
