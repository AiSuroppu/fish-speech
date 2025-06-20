import os
import queue
import threading
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import click
import numpy as np
import torch
import torch._inductor.config
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer

from fish_speech.content_sequence import (
    ContentSequence,
    TextPart,
    VQPart,
)
from fish_speech.tokenizer import IM_END_TOKEN, PAD_TOKEN

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

if hasattr(torch._inductor.config, "fx_graph_cache"):
    # Experimental feature to reduce compilation times, will be on by default in future
    torch._inductor.config.fx_graph_cache = True


from torch.nn.attention import SDPBackend, sdpa_kernel

from fish_speech.models.text2semantic.llama import (
    BaseTransformer,
    DualARTransformer,
    NaiveTransformer,
)


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
    pad_token_id: Optional[int] = None,
) -> torch.Tensor:
    # Apply repetition penalty
    if previous_tokens is not None and previous_tokens.numel() > 0:
        dim = logits.ndim - 1
        previous_tokens = previous_tokens.long()

        # Gather the original logits for the tokens in history
        score = torch.gather(logits, dim, previous_tokens)

        # Apply penalty: score > 0 -> score / penalty, score < 0 -> score * penalty
        penalized_score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        
        # Do not penalize padding tokens
        if pad_token_id is not None:
            valid_tokens_mask = (previous_tokens != pad_token_id)
            penalized_score = torch.where(valid_tokens_mask, penalized_score, score)

        # Scatter the updated scores back to the logits tensor
        logits = logits.scatter(dim, previous_tokens, penalized_score)

    # Apply top-p sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    
    # Ensure at least one token is kept
    if logits.dim() > 1:
        # Batched case
        sorted_indices_to_remove[:, 0] = False
    else:
        # Non-batched case
        sorted_indices_to_remove[0] = False
        
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))
    logits = logits / torch.clip(temperature, min=1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits[0, -1],
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        previous_tokens=previous_tokens,
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def decode_one_token_ar(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    previous_tokens: torch.Tensor = None,
) -> torch.Tensor:
    # print(x, torch.count_nonzero(vq_masks))
    x = model.forward_generate(
        x,
        input_pos,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
    )
    logits = x.logits  # [:, -1:]
    hidden_states = x.hidden_states  # [:, -1:]

    codebooks = [
        sample(
            logits,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            previous_tokens=(
                previous_tokens[:, 0] if previous_tokens is not None else None
            ),
        )[0]
    ]

    # Cleanup the cache
    for layer in model.fast_layers:
        layer.attention.kv_cache.k_cache.fill_(0)
        layer.attention.kv_cache.v_cache.fill_(0)

    input_pos = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
    model.forward_generate_fast(hidden_states, input_pos)
    a = codebooks[0] - model.tokenizer.semantic_begin_id
    a[a < 0] = 0
    hidden_states = model.fast_embeddings(a)
    codebooks.append(a)

    for codebook_idx in range(1, model.config.num_codebooks):
        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, input_pos)

        short_logits = logits[:, :, :1024]

        # Convert logits to probs
        a = sample(
            short_logits,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            previous_tokens=(
                previous_tokens[codebook_idx + 1]
                if previous_tokens is not None
                else None
            ),
        )[0]

        hidden_states = model.fast_embeddings(a)
        codebooks.append(a)

    codebooks = torch.stack(codebooks, dim=1)
    return codebooks.T


def decode_n_tokens(
    model: NaiveTransformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    decode_one_token=decode_one_token_ar,
):
    previous_tokens = torch.zeros(
        (model.config.num_codebooks + 1, model.config.max_seq_len),
        dtype=torch.int,
        device=cur_token.device,
    )

    for i in tqdm(range(num_new_tokens)):
        # We need to get windowed repeat penalty
        win_size = 16
        if i < win_size:
            window = previous_tokens[:, :win_size]
        else:
            window = previous_tokens[:, i - win_size : i]

        with sdpa_kernel(
            SDPBackend.MATH
        ):  # Actually better for Inductor to codegen attention here
            next_token = decode_one_token(
                model=model,
                x=cur_token,
                input_pos=input_pos,
                previous_tokens=window,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                audio_masks=audio_masks,
                audio_parts=audio_parts,
            ).clone()

        input_pos += 1
        cur_token = next_token.view(1, model.config.num_codebooks + 1, -1)
        previous_tokens[:, i : i + 1] = next_token.view(
            model.config.num_codebooks + 1, -1
        )

        if cur_token[0, 0, -1] == model.tokenizer.get_token_id(IM_END_TOKEN):
            break

    return previous_tokens[:, : i + 1]


@torch.no_grad()
@torch.inference_mode()
def generate(
    *,
    model: BaseTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    decode_one_token=decode_one_token_ar,
    num_samples: int = 1,
    **sampling_kwargs,
):
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(1)
    prompt = prompt[None].repeat(num_samples, 1, 1)

    if T >= model.config.max_seq_len:
        raise ValueError(
            f"Input sequence length {T} exceeds max_seq_len {model.config.max_seq_len}"
        )

    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T

        T_new = T + max_new_tokens
    else:
        T_new = model.config.max_seq_len
        max_new_tokens = T_new - T

    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(
            max_batch_size=num_samples,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )

    codebook_dim = 1 + model.config.num_codebooks
    input_pos = torch.arange(0, T, device=device)
    empty = torch.empty(
        (codebook_dim, model.config.max_seq_len), dtype=dtype, device=device
    )
    empty[:, :T] = prompt
    seq = empty

    temperature = torch.tensor(
        sampling_kwargs["temperature"], device=device, dtype=torch.bfloat16
    )
    top_p = torch.tensor(sampling_kwargs["top_p"], device=device, dtype=torch.bfloat16)
    repetition_penalty = torch.tensor(
        sampling_kwargs["repetition_penalty"], device=device, dtype=torch.bfloat16
    )

    prefill_decode = decode_one_token_ar

    first_token = prefill_decode(
        model,
        prompt.view(1, codebook_dim, -1),
        input_pos,
        temperature,
        top_p,
        repetition_penalty,
        audio_masks,
        audio_parts,
    )
    seq[:, T : T + 1] = first_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    x = decode_n_tokens(
        model,
        first_token.view(1, codebook_dim, -1),
        input_pos,
        max_new_tokens - 1,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
        decode_one_token=decode_one_token,
    )
    seq = seq[:, : T + 1 + x.size(1)]
    seq[:, T + 1 :] = x
    return seq


def init_model(checkpoint_path, device, precision, compile=False):
    model = DualARTransformer.from_pretrained(checkpoint_path, load_weights=True)

    model = model.to(device=device, dtype=precision)
    logger.info(f"Restored model from checkpoint")

    if isinstance(model, DualARTransformer):
        decode_one_token = decode_one_token_ar
        prefill_n_tokens = decode_one_token_ar
        logger.info("Using DualARTransformer")
    else:
        raise ValueError("Unsupported model type")

    # Initialize cache
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )

    if compile:
        logger.info("Compiling function...")
        decode_one_token = torch.compile(
            decode_one_token,
            # mode="max-autotune-no-cudagraphs",
            backend="inductor" if torch.cuda.is_available() else "aot_eager",
            mode="reduce-overhead" if torch.cuda.is_available() else None,
            fullgraph=True,
        )

    return model.eval(), decode_one_token


@dataclass
class GenerateResponse:
    action: Literal["sample", "next"]
    codes: Optional[torch.Tensor] = None
    text: Optional[str] = None


def generate_long(
    *,
    model,
    device: str | torch.device,
    decode_one_token: callable,
    text: str,
    num_samples: int = 1,
    max_new_tokens: int = 0,
    top_p: int = 0.8,
    repetition_penalty: float = 1.1,
    temperature: float = 0.8,
    compile: bool = False,
    iterative_prompt: bool = True,
    chunk_length: int = 512,
    prompt_text: Optional[str | list[str]] = None,
    prompt_tokens: Optional[torch.Tensor | list[torch.Tensor]] = None,
):
    assert 0 < top_p <= 1, "top_p must be in (0, 1]"
    assert 0 < repetition_penalty < 2, "repetition_penalty must be in (0, 2)"
    assert 0 < temperature < 2, "temperature must be in (0, 2)"

    use_prompt = prompt_text is not None and prompt_tokens is not None
    if use_prompt and isinstance(prompt_text, str):
        prompt_text = [prompt_text]
        prompt_tokens = [prompt_tokens]

    assert use_prompt is False or len(prompt_text) == len(
        prompt_tokens
    ), "Prompt text and tokens must have the same length"

    prompt_tokens = [i.cpu() for i in prompt_tokens]

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokenizer = model.tokenizer
    base_content_sequence = ContentSequence(modality="interleave")

    max_length = model.config.max_seq_len
    if use_prompt:
        for t, c in zip(prompt_text, prompt_tokens):
            base_content_sequence.append(
                [
                    TextPart(text=t),
                    VQPart(codes=c),
                ],
                add_end=True,
                speaker=0,
            )
    base_content_sequence.append(
        [
            TextPart(text=text),
        ],
        add_end=False,
        speaker=0,
    )

    encoded, audio_masks, audio_parts = base_content_sequence.encode_for_inference(
        tokenizer, num_codebooks=model.config.num_codebooks
    )
    if encoded.size(1) > max_length - 2048:
        raise ValueError(f"Prompt is too long: {encoded.size(1)} > {max_length - 2048}")

    encoded = encoded.to(device=device)
    logger.info(f"Encoded text: {text}")

    # Move temperature, top_p, repetition_penalty to device
    # This is important so that changing params doesn't trigger recompile
    temperature = torch.tensor(temperature, device=device, dtype=torch.float)
    top_p = torch.tensor(top_p, device=device, dtype=torch.float)
    repetition_penalty = torch.tensor(
        repetition_penalty, device=device, dtype=torch.float
    )

    for sample_idx in range(num_samples):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        global_encoded = []
        seg_idx = 0
        prompt_length = encoded.size(1)

        t0 = time.perf_counter()
        y = generate(
            model=model,
            prompt=encoded,
            max_new_tokens=max_new_tokens,
            audio_masks=audio_masks,
            audio_parts=audio_parts,
            decode_one_token=decode_one_token,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        if sample_idx == 0 and seg_idx == 0 and compile:
            logger.info(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t = time.perf_counter() - t0

        tokens_generated = y.size(1) - prompt_length
        tokens_sec = tokens_generated / t
        logger.info(
            f"Generated {tokens_generated} tokens in {t:.02f} seconds, {tokens_sec:.02f} tokens/sec"
        )
        logger.info(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")

        if torch.cuda.is_available():
            logger.info(
                f"GPU Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB"
            )

        # Put the generated tokens
        # since there is <im_end>, we remove last token
        codes = y[1:, prompt_length:-1].clone()
        assert (codes >= 0).all(), f"Negative code found"

        decoded = y[:, prompt_length:].clone()
        # But for global encoding, we should keep the <im_end> token

        global_encoded.append(decoded.cpu())
        assert (codes >= 0).all(), f"Negative code found: {codes}"
        yield GenerateResponse(action="sample", codes=codes, text=text)
        seg_idx += 1

        # This indicates the end of the current sample
        yield GenerateResponse(action="next")


def generate_t2s_batch(
    *,
    model: BaseTransformer,
    texts: list[str],
    device: str | torch.device,
    max_new_tokens: int = 0,
    top_p: float = 0.8,
    repetition_penalty: float = 1.1,
    temperature: float = 0.8,
    prompt_text: Optional[Union[str, list[str]]] = None,
    prompt_tokens: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
):
    assert 0 < top_p <= 1, "top_p must be in (0, 1]"
    assert 0 < repetition_penalty < 2, "repetition_penalty must be in (0, 2)"
    assert 0 < temperature < 2, "temperature must be in (0, 2)"

    tokenizer = model.tokenizer
    batch_size = len(texts)
    pad_token_id = tokenizer.get_token_id(PAD_TOKEN)
    im_end_token_id = tokenizer.get_token_id(IM_END_TOKEN)
    codebook_dim = 1 + model.config.num_codebooks

    # 1. Handle prompt broadcasting
    if prompt_text is not None and isinstance(prompt_text, str):
        prompt_text = [prompt_text] * batch_size
    if (
        prompt_tokens is not None
        and isinstance(prompt_tokens, torch.Tensor)
        and prompt_tokens.ndim == 2
    ):
        prompt_tokens = [prompt_tokens] * batch_size

    # 2. Preprocessing and Padding
    prompt_sequences = []
    prompt_lengths = []

    for i in range(batch_size):
        seq = ContentSequence(modality="interleave")

        current_prompt_text_exists = (
            prompt_text is not None and i < len(prompt_text) and prompt_text[i]
        )
        current_prompt_tokens_exist = (
            prompt_tokens is not None
            and i < len(prompt_tokens)
            and prompt_tokens[i] is not None
        )

        if current_prompt_text_exists and current_prompt_tokens_exist:
            seq.append(
                [TextPart(text=prompt_text[i]), VQPart(codes=prompt_tokens[i].cpu())],
                add_end=True,
                speaker=0,
            )

        seq.append([TextPart(text=texts[i])], add_end=False, speaker=0)

        encoded, _, _ = seq.encode_for_inference(
            tokenizer, num_codebooks=model.config.num_codebooks
        )
        prompt_sequences.append(encoded)
        prompt_lengths.append(encoded.shape[1])

    max_prompt_len = max(prompt_lengths)

    # Use right-padding for simplicity and correctness in generation
    padded_prompts = torch.full(
        (batch_size, codebook_dim, max_prompt_len),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )

    for i, seq in enumerate(prompt_sequences):
        length = seq.shape[1]
        padded_prompts[i, :, :length] = seq

    key_padding_mask = padded_prompts[:, 0, :] == pad_token_id

    # 3. Setup KV Cache for Batch
    if max_new_tokens <= 0:
        max_new_tokens = model.config.max_seq_len - max_prompt_len

    max_total_len = min(model.config.max_seq_len, max_prompt_len + max_new_tokens)
    max_new_tokens = max_total_len - max_prompt_len

    model.setup_caches(
        max_batch_size=batch_size,
        max_seq_len=max_total_len,
        dtype=next(model.parameters()).dtype,
    )

    # 4. Efficient Prefill Pass
    logger.info("Prefilling KV cache for batch...")
    input_pos = torch.arange(0, max_prompt_len, device=device)
    prefill_result = model.forward_generate(
        padded_prompts, input_pos, key_padding_mask=key_padding_mask, return_all=True
    )

    # For right-padding, the last meaningful token is at `length - 1`
    last_token_indices = torch.tensor(
        [l - 1 for l in prompt_lengths], device=device, dtype=torch.long
    )

    # 5. Iterative Generation
    current_logits = prefill_result.logits[torch.arange(batch_size), last_token_indices]
    current_hidden_states = prefill_result.hidden_states[
        torch.arange(batch_size), last_token_indices
    ]

    generated_codes_list = [[] for _ in range(batch_size)]
    eos_reached = torch.zeros(batch_size, dtype=torch.bool, device=device)

    temperature = torch.tensor(temperature, device=device, dtype=torch.bfloat16)
    top_p = torch.tensor(top_p, device=device, dtype=torch.bfloat16)
    repetition_penalty = torch.tensor(
        repetition_penalty, device=device, dtype=torch.bfloat16
    )

    sequence_lengths = torch.tensor(prompt_lengths, device=device, dtype=torch.long)

    token_history = torch.full(
        (batch_size, codebook_dim, max_new_tokens),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )

    for i in tqdm(range(max_new_tokens), desc="Generating tokens"):
        active_mask = ~eos_reached
        if not active_mask.any():
            break

        active_indices = torch.where(active_mask)[0]
        num_active = active_indices.shape[0]

        # A. Create active-only sub-batch of states
        active_logits = current_logits[active_indices]

        window_size = 16
        start_pos = max(0, i - window_size)
        active_history_window = token_history[active_indices, ..., start_pos:i]

        # B. Sample the next main token for active sequences
        probs = logits_to_probs(
            active_logits,
            temperature,
            top_p,
            repetition_penalty,
            previous_tokens=active_history_window[:, 0, :],
            pad_token_id=pad_token_id,
        )
        active_next_token = multinomial_sample_one_no_sync(probs)

        # C. Generate VQ codes for active sequences
        active_hidden_states = current_hidden_states[active_indices]

        # C.1. Clear fast VQ cache for the new step.
        # The VQ generation is a self-contained loop for each main token.
        # We must reset its KV cache to avoid state leakage between steps.
        for layer in model.fast_layers:
            if layer.attention.kv_cache is not None:
                layer.attention.kv_cache.k_cache.fill_(0)
                layer.attention.kv_cache.v_cache.fill_(0)

        # C.2. Prime the fast transformer
        input_pos_prime = torch.zeros(num_active, dtype=torch.long, device=device)
        model.forward_generate_fast(
            active_hidden_states.unsqueeze(1), input_pos_prime, active_cache_mask=active_mask
        )

        # C.3. Autoregressively generate remaining VQ codes
        active_first_vq_code = active_next_token - tokenizer.semantic_begin_id
        active_first_vq_code.clamp_(min=0)

        active_codes_for_next_step = [active_next_token, active_first_vq_code]
        active_hidden_states_fast = model.fast_embeddings(active_first_vq_code)

        for codebook_idx in range(1, model.config.num_codebooks):
            input_pos_fast = torch.full(
                (num_active,),
                fill_value=codebook_idx,
                device=device,
                dtype=torch.long,
            )
            fast_logits = model.forward_generate_fast(
                active_hidden_states_fast, input_pos_fast, active_cache_mask=active_mask
            )
            short_logits = fast_logits.squeeze(1)[:, :1024]

            probs_fast = logits_to_probs(
                short_logits,
                temperature,
                top_p,
                repetition_penalty,
                previous_tokens=active_history_window[:, codebook_idx + 1, :],
                pad_token_id=pad_token_id,
            )
            active_next_vq_code = multinomial_sample_one_no_sync(probs_fast)

            active_hidden_states_fast = model.fast_embeddings(active_next_vq_code)
            active_codes_for_next_step.append(active_next_vq_code)

        active_current_batch_all_codes = torch.cat(active_codes_for_next_step, dim=1)

        # D. Update history and EOS state for the full batch
        token_history[active_indices, :, i] = active_current_batch_all_codes.long()

        next_token = torch.full(
            (batch_size,), pad_token_id, device=device, dtype=torch.long
        )
        next_token[active_indices] = active_current_batch_all_codes[:, 0].long()
        eos_reached |= next_token == im_end_token_id

        # E. Append VQ codes to output list for non-finished sequences
        active_vq_codes_for_step = active_current_batch_all_codes[:, 1:]
        for idx, original_idx in enumerate(active_indices):
            if next_token[original_idx] != im_end_token_id:
                generated_codes_list[original_idx].append(
                    active_vq_codes_for_step[idx].clone()
                )

        # F. Prepare for the next main transformer step using the active-only sub-batch
        active_codes_for_slow_model = active_current_batch_all_codes.unsqueeze(-1)
        active_sequence_lengths = sequence_lengths[active_indices]

        # Create a causal attention mask for the sub-batch, respecting each sequence's length
        max_len_in_cache = active_sequence_lengths.max() + 1
        generation_attn_mask = (
            torch.arange(max_len_in_cache, device=device)[None, :]
            < (active_sequence_lengths[:, None] + 1)
        )

        result = model.forward_generate(
            active_codes_for_slow_model,
            active_sequence_lengths,  # Pass current lengths as positions
            active_cache_mask=active_mask,
            generation_attn_mask=generation_attn_mask,
        )

        # G. Scatter results back to the full-sized state tensors
        current_logits[active_indices] = result.logits.squeeze(1)
        current_hidden_states[active_indices] = result.hidden_states.squeeze(1)

        # H. Update sequence lengths for active sequences
        sequence_lengths[active_indices] += 1

    # 6. Post-processing
    output_codes = []
    for i in range(batch_size):
        if not generated_codes_list[i]:
            output_codes.append(None)
            continue

        codes_tensor = torch.stack(generated_codes_list[i], dim=1)
        output_codes.append(codes_tensor.cpu())

    return output_codes


@dataclass
class WrappedGenerateResponse:
    status: Literal["success", "error"]
    response: Optional[GenerateResponse | Exception] = None


@dataclass
class GenerateRequest:
    request: dict
    response_queue: queue.Queue


def launch_thread_safe_queue(
    checkpoint_path,
    device,
    precision,
    compile: bool = False,
):
    input_queue = queue.Queue()
    init_event = threading.Event()

    def worker():
        model, decode_one_token = init_model(
            checkpoint_path, device, precision, compile=compile
        )
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        init_event.set()

        while True:
            item: GenerateRequest | None = input_queue.get()
            if item is None:
                break

            kwargs = item.request
            response_queue = item.response_queue

            try:
                for chunk in generate_long(
                    model=model, decode_one_token=decode_one_token, **kwargs
                ):
                    response_queue.put(
                        WrappedGenerateResponse(status="success", response=chunk)
                    )
            except Exception as e:
                logger.error(traceback.format_exc())
                response_queue.put(WrappedGenerateResponse(status="error", response=e))

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()

    return input_queue


@click.command()
@click.option(
    "--text",
    type=str,
    default="你说的对, 但是原神是一款由米哈游自主研发的开放世界手游.",
)
@click.option("--prompt-text", type=str, default=None, multiple=True)
@click.option(
    "--prompt-tokens",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    multiple=True,
)
@click.option("--num-samples", type=int, default=1)
@click.option("--max-new-tokens", type=int, default=0)
@click.option("--top-p", type=float, default=0.8)
@click.option("--repetition-penalty", type=float, default=1.1)
@click.option("--temperature", type=float, default=0.8)
@click.option(
    "--checkpoint-path",
    type=click.Path(path_type=Path, exists=True),
    default="checkpoints/openaudio-s1-mini",
)
@click.option("--device", type=str, default="cuda")
@click.option("--compile/--no-compile", default=False)
@click.option("--seed", type=int, default=42)
@click.option("--half/--no-half", default=False)
@click.option("--iterative-prompt/--no-iterative-prompt", default=True)
@click.option("--chunk-length", type=int, default=300)
@click.option("--output-dir", type=Path, default="temp")
def main(
    text: str,
    prompt_text: Optional[list[str]],
    prompt_tokens: Optional[list[Path]],
    num_samples: int,
    max_new_tokens: int,
    top_p: int,
    repetition_penalty: float,
    temperature: float,
    checkpoint_path: Path,
    device: str,
    compile: bool,
    seed: int,
    half: bool,
    iterative_prompt: bool,
    chunk_length: int,
    output_dir: Path,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    precision = torch.half if half else torch.bfloat16

    if prompt_text is not None and len(prompt_text) != len(prompt_tokens):
        raise ValueError(
            f"Number of prompt text ({len(prompt_text)}) and prompt tokens ({len(prompt_tokens)}) should be the same"
        )

    logger.info("Loading model ...")
    t0 = time.time()
    model, decode_one_token = init_model(
        checkpoint_path, device, precision, compile=compile
    )
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

    if prompt_tokens is not None:
        prompt_tokens = [torch.from_numpy(np.load(p)) for p in prompt_tokens]

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    generator = generate_long(
        model=model,
        device=device,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        compile=compile,
        iterative_prompt=iterative_prompt,
        chunk_length=chunk_length,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
    )

    idx = 0
    codes = []

    for response in generator:
        if response.action == "sample":
            codes.append(response.codes)
            logger.info(f"Sampled text: {response.text}")
        elif response.action == "next":
            if codes:
                codes_npy_path = os.path.join(output_dir, f"codes_{idx}.npy")
                np.save(codes_npy_path, torch.cat(codes, dim=1).cpu().numpy())
                logger.info(f"Saved codes to {codes_npy_path}")
            logger.info(f"Next sample")
            codes = []
            idx += 1
        else:
            logger.error(f"Error: {response}")


if __name__ == "__main__":
    main()
