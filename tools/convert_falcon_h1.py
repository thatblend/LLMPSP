#!/usr/bin/env python3
"""Convert Falcon-H1-Tiny-90M-Instruct to FalconPSP's streamed FHQ4 format.

Needs NumPy only. Dense matrices become Q4_0 with one 8-bit scale code per
32-weight block, stored as four-row tile groups (FHQ4 version 3, the layout
the PSP VFPU kernel reads; see PROJECT.md sections 2.10 and 8) and the
exact Falcon byte-level BPE vocabulary and merge ranks are embedded in the
output.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import urllib.request
import numpy as np

MAGIC = b"FHQ4"
VERSION = 3
HEADER_BYTES = 256
QK = 32
GROUP_ROWS = 4
# Version 3 header fields after the four header floats (bytes 140..155):
# absolute offset, entry count and entry size of the per-matrix scale-code
# table that sits between the tokenizer and the weight region.
QPARAM_HEADER = 160
QPARAM_ENTRY = 8
# Mantissa widths a matrix may use for its 8-bit block-scale codes; the PSP
# has one kernel variant per width (src/falcon_q4_psp.c).
SCALE_MANTISSA_BITS = (3, 4, 5)
DEFAULT_CONTEXT = 512
DEFAULT_MODEL = "tiiuae/Falcon-H1-Tiny-90M-Instruct"
FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")

def download_model(model_id: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = destination / name
        if target.exists() and target.stat().st_size:
            print(f"[download] keeping {target}")
            continue
        url = f"https://huggingface.co/{model_id}/resolve/main/{name}"
        print(f"[download] {url}")
        request = urllib.request.Request(url, headers={"User-Agent": "FalconH1PSP/1.0"})
        with urllib.request.urlopen(request) as response, target.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)

class SafeTensors:
    def __init__(self, path: Path):
        self.file = path.open("rb")
        header_len = struct.unpack("<Q", self.file.read(8))[0]
        self.header = json.loads(self.file.read(header_len))
        self.data_start = 8 + header_len

    def close(self) -> None:
        self.file.close()

    def tensor(self, name: str) -> np.ndarray:
        if name not in self.header:
            raise KeyError(f"checkpoint is missing tensor {name!r}")
        info = self.header[name]
        begin, end = info["data_offsets"]
        self.file.seek(self.data_start + begin)
        raw = self.file.read(end - begin)
        if len(raw) != end - begin:
            raise IOError(f"short read for tensor {name}")
        dtype = info["dtype"]
        if dtype == "BF16":
            words = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            values = (words << 16).view(np.float32)
        elif dtype == "F32":
            values = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)
        elif dtype == "F16":
            values = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        else:
            raise ValueError(f"unsupported dtype {dtype} for {name}")
        return values.reshape(info["shape"])

def rows_to_tiles(row_bytes: bytes, rows: int, cols: int) -> bytes:
    """FHQ4 version-1 Q4_0 rows -> version-2 tile groups (lossless).

    Version 1 stores each row as cols/32 blocks of {fp16 scale, 16 packed
    bytes} (byte j: low nibble = weight j + 8, high nibble = weight
    j + 16 + 8). Version 2 stores four rows per group:
    blocks * 8 bytes of scales (block 0 rows 0..3, block 1 rows 0..3, ...)
    then blocks * 64 bytes of tiles, where word r of tile t of block b
    holds row r's packed bytes 4t..4t+3 XOR 0x80. Same 18 bytes per 32
    weights; only the order and the high-nibble encoding change.
    """
    # An even block count keeps the tiles (blocks * 8 bytes into the
    # group) 16-byte aligned for the PSP kernel's lv.q.
    if rows % GROUP_ROWS or cols % (2 * QK):
        raise ValueError(f"tile groups need rows % {GROUP_ROWS} == 0 and "
                         f"cols % {2 * QK} == 0, got {rows}x{cols}")
    blocks = cols // QK
    data = np.frombuffer(row_bytes, dtype=np.uint8)
    if data.size != rows * blocks * 18:
        raise ValueError("Q4 byte count does not match the shape")
    data = data.reshape(rows // GROUP_ROWS, GROUP_ROWS, blocks, 18)
    # scales: [group, row, block, 2] -> [group, block, row, 2]
    scales = data[..., :2].transpose(0, 2, 1, 3).reshape(rows // GROUP_ROWS, -1)
    # packed: [group, row, block, tile, k] -> [group, block, tile, row, k]
    packed = (data[..., 2:] ^ np.uint8(0x80)).reshape(
        rows // GROUP_ROWS, GROUP_ROWS, blocks, 4, 4)
    tiles = packed.transpose(0, 2, 3, 1, 4).reshape(rows // GROUP_ROWS, -1)
    return np.ascontiguousarray(np.concatenate([scales, tiles], axis=1)).tobytes()

def scale_code_valid(m: int, k: int) -> bool:
    """Every magnitude index 1..128 must decode to a normal, finite float."""
    return (m in SCALE_MANTISSA_BITS and 0 <= k < 65536 and
            ((1 + k) >> m) >= 1 and ((128 + k) >> m) <= 254)


def scale_table(m: int, k: int) -> np.ndarray:
    """Decoded value of every scale-code byte 0..255 (FHQ4 version 3).

    The byte is a signed int8 c. c == 0 is a zero scale; otherwise the
    magnitude is the float32 whose bit pattern is (|c| + k) << (23 - m),
    i.e. a minifloat with m mantissa bits placed by the per-matrix bias k,
    and the sign is the sign of c. The C runtime (scale_code_value in
    src/falcon_h1.c) and the PSP kernel (vi2f/vabs/vadd/vf2iz/vsgn in
    src/falcon_q4_tile_psp.h) compute exactly these bits."""
    if not scale_code_valid(m, k):
        raise ValueError(f"invalid scale code parameters m={m} k={k}")
    c = np.arange(256, dtype=np.uint8).view(np.int8).astype(np.int32)
    magnitude = ((np.abs(c) + k).astype(np.uint32) << np.uint32(23 - m))
    values = magnitude.view(np.float32) * np.sign(c).astype(np.float32)
    values[c == 0] = 0.0
    return values.astype(np.float32)


def encode_scales(scales: np.ndarray, m: int, k: int) -> np.ndarray:
    """Nearest code (linear distance) for every scale; int8, never -128."""
    table = scale_table(m, k)
    positive = table[:128]                      # codes 0..127, increasing
    a = np.abs(scales.astype(np.float32))
    hi = np.clip(np.searchsorted(positive, a), 1, 127)
    lo = hi - 1
    pick_lo = (a - positive[lo]) <= (positive[hi] - a)
    index = np.where(pick_lo, lo, hi).astype(np.int16)
    codes = np.where(scales < 0, -index, index)
    return codes.astype(np.int8)


def choose_scale_code(scales: np.ndarray, energy: np.ndarray):
    """Per-matrix (m, k) minimising the added weight error energy
    sum(energy * (decoded - scale)^2), energy = sum of squared signed
    nibbles of the block. Returns (m, k, codes, decoded)."""
    s = scales.astype(np.float32).ravel()
    e = energy.astype(np.float64).ravel()
    top = float(np.abs(s).max())
    best = None
    if top == 0.0:
        m, k = 4, 112 << 4
        codes = np.zeros(scales.shape, np.int8)
        return m, k, codes, np.zeros(scales.shape, np.float32)
    # Subsample huge matrices for the search; the winner is re-encoded in full.
    if s.size > 200000:
        pick = np.random.default_rng(0).choice(s.size, 200000, replace=False)
        s_search, e_search = s[pick], e[pick]
    else:
        s_search, e_search = s, e
    for m in SCALE_MANTISSA_BITS:
        # Smallest k whose top code (127) still reaches the largest scale.
        top_bits = int(np.array([top], np.float32).view(np.uint32)[0])
        k_top = -(-top_bits // (1 << (23 - m))) - 127
        for k in range(k_top - (2 << m), k_top + 2):
            if not scale_code_valid(m, k):
                continue
            table = scale_table(m, k)
            decoded = table[encode_scales(s_search, m, k).view(np.uint8)]
            err = float(np.sum(e_search * (decoded.astype(np.float64) -
                                           s_search) ** 2))
            if best is None or err < best[0]:
                best = (err, m, k)
    _, m, k = best
    codes = encode_scales(scales, m, k)
    return m, k, codes, scale_table(m, k)[codes.view(np.uint8)]


def pack_groups_v3(codes: np.ndarray, quants: np.ndarray) -> bytes:
    """[rows, blocks] int8 scale codes + [rows, cols] nibbles (w + 8) ->
    version-3 tile groups. A group of four rows with B blocks is
    B * 4 code bytes (block 0 rows 0..3, block 1 rows 0..3, ...) followed
    by the version-2 tiles (B * 64 bytes): word r of tile t of block b holds
    row r's packed bytes 4t..4t+3 (low nibble weight j + 8, high nibble
    weight j + 16 + 8) XOR 0x80. 17 bytes per 32 weights."""
    rows, cols = quants.shape
    if rows % GROUP_ROWS or cols % (4 * QK):
        # B % 4 == 0 keeps both the tiles (B * 4 bytes into the group) and
        # the group size (68 * B) 16-byte aligned for the PSP's lv.q.
        raise ValueError(f"version-3 groups need rows % {GROUP_ROWS} == 0 and "
                         f"cols % {4 * QK} == 0, got {rows}x{cols}")
    blocks = cols // QK
    groups = rows // GROUP_ROWS
    q = quants.astype(np.uint8).reshape(rows, blocks, QK)
    packed = (q[..., :16] | (q[..., 16:] << 4)) ^ np.uint8(0x80)
    head = codes.view(np.uint8).reshape(groups, GROUP_ROWS, blocks)
    head = head.transpose(0, 2, 1).reshape(groups, -1)
    tiles = packed.reshape(groups, GROUP_ROWS, blocks, 4, 4)
    tiles = tiles.transpose(0, 2, 3, 1, 4).reshape(groups, -1)
    return np.ascontiguousarray(np.concatenate([head, tiles], axis=1)).tobytes()


def q4_v3(values: np.ndarray):
    """Q4_0 with 8-bit scale codes: returns (m, k, version-3 bytes).

    The block scale is chosen as in version 2 (signed maximum / -8), then
    replaced by its nearest code, and the weights are quantised against
    the DECODED scale so the nibbles absorb part of the code error."""
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] % QK:
        raise ValueError(f"Q4 shape must be [rows, cols divisible by {QK}], got {matrix.shape}")
    rows, cols = matrix.shape
    blocks = np.ascontiguousarray(matrix).reshape(rows, cols // QK, QK)
    maxima = np.argmax(np.abs(blocks), axis=2)
    signed_max = np.take_along_axis(blocks, maxima[..., None], axis=2)[..., 0]
    scales = (signed_max / -8.0).astype(np.float32)
    inverse = np.zeros_like(scales)
    np.divide(1.0, scales, out=inverse, where=scales != 0.0)
    first = np.clip(np.trunc(blocks * inverse[..., None] + 8.5), 0, 15) - 8
    energy = np.sum(first * first, axis=2)
    m, k, codes, decoded = choose_scale_code(scales, energy)
    inverse = np.zeros_like(decoded)
    np.divide(1.0, decoded, out=inverse, where=decoded != 0.0)
    quants = np.clip(np.trunc(blocks * inverse[..., None] + 8.5), 0, 15)
    quants = quants.astype(np.uint8).reshape(rows, cols)
    return m, k, pack_groups_v3(codes, quants)


def q4_0_bytes(values: np.ndarray) -> bytes:
    """The version-1/2 quantiser (fp16 scales, row layout); no longer
    written, kept as the reference for what versions 1 and 2 contain."""
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] % QK:
        raise ValueError(f"Q4 shape must be [rows, cols divisible by {QK}], got {matrix.shape}")
    rows, cols = matrix.shape
    blocks = np.ascontiguousarray(matrix).reshape(rows, cols // QK, QK)
    maxima = np.argmax(np.abs(blocks), axis=2)
    signed_max = np.take_along_axis(blocks, maxima[..., None], axis=2)[..., 0]
    scales = signed_max / -8.0
    inverse = np.zeros_like(scales)
    np.divide(1.0, scales, out=inverse, where=scales != 0.0)
    quants = np.trunc(blocks * inverse[..., None] + 8.5).astype(np.int16)
    quants = np.clip(quants, 0, 15).astype(np.uint8)
    packed = quants[..., :16] | (quants[..., 16:] << 4)
    output = np.empty((rows, cols // QK, 18), dtype=np.uint8)
    output[..., :2] = scales.astype("<f2").view(np.uint8).reshape(rows, cols // QK, 2)
    output[..., 2:] = packed
    return output.tobytes()

def bytes_to_unicode() -> dict[int, str]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    unicode_values = list(byte_values)
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            unicode_values.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, map(chr, unicode_values)))

def tokenizer_section(path: Path, vocab_size: int) -> tuple[bytes, int, dict[str, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    model = data["model"]
    if model.get("type") != "BPE":
        raise ValueError("FalconPSP requires a BPE tokenizer")
    vocab = model["vocab"]
    if len(vocab) != vocab_size:
        raise ValueError(f"tokenizer has {len(vocab)} entries, expected {vocab_size}")
    tokens: list[str | None] = [None] * vocab_size
    for text, token_id in vocab.items():
        tokens[token_id] = text
    if any(text is None for text in tokens):
        raise ValueError("tokenizer vocabulary has holes")
    encoder = bytes_to_unicode()
    decoder = {character: value for value, character in encoder.items()}

    def raw_piece(text: str) -> bytes:
        try:
            return bytes(decoder[c] for c in text)
        except KeyError:
            return text.encode("utf-8")

    pieces = [raw_piece(text) for text in tokens]
    max_piece = max(map(len, pieces))
    offsets = [0]
    piece_blob = bytearray()
    for piece in pieces:
        piece_blob.extend(piece)
        offsets.append(len(piece_blob))
    byte_ids = []
    for value in range(256):
        token_text = encoder[value]
        # Falcon omits 13 byte values that cannot occur in valid UTF-8.
        # Map unreachable entries to PAD and preserve all valid bytes.
        byte_ids.append(vocab.get(token_text, 0))
    merge_rank = [0xFFFFFFFF] * vocab_size
    for rank, merge in enumerate(model.get("merges", [])):
        left, right = merge if isinstance(merge, list) else merge.split(" ", 1)
        combined = left + right
        if combined not in vocab:
            raise ValueError(f"merge {rank} produces missing token {combined!r}")
        merge_rank[vocab[combined]] = rank
    sorted_ids = sorted(range(vocab_size), key=lambda token_id: pieces[token_id])
    header_size = 44
    offsets_off = header_size
    offsets_blob = struct.pack(f"<{vocab_size + 1}I", *offsets)
    pieces_off = offsets_off + len(offsets_blob)
    sorted_off = (pieces_off + len(piece_blob) + 1) & ~1
    ranks_off = (sorted_off + vocab_size * 2 + 3) & ~3
    byte_ids_off = ranks_off + vocab_size * 4
    section_bytes = byte_ids_off + 256 * 2
    section = bytearray(section_bytes)
    struct.pack_into("<4s10I", section, 0, b"TOK2", 1, vocab_size,
                     len(piece_blob), offsets_off, pieces_off, sorted_off,
                     ranks_off, byte_ids_off, section_bytes, max_piece)
    section[offsets_off:pieces_off] = offsets_blob
    section[pieces_off:pieces_off + len(piece_blob)] = piece_blob
    struct.pack_into(f"<{vocab_size}H", section, sorted_off, *sorted_ids)
    struct.pack_into(f"<{vocab_size}I", section, ranks_off, *merge_rank)
    struct.pack_into("<256H", section, byte_ids_off, *byte_ids)
    specials = {name: vocab[name] for name in
                ("<|begin_of_text|>", "<|end_of_text|>", "<|eom_id|>",
                 "<|eot_id|>", "<|im_start|>", "<|im_end|>")}
    return bytes(section), max_piece, specials

def align_file(file, alignment: int = 64) -> None:
    padding = (-file.tell()) % alignment
    if padding:
        file.write(b"\0" * padding)

def write_f32(file, values: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    values = np.asarray(values)
    if tuple(values.shape) != shape:
        raise ValueError(f"{name} shape {values.shape}, expected {shape}")
    file.write(values.astype("<f4", copy=False).tobytes())

def write_q4_matrix(file, values: np.ndarray, shape: tuple[int, int],
                    name: str, weights_offset: int,
                    params: list[tuple[int, int, int]]) -> None:
    if tuple(values.shape) != shape:
        raise ValueError(f"{name} shape {values.shape}, expected {shape}")
    # The PSP kernel loads tile groups with lv.q, which faults on real
    # hardware unless the address is 16-byte aligned.
    if (file.tell() - weights_offset) % 16 or shape[0] % GROUP_ROWS:
        raise ValueError(f"{name}: Q4 matrix not 16-byte aligned or rows "
                         f"not a multiple of {GROUP_ROWS}")
    m, k, data = q4_v3(values)
    print(f"[quantize] {name:<55} {values.shape} scale code m={m} k={k}")
    params.append((file.tell() - weights_offset, m, k))
    file.write(data)


def qparam_table(params: list[tuple[int, int, int]]) -> bytes:
    """One 8-byte entry per Q4 matrix in file order: u32 offset relative to
    the weight region, u8 mantissa bits m, u8 0, u16 bias k."""
    out = bytearray()
    for offset, m, k in params:
        if not scale_code_valid(m, k):
            raise ValueError(f"invalid scale code parameters m={m} k={k}")
        out += struct.pack("<IBBH", offset, m, 0, k)
    return bytes(out)


def q4_matrix_count(layers: int) -> int:
    return 1 + 9 * layers

def require_config(config: dict) -> None:
    expected = {
        "model_type": "falcon_h1", "hidden_size": 512,
        "intermediate_size": 768, "num_hidden_layers": 24,
        "num_attention_heads": 8, "num_key_value_heads": 2,
        "head_dim": 64, "vocab_size": 32768, "mamba_d_ssm": 768,
        "mamba_d_conv": 4, "mamba_d_head": 32,
        "mamba_d_state": 64, "mamba_n_groups": 1,
        "mamba_n_heads": 24, "tie_word_embeddings": True,
        "mamba_rms_norm": False, "mamba_norm_before_gate": False,
    }
    mismatches = [f"{key}={config.get(key)!r} (need {value!r})"
                  for key, value in expected.items() if config.get(key) != value]
    if mismatches:
        raise ValueError("unsupported Falcon-H1 variant: " + ", ".join(mismatches))

def convert(model_dir: Path, output: Path, context: int) -> None:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    require_config(config)
    dim = int(config["hidden_size"])
    hidden = int(config["intermediate_size"])
    layers = int(config["num_hidden_layers"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    vocab = int(config["vocab_size"])
    model_context = int(config["max_position_embeddings"])
    m_dim = int(config["mamba_d_ssm"])
    m_heads = int(config["mamba_n_heads"])
    m_head_dim = int(config["mamba_d_head"])
    m_state = int(config["mamba_d_state"])
    m_groups = int(config["mamba_n_groups"])
    m_conv = int(config["mamba_d_conv"])
    m_conv_dim = m_dim + 2 * m_groups * m_state
    m_projection = m_dim + m_conv_dim + m_heads
    if context < 32 or context > model_context:
        raise ValueError(f"context must be 32..{model_context}")
    tokenizer, max_piece, special = tokenizer_section(model_dir / "tokenizer.json", vocab)
    tensors = SafeTensors(model_dir / "model.safetensors")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("w+b") as out:
            out.write(b"\0" * HEADER_BYTES)
            tokenizer_offset = out.tell()
            out.write(tokenizer)
            align_file(out, 16)
            qparam_offset = out.tell()
            qparam_count = q4_matrix_count(layers)
            out.write(b"\0" * (qparam_count * QPARAM_ENTRY))
            align_file(out)
            weights_offset = out.tell()
            params: list[tuple[int, int, int]] = []

            def write_q4(file, values, shape, name):
                write_q4_matrix(file, values, shape, name, weights_offset,
                                params)
            write_q4(out, tensors.tensor("model.embed_tokens.weight"),
                     (vocab, dim), "model.embed_tokens.weight")
            for layer in range(layers):
                prefix = f"model.layers.{layer}"
                write_f32(out, tensors.tensor(f"{prefix}.input_layernorm.weight"),
                          (dim,), f"{prefix}.input_layernorm.weight")
                write_q4(out, tensors.tensor(f"{prefix}.self_attn.q_proj.weight"),
                         (heads * head_dim, dim), f"{prefix}.self_attn.q_proj.weight")
                write_q4(out, tensors.tensor(f"{prefix}.self_attn.k_proj.weight"),
                         (kv_heads * head_dim, dim), f"{prefix}.self_attn.k_proj.weight")
                write_q4(out, tensors.tensor(f"{prefix}.self_attn.v_proj.weight"),
                         (kv_heads * head_dim, dim), f"{prefix}.self_attn.v_proj.weight")
                write_q4(out, tensors.tensor(f"{prefix}.self_attn.o_proj.weight"),
                         (dim, heads * head_dim), f"{prefix}.self_attn.o_proj.weight")
                write_q4(out, tensors.tensor(f"{prefix}.mamba.in_proj.weight"),
                         (m_projection, dim), f"{prefix}.mamba.in_proj.weight")
                conv = tensors.tensor(f"{prefix}.mamba.conv1d.weight")
                write_f32(out, np.squeeze(conv, axis=1), (m_conv_dim, m_conv),
                          f"{prefix}.mamba.conv1d.weight")
                write_f32(out, tensors.tensor(f"{prefix}.mamba.conv1d.bias"),
                          (m_conv_dim,), f"{prefix}.mamba.conv1d.bias")
                write_f32(out, tensors.tensor(f"{prefix}.mamba.dt_bias"),
                          (m_heads,), f"{prefix}.mamba.dt_bias")
                write_f32(out, tensors.tensor(f"{prefix}.mamba.A_log"),
                          (m_heads,), f"{prefix}.mamba.A_log")
                write_f32(out, tensors.tensor(f"{prefix}.mamba.D"),
                          (m_heads,), f"{prefix}.mamba.D")
                write_q4(out, tensors.tensor(f"{prefix}.mamba.out_proj.weight"),
                         (dim, m_dim), f"{prefix}.mamba.out_proj.weight")
                write_f32(out, tensors.tensor(f"{prefix}.pre_ff_layernorm.weight"),
                          (dim,), f"{prefix}.pre_ff_layernorm.weight")
                write_q4(out, tensors.tensor(f"{prefix}.feed_forward.gate_proj.weight"),
                         (hidden, dim), f"{prefix}.feed_forward.gate_proj.weight")
                write_q4(out, tensors.tensor(f"{prefix}.feed_forward.up_proj.weight"),
                         (hidden, dim), f"{prefix}.feed_forward.up_proj.weight")
                write_q4(out, tensors.tensor(f"{prefix}.feed_forward.down_proj.weight"),
                         (dim, hidden), f"{prefix}.feed_forward.down_proj.weight")
            write_f32(out, tensors.tensor("model.final_layernorm.weight"),
                      (dim,), "model.final_layernorm.weight")
            file_size = out.tell()
            header = bytearray(HEADER_BYTES)
            struct.pack_into("<4sIIIQQQ", header, 0, MAGIC, VERSION, HEADER_BYTES, 0,
                             file_size, tokenizer_offset, weights_offset)
            integers = (
                dim, hidden, layers, heads, kv_heads, head_dim, vocab,
                model_context, context, int(config["pad_token_id"]),
                int(config["eos_token_id"]), special["<|begin_of_text|>"],
                special["<|im_start|>"], special["<|im_end|>"],
                special["<|eom_id|>"], special["<|eot_id|>"],
                m_dim, m_heads, m_head_dim, m_state, m_groups, m_conv,
                m_projection, QK, max_piece,
            )
            struct.pack_into("<25I", header, 40, *integers)
            struct.pack_into("<4f", header, 140,
                             float(config["rope_theta"]), float(config["rms_norm_eps"]),
                             float(config["embedding_multiplier"]),
                             float(config["lm_head_multiplier"]))
            if len(params) != qparam_count:
                raise ValueError("Q4 matrix count does not match the table")
            struct.pack_into("<3I", header, QPARAM_HEADER, qparam_offset,
                             qparam_count, QPARAM_ENTRY)
            out.seek(qparam_offset)
            out.write(qparam_table(params))
            out.seek(0)
            out.write(header)
    finally:
        tensors.close()
    digest = hashlib.sha256()
    with output.open("rb") as converted:
        while chunk := converted.read(1024 * 1024):
            digest.update(chunk)
    manifest = {
        "format": "FalconH1PSP streamed FHQ4", "version": VERSION,
        "source_model": DEFAULT_MODEL, "output": output.name,
        "bytes": output.stat().st_size, "sha256": digest.hexdigest(),
        "architecture": {"dim": dim, "hidden_dim": hidden, "layers": layers,
                         "attention_heads": heads, "kv_heads": kv_heads,
                         "mamba_dim": m_dim, "mamba_heads": m_heads,
                         "mamba_state": m_state, "vocab": vocab,
                         "model_context": model_context, "psp_context": context},
        "quantization": ("Q4_0 with 8-bit minifloat block-scale codes in "
                         "four-row tile groups (FHQ4 version 3, 17 bytes "
                         "per 32 weights)"),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {output} ({output.stat().st_size / 1024 / 1024:.2f} MiB)")
    print(f"[done] sha256 {digest.hexdigest()}")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=Path("models/falcon-h1-tiny-90m"))
    parser.add_argument("--output", type=Path,
                        default=Path("model.fhq4"))
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    args = parser.parse_args()
    model_dir = args.model_dir
    if args.download:
        model_dir = model_dir or args.cache_dir
        download_model(args.model_id, model_dir)
    if model_dir is None:
        parser.error("use --model-dir or --download")
    missing = [name for name in FILES if not (model_dir / name).exists()]
    if missing:
        parser.error(f"missing from {model_dir}: {', '.join(missing)}")
    convert(model_dir, args.output, args.context)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

