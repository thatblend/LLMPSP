#!/usr/bin/env python3
"""Re-pack an FHQ4 model to a newer format version without the checkpoint.

    python tools/repack_fhq4.py model_v2.fhq4 model.fhq4          (-> v3)
    python tools/repack_fhq4.py --to 2 model_v1.fhq4 model_v2.fhq4
    python tools/repack_fhq4.py --check model_v2.fhq4 model.fhq4

Version 1 -> 2 is lossless: the same scales and nibbles rearranged into
four-row tile groups (18 bytes per 32 weights). Version 2 -> 3 keeps every
nibble and float tensor but replaces each fp16 block scale by an 8-bit
code (17 bytes per 32 weights, PROJECT.md section 2.10): per matrix the
tool picks the mantissa width m and bias k that minimise the added weight
error energy, then stores the nearest code of every scale. A version-1
input is taken through version 2 first.

Every run ends with the --check comparison, which decodes both files with
code independent of the writers (byte offsets by the format formula), and
requires: identical nibbles and float tensors, identical header and
tokenizer bytes apart from the version field and the version-3 table
fields, and for 1 -> 2 identical scales. For 2 -> 3 it reports the scale
error statistics per matrix type. NumPy only.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import struct
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_falcon_h1 import (GROUP_ROWS, HEADER_BYTES, MAGIC, QK,  # noqa: E402
                               QPARAM_ENTRY, QPARAM_HEADER, choose_scale_code,
                               pack_groups_v3, q4_v3, qparam_table,
                               rows_to_tiles, scale_code_valid, scale_table)


def read_header(data: bytes) -> dict:
    magic, version, header_bytes, _, file_size, tok_off, weights_off = \
        struct.unpack_from("<4sIIIQQQ", data, 0)
    if magic != MAGIC or header_bytes != HEADER_BYTES:
        raise ValueError("not an FHQ4 file")
    fields = struct.unpack_from("<25I", data, 40)
    names = ("dim", "hidden", "layers", "heads", "kv_heads", "head_dim",
             "vocab", "model_context", "context", "pad", "eos", "bos",
             "im_start", "im_end", "eom", "eot", "m_dim", "m_heads",
             "m_head_dim", "m_state", "m_groups", "m_conv", "m_projection",
             "qk", "max_piece")
    header = dict(zip(names, fields))
    header.update(version=version, file_size=file_size,
                  tokenizer_offset=tok_off, weights_offset=weights_off)
    if header["qk"] != QK:
        raise ValueError("unexpected Q4 block size")
    if file_size != len(data):
        raise ValueError(f"header says {file_size} bytes, file has {len(data)}")
    if version >= 3:
        (header["qparam_offset"], header["qparam_count"],
         header["qparam_entry"]) = struct.unpack_from("<3I", data,
                                                      QPARAM_HEADER)
    return header


def tensor_layout(h: dict) -> list[tuple[str, str, tuple[int, ...]]]:
    """(name, kind, shape) in file order; mirrors falcon_model_open."""
    dim, hidden = h["dim"], h["hidden"]
    conv_dim = h["m_dim"] + 2 * h["m_groups"] * h["m_state"]
    out = [("embedding", "q4", (h["vocab"], dim))]
    for i in range(h["layers"]):
        p = f"layer{i}."
        out += [
            (p + "input_norm", "f32", (dim,)),
            (p + "wq", "q4", (h["heads"] * h["head_dim"], dim)),
            (p + "wk", "q4", (h["kv_heads"] * h["head_dim"], dim)),
            (p + "wv", "q4", (h["kv_heads"] * h["head_dim"], dim)),
            (p + "wo", "q4", (dim, h["heads"] * h["head_dim"])),
            (p + "mamba_in", "q4", (h["m_projection"], dim)),
            (p + "conv_weight", "f32", (conv_dim * h["m_conv"],)),
            (p + "conv_bias", "f32", (conv_dim,)),
            (p + "dt_bias", "f32", (h["m_heads"],)),
            (p + "a_log", "f32", (h["m_heads"],)),
            (p + "d", "f32", (h["m_heads"],)),
            (p + "mamba_out", "q4", (dim, h["m_dim"])),
            (p + "ffn_norm", "f32", (dim,)),
            (p + "w_gate", "q4", (hidden, dim)),
            (p + "w_up", "q4", (hidden, dim)),
            (p + "w_down", "q4", (dim, hidden)),
        ]
    out.append(("final_norm", "f32", (dim,)))
    return out


def tensor_bytes(kind: str, shape: tuple[int, ...], version: int) -> int:
    if kind == "f32":
        return shape[0] * 4
    rows, cols = shape
    return rows * (cols // QK) * (17 if version >= 3 else 18)


def walk(h: dict):
    """Yields (name, kind, shape, offset, size) and checks the total."""
    offset = h["weights_offset"]
    for name, kind, shape in tensor_layout(h):
        size = tensor_bytes(kind, shape, h["version"])
        yield name, kind, shape, offset, size
        offset += size
    if offset != h["file_size"]:
        raise ValueError(f"layout ends at {offset}, file is {h['file_size']}")


def decode_v1(raw: bytes, rows: int, cols: int):
    blocks = cols // QK
    d = np.frombuffer(raw, dtype=np.uint8).reshape(rows, blocks, 18)
    scales = d[..., 0].astype(np.uint16) | (d[..., 1].astype(np.uint16) << 8)
    packed = d[..., 2:]
    nibbles = np.concatenate([packed & 15, packed >> 4], axis=2)
    return scales, nibbles.reshape(rows, cols)


def decode_v2(raw: bytes, rows: int, cols: int):
    """Independent of rows_to_tiles: indexes bytes by the format formula.
    Returns (fp16 scale bit patterns [rows, blocks], nibbles [rows, cols])."""
    blocks = cols // QK
    groups = rows // GROUP_ROWS
    g = np.frombuffer(raw, dtype=np.uint8).reshape(groups, blocks * 72)
    r = np.arange(GROUP_ROWS)[:, None]
    b = np.arange(blocks)[None, :]
    scale_index = b * 8 + r * 2                                   # [4, B]
    scales = (g[:, scale_index].astype(np.uint16) |
              (g[:, scale_index + 1].astype(np.uint16) << 8))
    j = np.arange(16)[None, None, :]
    tile_index = (blocks * 8 + b[..., None] * 64 + (j // 4) * 16 +
                  r[..., None] * 4 + (j % 4))                     # [4, B, 16]
    byte = g[:, tile_index] ^ np.uint8(0x80)                      # [G,4,B,16]
    nibbles = np.concatenate([byte & 15, byte >> 4], axis=3)
    return scales.reshape(rows, blocks), nibbles.reshape(rows, cols)


def decode_v3(raw: bytes, rows: int, cols: int):
    """Independent of pack_groups_v3. Returns (int8 codes [rows, blocks],
    nibbles [rows, cols])."""
    blocks = cols // QK
    groups = rows // GROUP_ROWS
    g = np.frombuffer(raw, dtype=np.uint8).reshape(groups, blocks * 68)
    r = np.arange(GROUP_ROWS)[:, None]
    b = np.arange(blocks)[None, :]
    codes = g[:, b * 4 + r].view(np.int8)                         # [G,4,B]
    j = np.arange(16)[None, None, :]
    tile_index = (blocks * 4 + b[..., None] * 64 + (j // 4) * 16 +
                  r[..., None] * 4 + (j % 4))
    byte = g[:, tile_index] ^ np.uint8(0x80)
    nibbles = np.concatenate([byte & 15, byte >> 4], axis=3)
    return codes.reshape(rows, blocks), nibbles.reshape(rows, cols)


def decode_code_value(codes: np.ndarray, m: int, k: int) -> np.ndarray:
    """The version-3 decode rule, written out independently of scale_table:
    sign(c) * float_from_bits((|c| + k) << (23 - m)), 0 for c == 0."""
    c = codes.astype(np.int64)
    bits = ((np.abs(c) + k) << (23 - m)).astype(np.uint32)
    value = bits.view(np.float32).astype(np.float64)
    return np.where(c == 0, 0.0, np.where(c < 0, -value, value))


def repack_v1_to_v2(src: bytes) -> bytes:
    h = read_header(src)
    if h["version"] != 1:
        raise ValueError(f"input is FHQ4 version {h['version']}, expected 1")
    if h["weights_offset"] % 16:
        raise ValueError("weight region is not 16-byte aligned")
    out = bytearray(src[:h["weights_offset"]])
    struct.pack_into("<I", out, 4, 2)
    for name, kind, shape, offset, size in walk(h):
        chunk = src[offset:offset + size]
        if kind == "q4":
            if (offset - h["weights_offset"]) % 16 or shape[0] % GROUP_ROWS:
                raise ValueError(f"{name}: not 16-byte aligned 4-row groups")
            chunk = rows_to_tiles(chunk, *shape)
        if len(chunk) != size:
            raise ValueError(f"{name}: size changed")
        out += chunk
    return bytes(out)


def v2_matrix(h: dict, src: bytes, offset: int, size: int, shape):
    """fp16 scales as float32, nibbles, and the squared-weight energy of
    every block (what the scale error is multiplied by)."""
    s16, nib = decode_v2(src[offset:offset + size], *shape)
    scales = s16.view(np.float16).astype(np.float32)
    q = nib.astype(np.int32).reshape(shape[0], shape[1] // QK, QK) - 8
    energy = np.sum(q * q, axis=2)
    return scales, nib, energy


def repack_v2_to_v3(src: bytes, log=print) -> bytes:
    h = read_header(src)
    if h["version"] != 2:
        raise ValueError(f"input is FHQ4 version {h['version']}, expected 2")
    q4_count = sum(1 for _, kind, _ in tensor_layout(h) if kind == "q4")
    tok_end = h["tokenizer_offset"] + struct.unpack_from(
        "<I", src, h["tokenizer_offset"] + 36)[0]
    qparam_offset = (tok_end + 15) & ~15
    weights_offset = (qparam_offset + q4_count * QPARAM_ENTRY + 63) & ~63
    head = bytearray(src[:tok_end])
    head += b"\0" * (weights_offset - tok_end)
    struct.pack_into("<I", head, 4, 3)
    struct.pack_into("<Q", head, 32, weights_offset)
    struct.pack_into("<3I", head, QPARAM_HEADER, qparam_offset, q4_count,
                     QPARAM_ENTRY)
    body = bytearray()
    params = []
    for name, kind, shape, offset, size in walk(h):
        if kind == "f32":
            body += src[offset:offset + size]
            continue
        if len(body) % 16 or shape[0] % GROUP_ROWS or shape[1] % (4 * QK):
            raise ValueError(f"{name}: not a 16-byte aligned version-3 matrix")
        scales, nib, energy = v2_matrix(h, src, offset, size, shape)
        m, k, codes, _ = choose_scale_code(scales, energy)
        params.append((len(body), m, k))
        body += pack_groups_v3(codes, nib)
        log(f"[repack] {name:<20} m={m} k={k}")
    table = qparam_table(params)
    head[qparam_offset:qparam_offset + len(table)] = table
    out = bytes(head) + bytes(body)
    struct.pack_into("<Q", head, 16, len(out))
    return bytes(head) + bytes(body)


def check_v1_v2(v1: bytes, v2: bytes) -> None:
    h1, h2 = read_header(v1), read_header(v2)
    if h1["version"] != 1 or h2["version"] != 2:
        raise SystemExit(f"FAIL: versions {h1['version']} / {h2['version']}")
    if len(v1) != len(v2):
        raise SystemExit("FAIL: file sizes differ")
    head1 = bytearray(v1[:h1["weights_offset"]])
    head2 = bytearray(v2[:h2["weights_offset"]])
    struct.pack_into("<I", head2, 4, 1)
    if head1 != head2:
        raise SystemExit("FAIL: header/tokenizer bytes differ")
    matrices = weights = f32 = 0
    for (name, kind, shape, offset, size), other in zip(walk(h1), walk(h2)):
        if other[3] != offset:
            raise SystemExit(f"FAIL: {name} offsets differ")
        a, b = v1[offset:offset + size], v2[offset:offset + size]
        if kind == "f32":
            if a != b:
                raise SystemExit(f"FAIL: {name} float tensor differs")
            f32 += 1
            continue
        s1, n1 = decode_v1(a, *shape)
        s2, n2 = decode_v2(b, *shape)
        if not (np.array_equal(s1, s2) and np.array_equal(n1, n2)):
            raise SystemExit(f"FAIL: {name} decodes differently")
        matrices += 1
        weights += shape[0] * shape[1]
    print(f"[check] lossless: {matrices} Q4 matrices ({weights:,} weights, "
          f"{weights // QK:,} scales) decode identically; {f32} float "
          f"tensors and {h1['weights_offset']} header/tokenizer bytes "
          "match (version field aside)")


def check_v2_v3(v2: bytes, v3: bytes) -> None:
    h2, h3 = read_header(v2), read_header(v3)
    if h2["version"] != 2 or h3["version"] != 3:
        raise SystemExit(f"FAIL: versions {h2['version']} / {h3['version']}")
    tok_bytes = struct.unpack_from("<I", v2, h2["tokenizer_offset"] + 36)[0]
    tok_end = h2["tokenizer_offset"] + tok_bytes
    a = bytearray(v2[:tok_end])
    b = bytearray(v3[:tok_end])
    for buf in (a, b):
        struct.pack_into("<I", buf, 4, 0)            # version
        struct.pack_into("<QQ", buf, 16, 0, 0)       # file size (+tok off)
        struct.pack_into("<Q", buf, 32, 0)           # weights offset
        struct.pack_into("<3I", buf, QPARAM_HEADER, 0, 0, 0)
    if a != b or h2["tokenizer_offset"] != h3["tokenizer_offset"]:
        raise SystemExit("FAIL: header/tokenizer bytes differ")
    entries = [struct.unpack_from("<IBBH", v3, h3["qparam_offset"] + i * 8)
               for i in range(h3["qparam_count"])]
    if (h3["qparam_entry"] != QPARAM_ENTRY or
            h3["qparam_offset"] % 16 or h3["weights_offset"] % 64 or
            h3["qparam_offset"] < tok_end or
            h3["qparam_offset"] + 8 * len(entries) > h3["weights_offset"]):
        raise SystemExit("FAIL: version-3 table placement")
    q4 = [(n, s, o, z) for n, k_, s, o, z in walk(h3) if k_ == "q4"]
    if len(entries) != len(q4):
        raise SystemExit("FAIL: table entry count")
    matrices = weights = f32 = 0
    stats: dict[str, list] = {}
    all_rel = []
    for (name, kind, shape, off2, size2), (_, _, _, off3, size3) in \
            zip(walk(h2), walk(h3)):
        if kind == "f32":
            if v2[off2:off2 + size2] != v3[off3:off3 + size3]:
                raise SystemExit(f"FAIL: {name} float tensor differs")
            f32 += 1
            continue
        entry = entries[matrices]
        rel_off, m, pad, k = entry
        if (rel_off != off3 - h3["weights_offset"] or pad or
                not scale_code_valid(m, k) or rel_off % 16):
            raise SystemExit(f"FAIL: {name} table entry {entry}")
        scales, n2, energy = v2_matrix(h2, v2, off2, size2, shape)
        codes, n3 = decode_v3(v3[off3:off3 + size3], *shape)
        if not np.array_equal(n2, n3):
            raise SystemExit(f"FAIL: {name} nibbles differ")
        if np.any(codes == -128):
            raise SystemExit(f"FAIL: {name} uses code -128")
        decoded = decode_code_value(codes, m, k)
        same = scale_table(m, k)[codes.view(np.uint8)].astype(np.float64)
        if not np.array_equal(decoded, same):
            raise SystemExit(f"FAIL: {name} decode rules disagree")
        s = scales.astype(np.float64)
        nz = s != 0
        if np.any(np.sign(decoded[nz]) * np.sign(s[nz]) < 0):
            raise SystemExit(f"FAIL: {name} scale sign flipped")
        rel = np.abs(decoded[nz] - s[nz]) / np.abs(s[nz])
        zeroed = int(np.sum(nz & (decoded == 0)))
        # Added weight-error energy relative to the Q4 rounding noise of the
        # same blocks (uniform rounding: scale^2 / 12 per weight).
        added = float(np.sum(energy * (decoded - s) ** 2))
        noise = float(np.sum(s * s) * QK / 12.0)
        kind_name = name.split(".")[-1]
        stats.setdefault(kind_name, []).append(
            (rel, zeroed, added, noise, m))
        all_rel.append(rel)
        matrices += 1
        weights += shape[0] * shape[1]
    print(f"[check] {matrices} Q4 matrices ({weights:,} weights, "
          f"{weights // QK:,} scales): nibbles identical; {f32} float tensors "
          "and the header/tokenizer bytes identical (version, size, offset "
          "and table fields aside)")
    print("[check] scale code error per matrix type (relative |d'-d|/|d|; "
          "energy = added weight error / Q4 rounding noise):")
    print(f"        {'type':<10} {'m used':<12} {'mean':>8} {'p99':>8} "
          f"{'max':>8} {'zeroed':>7} {'energy':>8}")
    for kind_name, items in stats.items():
        rel = np.concatenate([x[0] for x in items])
        ms = sorted({x[4] for x in items})
        mcount = {mm: sum(1 for x in items if x[4] == mm) for mm in ms}
        print(f"        {kind_name:<10} "
              f"{','.join(f'{mm}x{mcount[mm]}' for mm in ms):<12} "
              f"{100 * rel.mean():7.3f}% {100 * np.percentile(rel, 99):7.3f}% "
              f"{100 * rel.max():7.2f}% {sum(x[1] for x in items):7d} "
              f"{sum(x[2] for x in items) / sum(x[3] for x in items):8.5f}")
    rel = np.concatenate(all_rel)
    print(f"        {'all':<10} {'':<12} {100 * rel.mean():7.3f}% "
          f"{100 * np.percentile(rel, 99):7.3f}% {100 * rel.max():7.2f}%")


def converter_self_test(v2: bytes, v3: bytes, names=("embedding",
                        "layer0.wv", "layer7.mamba_out")) -> None:
    """The converter's quantiser (q4_v3), fed the version-2 weights as
    floats, must reproduce the re-packed matrix: same (m, k), same codes,
    and the same decoded weights (the scale is the signed maximum / -8 and
    the nibbles move by less than half a step; blocks whose code is 0 hold
    zero weights either way)."""
    h2, h3 = read_header(v2), read_header(v3)
    layout3 = {n: (o, z, s) for n, k_, s, o, z in walk(h3)}
    q4_names = [n for n, k_, s, o, z in walk(h3) if k_ == "q4"]
    for name, kind, shape, off, size in walk(h2):
        if name not in names:
            continue
        scales, nib, _ = v2_matrix(h2, v2, off, size, shape)
        weights = (scales[..., None] *
                   (nib.reshape(shape[0], -1, QK).astype(np.float32) - 8))
        m, k, data = q4_v3(weights.reshape(shape))
        o3, z3, _ = layout3[name]
        entry = struct.unpack_from("<IBBH", v3, h3["qparam_offset"] +
                                   8 * q4_names.index(name))
        c_new, n_new = decode_v3(data, *shape)
        c_old, n_old = decode_v3(v3[o3:o3 + z3], *shape)
        w_new = (decode_code_value(c_new, m, k)[..., None] *
                 (n_new.reshape(shape[0], -1, QK).astype(np.float64) - 8))
        w_old = (decode_code_value(c_old, entry[1], entry[3])[..., None] *
                 (n_old.reshape(shape[0], -1, QK).astype(np.float64) - 8))
        # Where the code moved the scale by 1/16 or more (blocks clipped at
        # the bottom of the code range), the converter rightly re-rounds
        # the nibbles against the decoded scale; elsewhere every weight
        # must come out identical.
        d = decode_code_value(c_old, entry[1], entry[3])
        s = scales.astype(np.float64)
        same = (d == 0) | (np.abs(s / np.where(d == 0, 1, d) - 1) < 1 / 16)
        moved = int(np.sum(~same))
        if ((m, k) != (entry[1], entry[3]) or
                not np.array_equal(c_new, c_old) or
                not np.array_equal(w_new[same], w_old[same])):
            raise SystemExit(f"FAIL: converter path differs for {name}")
        print(f"[check] converter quantiser reproduces the re-pack for "
              f"{name} ({moved} clipped blocks re-rounded)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="only compare an existing pair")
    parser.add_argument("--to", type=int, default=3, choices=(2, 3),
                        help="output format version (default 3)")
    parser.add_argument("input", type=Path, help="FHQ4 version-1 or -2 file")
    parser.add_argument("output", type=Path, help="FHQ4 output file")
    args = parser.parse_args()
    src = args.input.read_bytes()
    version = read_header(src)["version"]
    if args.check:
        out = args.output.read_bytes()
        target = read_header(out)["version"]
    else:
        if args.output.resolve() == args.input.resolve():
            parser.error("write to a different file, then replace")
        target = args.to
        if version >= target:
            parser.error(f"input is already version {version}")
        out = src
        if version == 1:
            out = repack_v1_to_v2(out)
            if target == 2:
                check_v1_v2(src, out)
        if target == 3:
            out = repack_v2_to_v3(out)
        args.output.write_bytes(out)
        print(f"[repack] wrote {args.output} ({len(out)} bytes)")
        print(f"[repack] sha256 {hashlib.sha256(out).hexdigest()}")
    if version == 1 and target == 2:
        check_v1_v2(src, out)
    elif target == 3:
        v2 = repack_v1_to_v2(src) if version == 1 else src
        if version == 1:
            check_v1_v2(src, v2)
        check_v2_v3(v2, out)
        converter_self_test(v2, out)
    else:
        raise SystemExit(f"FAIL: no check for version {version} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
