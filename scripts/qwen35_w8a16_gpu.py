#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# State-transfer math/asset contracts: PrismPhi Qwen3.5 Q6A port (Apache-2.0).
# This is an independent CUDA Q/DQ bridge and portable NumPy host loop.
"""Experimental GPU execution of the frozen Q6A Qwen3.5 quantization recipe.

This preserves the original integer tensor types, weights, scales, zero points,
and quantization boundaries. Custom CUDA kernels implement integer Q/DQ;
matrix/convolution math on CUDA is FP32, NOT native W8A16. No speed/accuracy claim
is made. The published QNN contexts can be run with the same Python host loop.

Source contracts/state rules (Apache-2.0), pinned to upstream commit
97650f6ed3a1178dcf1ff761d95e4523819bfe08:
https://github.com/PrismPhi/radxa-dragon-q6a-qwen3.5-0.8b-qcs6490-qnn-npu
The CUDA build targets Jetson Orin (SM87). Each command writes to a new output
directory. check builds only the small extension, not ONNX Runtime or the LLM.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import onnx
from onnx import TensorProto as T, helper as H, numpy_helper as NH
import onnxruntime as ort

REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
SOURCE_REPO = "PrismPhi/Qwen3.5-0.8B-Radxa-Dragon-Q6A-QCS6490-QNN-NPU"
ASSET_REVISION = "05de67d2f33c0cb83e375f13788895c761d30b1e"
GRAPH = "sources/common/2k_decode.onnx"
WEIGHTS = "sources/common/language_weights.data"
PINS = {
    GRAPH: "3f02d88c99bf809832c0827860f595a8267ba18e9f2cf5f182c73199d93bd428",
    WEIGHTS: "b0999d1b4fc79342aa21ec190797cfe322f207af28b695db195ead187acff62d",
    "common/embedding_fp32.npy": "6a647a0e81e9586346987a22cfe4ce06d36279fe0ab35805acc3f7cae018358a",
    "common/tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "contracts/2k/rope_ctx2048_cos_sin_fp32.bin": "09aab45c00a59ec4a209cc09ca3176a80c6ec1fbe8f4f7f9f360203d000dccc4",
    "contracts/2k/tensor_contract.tsv": "db6fd93f9a6ec451ebd238ca36416893f21b6f8752a3c4aefd7fccec4337d0ff",
    "contracts/2k/advance_plan.tsv": "c883de8d7c9fbe317dcda67c5c3470fb78c0450dbd32868caedebd845fb399a8",
    "models/2k/decode_ctx.onnx": "caff049aceadcb024fcc2d9a0d02864200b286f2fda764db6893c2dc3a92cf18",
    "models/2k/decode_ctx_qnn.bin": "b2b636e939c3da86898d76186af3e5aa9eba173302d9a8deb9ea0dd37194ad75",
}
HOST_FILES = tuple(k for k in PINS if k.startswith(("common/", "contracts/")))
INTEGER_TYPES = {T.INT8, T.UINT8, T.INT16, T.UINT16}
CUSTOM_DOMAIN = "airgap.quant"


def compact_qdq(original):
    """Replace Q/DQ implementation names; preserve weights, grids and topology."""
    model = copy.deepcopy(original)
    g = model.graph
    if g.sparse_initializer or model.functions or any(
            a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for n in g.node for a in n.attribute):
        raise ValueError("Sparse tensors, functions and nested graphs need review")
    initializers = {t.name: t for t in g.initializer}
    counts = {}
    for n in g.node:
        if n.op_type not in ("QuantizeLinear", "DequantizeLinear") or n.domain not in ("", "ai.onnx", "com.microsoft"):
            if n.domain not in ("", "ai.onnx"):
                raise ValueError(f"Unreviewed custom operation: {n.domain}::{n.op_type}")
            continue
        if len(n.input) != 3 or len(n.output) != 1:
            raise ValueError(f"Require explicit scale and zero point: {n.name}")
        scale, zp = initializers.get(n.input[1]), initializers.get(n.input[2])
        if scale is None or zp is None or scale.external_data or zp.external_data:
            raise ValueError(f"QDQ parameters must be inline constants: {n.name}")
        if scale.data_type != T.FLOAT or zp.data_type not in (T.UINT8, T.UINT16):
            raise ValueError(f"Unreviewed compact QDQ dtype: {n.name}")
        attrs = {a.name: H.get_attribute_value(a) for a in n.attribute}
        if set(attrs) - {"axis"}:
            raise ValueError(f"Unreviewed QDQ attributes: {n.name}: {attrs}")
        sv, zv = NH.to_array(scale), NH.to_array(zp)
        if sv.shape != zv.shape or sv.ndim > 1 or not sv.size or not np.isfinite(sv).all() or np.any(sv <= 0):
            raise ValueError(f"Invalid QDQ parameters: {n.name}")
        key = n.op_type + ":" + T.DataType.Name(zp.data_type)
        counts[key] = counts.get(key, 0) + 1
        n.op_type = ("Quantize" if n.op_type == "QuantizeLinear" else "Dequantize") + ("U16" if zp.data_type == T.UINT16 else "U8")
        n.domain = CUSTOM_DOMAIN
    model.opset_import.append(H.make_opsetid(CUSTOM_DOMAIN, 1))
    if initializer_digest(model) != initializer_digest(original):
        raise RuntimeError("An original initializer changed")
    for kind in ("input", "output"):
        if [v.SerializeToString() for v in getattr(g, kind)] != [v.SerializeToString() for v in getattr(original.graph, kind)]:
            raise RuntimeError("Graph I/O changed")
    return model, {"qdq_replacements": counts, "original_nodes": len(original.graph.node),
                   "bridge_nodes": len(g.node), "original_initializer_digest": initializer_digest(original),
                   "original_initializers_unchanged": True, "original_io_unchanged": True}


def build_cuda(output, ort_include):
    """Build a small extension against the existing ORT/CUDA installation."""
    if output.exists():
        raise FileExistsError(f"Refuse to replace extension: {output}")
    if ort.__version__ != "1.27.0" or "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("Use ONNX Runtime 1.27.0 CUDA; CPU fallback is forbidden")
    source = Path(__file__).with_name("qwen35_w8a16_cuda.cu")
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not source.is_file() or not (ort_include / "onnxruntime_cxx_api.h").is_file():
        raise FileNotFoundError("Need adjacent CUDA source and ORT 1.27 headers (--ort-include)")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([nvcc, "-shared", "-Xcompiler=-fPIC,-ffp-contract=off", "-std=c++17", "-O2",
                    "--fmad=false", "--prec-div=true", "--ftz=false", "--expt-relaxed-constexpr",
                    "-arch=sm_87", "-I", str(ort_include), str(source), "-o", str(output)], check=True)
    return output


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")


def verify(root, names):
    for name in names:
        p = root / name
        if not p.is_file() or sha(p) != PINS[name]:
            raise ValueError(f"Missing or changed published asset: {p}")


def initializer_digest(model):
    h = hashlib.sha256()
    for t in model.graph.initializer:
        b = t.SerializeToString()
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()


def expand_qdq(original):
    """Lower integer Q/DQ only; fail on an unrecognized precision contract."""
    model = copy.deepcopy(original)
    g = model.graph
    if any(a.type == onnx.AttributeProto.GRAPH or a.type == onnx.AttributeProto.GRAPHS
           for n in g.node for a in n.attribute):
        raise ValueError("Nested graphs need a separate review")
    if g.sparse_initializer:
        raise ValueError("Sparse initializers are not covered")
    opsets = {o.domain: o.version for o in model.opset_import}
    if opsets.get("", 0) < 13:
        raise ValueError("Need ONNX opset >= 13")
    initializers = {t.name: t for t in g.initializer}
    old_bytes = initializer_digest(original)
    shapes = {t.name: list(t.dims) for t in g.initializer}
    for v in [*g.input, *g.value_info, *g.output]:
        if v.type.tensor_type.HasField("shape"):
            shapes[v.name] = [d.dim_value if d.HasField("dim_value") else None
                              for d in v.type.tensor_type.shape.dim]
    used = set(initializers) | {x for n in g.node for x in [*n.input, *n.output]} | {n.name for n in g.node}
    nodes, added = [], []
    constants = {}
    counts = {}

    def name(stem):
        candidate = "__q6a_gpu_" + stem
        while candidate in used:
            candidate += "_"
        used.add(candidate)
        return candidate

    def constant(stem, value):
        value = np.asarray(value)
        signature = (value.dtype.str, value.shape, value.tobytes())
        if signature in constants:
            return constants[signature]
        key = name(stem)
        added.append(NH.from_array(value, key))
        constants[signature] = key
        return key

    def node(op, inputs, stem, output=None, **attrs):
        out = output or name(stem)
        nodes.append(H.make_node(op, inputs, [out], name=name(stem + "_node"), **attrs))
        return out

    for index, n in enumerate(g.node):
        if n.op_type not in ("QuantizeLinear", "DequantizeLinear") or n.domain not in ("", "ai.onnx", "com.microsoft"):
            if n.domain not in ("", "ai.onnx"):
                raise ValueError(f"Unreviewed custom operation: {n.domain}::{n.op_type}")
            nodes.append(copy.deepcopy(n))
            continue
        if len(n.input) != 3 or len(n.output) != 1:
            raise ValueError(f"Require explicit scale and zero point: {n.name}")
        scale, zp = initializers.get(n.input[1]), initializers.get(n.input[2])
        if scale is None or zp is None or scale.external_data or zp.external_data:
            raise ValueError(f"Quantization parameters must be inline constants: {n.name}")
        if scale.data_type != T.FLOAT or zp.data_type not in INTEGER_TYPES | {T.INT32}:
            raise ValueError(f"Unreviewed quantized dtype: {n.name}")
        if n.op_type == "QuantizeLinear" and zp.data_type == T.INT32:
            raise ValueError("INT32 quantization is not supported by this bridge")
        attrs = {a.name: H.get_attribute_value(a) for a in n.attribute}
        if set(attrs) - {"axis"}:
            raise ValueError(f"Unreviewed QDQ attributes: {n.name}: {attrs}")
        sv, zv = NH.to_array(scale), NH.to_array(zp)
        if sv.shape != zv.shape or sv.ndim > 1 or not sv.size or not np.isfinite(sv).all() or np.any(sv <= 0):
            raise ValueError(f"Invalid QDQ parameters: {n.name}")
        stem = str(index)
        sc = scale.name
        # Integer zero points up to UINT16 are represented exactly in FP32.
        if zp.data_type == T.INT32 and np.any(zv != 0):
            raise ValueError("Only zero INT32 bias zero points are covered")
        # Keep the actual zero-point initializer connected, rather than adding
        # tens of thousands of redundant FP32 constants and orphaned tensors.
        zf = node("Cast", [zp.name], stem + "_zp_float", to=T.FLOAT)
        if sv.size > 1:
            shape = shapes.get(n.input[0])
            if shape is None:
                raise ValueError(f"Missing rank for per-axis QDQ: {n.name}")
            axis = int(attrs.get("axis", 1))
            if not -len(shape) <= axis < len(shape):
                raise ValueError(f"Invalid quantization axis: {n.name}")
            axis %= len(shape)
            if shape[axis] is not None and shape[axis] != sv.size:
                raise ValueError(f"Per-axis scale length mismatch: {n.name}")
            broadcast = np.ones(len(shape), dtype=np.int64)
            broadcast[axis] = sv.size
            bs = constant(stem + "_broadcast", broadcast)
            sc = node("Reshape", [sc, bs], stem + "_scale_shape")
            zf = node("Reshape", [zf, bs], stem + "_zp_shape")
        if n.op_type == "QuantizeLinear":
            value = node("Div", [n.input[0], sc], stem + "_divide")
            value = node("Round", [value], stem + "_ties_to_even")
            value = node("Add", [value, zf], stem + "_offset")
            info = np.iinfo(NH.to_array(zp).dtype)
            lo = constant(stem + "_min", np.float32(info.min))
            hi = constant(stem + "_max", np.float32(info.max))
            value = node("Clip", [value, lo, hi], stem + "_saturate")
            node("Cast", [value], stem + "_integer_storage", output=n.output[0], to=zp.data_type)
        else:
            value = node("Cast", [n.input[0]], stem + "_decode_cast", to=T.FLOAT)
            value = node("Sub", [value, zf], stem + "_decode_offset")
            node("Mul", [value, sc], stem + "_decode_scale", output=n.output[0])
        key = n.op_type + ":" + T.DataType.Name(zp.data_type)
        counts[key] = counts.get(key, 0) + 1
    del g.node[:]
    g.node.extend(nodes)
    g.initializer.extend(added)
    # Check original tensors in order, byte for byte, including external offsets.
    check = copy.deepcopy(model)
    del check.graph.initializer[len(original.graph.initializer):]
    if initializer_digest(check) != old_bytes:
        raise RuntimeError("An original initializer changed")
    if [v.SerializeToString() for v in g.input] != [v.SerializeToString() for v in original.graph.input] or [v.SerializeToString() for v in g.output] != [v.SerializeToString() for v in original.graph.output]:
        raise RuntimeError("Graph I/O changed")
    return model, {"qdq_replacements": counts, "original_nodes": len(original.graph.node),
                   "bridge_nodes": len(nodes), "original_initializer_digest": old_bytes,
                   "original_initializers_unchanged": True, "original_io_unchanged": True}


def session_options(profile):
    options = ort.SessionOptions()
    # Do not fold all quantized weights into persistent FP32 initializers or
    # rewrite explicit Q/DQ arithmetic during this numerical experiment.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_profiling = True
    options.profile_file_prefix = str(profile)
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return options


def gpu_session(model, profile, library):
    if ort.__version__ != "1.27.0":
        raise RuntimeError("Use the already-built ONNX Runtime 1.27.0 environment")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is unavailable; CPU fallback is forbidden")
    options = session_options(profile)
    options.register_custom_ops_library(str(library))
    sess = ort.InferenceSession(model, sess_options=options,
                               providers=[("CUDAExecutionProvider", {"use_tf32": "0", "cudnn_conv_use_max_workspace": "0"})],
                               enable_fallback=False)
    sess.disable_fallback()
    return sess


def audit_profile(path, expected):
    counts = {}
    for event in json.loads(Path(path).read_text()):
        provider = event.get("args", {}).get("provider")
        if provider:
            counts[provider] = counts.get(provider, 0) + 1
    if counts.get(expected, 0) == 0 or set(counts) - {expected}:
        raise RuntimeError(f"Provider audit failed: {counts}")
    return counts


def fixture(dtype=T.UINT16, scale=.125):
    npdtype = {T.UINT16: np.uint16, T.INT16: np.int16, T.UINT8: np.uint8, T.INT8: np.int8}[dtype]
    zp = np.array(32768 if dtype == T.UINT16 else 0, dtype=npdtype)
    tensors = [NH.from_array(np.array(scale, np.float32), "scale"), NH.from_array(zp, "zp")]
    graph = H.make_graph([H.make_node("QuantizeLinear", ["X", "scale", "zp"], ["Q"], domain="com.microsoft"),
                          H.make_node("DequantizeLinear", ["Q", "scale", "zp"], ["Y"], domain="com.microsoft")],
                         "quantization_semantics_gate", [H.make_tensor_value_info("X", T.FLOAT, ["N"])],
                         [H.make_tensor_value_info("Q", dtype, ["N"]), H.make_tensor_value_info("Y", T.FLOAT, ["N"])], tensors)
    model = H.make_model(graph, opset_imports=[H.make_opsetid("", 17), H.make_opsetid("com.microsoft", 1)])
    model.ir_version = 10
    return model


def weight_fixture(axis):
    rng = np.random.default_rng(13)
    shape = (5, 7) if axis == 0 else (7, 5)
    weight = rng.integers(0, 256, shape, dtype=np.uint8)
    scales = np.array([.0031, .07, .014, .021, .23], np.float32)
    zeros = np.array([120, 17, 128, 250, 0], np.uint8)
    g = H.make_graph([H.make_node("DequantizeLinear", ["W", "S", "Z"], ["Y"], domain="com.microsoft", axis=axis)],
                     "per_channel_weights", [], [H.make_tensor_value_info("Y", T.FLOAT, shape)],
                     [NH.from_array(weight, "W"), NH.from_array(scales, "S"), NH.from_array(zeros, "Z")])
    model = H.make_model(g, opset_imports=[H.make_opsetid("", 17), H.make_opsetid("com.microsoft", 1)])
    model.ir_version = 10
    broadcast = [1, 1]; broadcast[axis] = 5
    expected = (weight.astype(np.float32) - zeros.astype(np.float32).reshape(broadcast)) * scales.reshape(broadcast)
    return model, expected


def check_gpu(output, ort_include, library=None):
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "scope": "quantization primitives only; full model pending",
              "native_w8a16_gpu_math": False, "results": []}
    try:
        if library is None:
            library = build_cuda(output / "libqwen35_qdq_cuda.so", ort_include)
        report["cuda_library"] = str(library.resolve())
        report["cuda_library_sha256"] = sha(library)
        report["cuda_source_sha256"] = sha(Path(__file__).with_name("qwen35_w8a16_cuda.cu"))
        cases = [(T.UINT16, scale) for scale in (.125, 5.893147772439988e-06, .0031, .07)]
        cases += [(T.UINT8, .125)]  # Only dtypes present in the frozen graph.
        for index, (dtype, scale) in enumerate(cases):
            original = fixture(dtype, scale)
            model, _ = compact_qdq(original)
            dt = NH.to_array(original.graph.initializer[1]).dtype
            limits = np.iinfo(dt)
            zp = NH.to_array(original.graph.initializer[1]).item()
            codes = np.arange(limits.min, limits.max + 1, dtype=np.float32)
            # Every code, halfway ties of both signs, underflow, overflow.
            x = np.concatenate([(codes - zp) * np.float32(scale), (np.arange(-100, 100, dtype=np.float32) + .5) * np.float32(scale),
                                np.array([-1e20, 1e20], np.float32)])
            expected_q = np.clip(np.rint(x / np.float32(scale)) + zp, limits.min, limits.max).astype(dt)
            expected_y = (expected_q.astype(np.float32) - np.float32(zp)) * np.float32(scale)
            sess = gpu_session(model.SerializeToString(), output / (str(index) + T.DataType.Name(dtype)), library)
            q, y = sess.run(None, {"X": x})
            profile = sess.end_profiling()
            providers = audit_profile(profile, "CUDAExecutionProvider")
            np.testing.assert_array_equal(q, expected_q)
            np.testing.assert_array_equal(y, expected_y)
            report["results"].append({"dtype": T.DataType.Name(dtype), "scale": scale, "values_checked": int(x.size),
                                       "exact_formula_match": True, "providers": providers})
            del sess
        for axis in (0, 1, -1):
            original, expected = weight_fixture(axis)
            model, _ = compact_qdq(original)
            sess = gpu_session(model.SerializeToString(), output / ("weight_axis_" + str(axis)), library)
            actual = sess.run(None, {})[0]
            providers = audit_profile(sess.end_profiling(), "CUDAExecutionProvider")
            np.testing.assert_array_equal(actual, expected)
            report["results"].append({"dtype": "UINT8_weights", "axis": axis,
                                      "exact_formula_match": True, "providers": providers})
            del sess
        report["status"] = "gpu_quantization_primitives_passed_full_model_pending"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(output / "check.json", report)
        print(json.dumps(report, indent=2))


def prepare(assets, output):
    verify(assets, [GRAPH, WEIGHTS])
    if output.exists():
        raise FileExistsError(f"Refuse to replace candidate: {output}")
    original = onnx.load(assets / GRAPH, load_external_data=False)
    model, details = compact_qdq(original)
    output.mkdir(parents=True)
    external = output / "language_weights.data"
    try:
        os.link(assets / WEIGHTS, external)
    except OSError:
        shutil.copyfile(assets / WEIGHTS, external)
    path = output / "decode_gpu.onnx"
    onnx.save_model(model, path)
    onnx.checker.check_model(str(path), full_check=False)
    # Run-time verification hashes both files again; originals are never edited.
    report = {"status": "prepared_not_hardware_or_accuracy_validated", "checkpoint_revision": REVISION,
              "asset_revision": ASSET_REVISION, "source_graph_sha256": PINS[GRAPH],
              "source_weights_sha256": PINS[WEIGHTS], "bridge_graph_sha256": sha(path),
              "weights_sha256": sha(external), "execution_math": "FP32 GEMM/Conv with custom CUDA integer Q/DQ emulation",
              "custom_op_domain": CUSTOM_DOMAIN,
              "native_w8a16_gpu_math": False, **details}
    if report["weights_sha256"] != PINS[WEIGHTS]:
        raise RuntimeError("External weights changed while preparing")
    write_json(output / "bridge.json", report)
    print(json.dumps(report, indent=2))


def table(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def requantize(values, row):
    # Match the published native host's distinct FP32 multiply/divide steps.
    centered = values.astype(np.int32) - int(row["output_zero_point"])
    decoded = centered.astype(np.float32) * np.float32(row["output_scale"])
    scaled = decoded / np.float32(row["input_scale"])
    result = np.rint(scaled).astype(np.int64) + int(row["input_zero_point"])
    return np.clip(result, 0, 65535).astype(np.uint16)


def initial_state(contract, plan):
    feeds = {r["name"]: np.zeros(tuple(map(int, r["shape"].split(","))), dtype=np.dtype(r["dtype"]))
             for r in contract if r["kind"] == "input"}
    seen = set()
    for row in plan:
        dst = feeds[row["destination"]]
        if row["operation"] in ("copy_uint16", "requantize_uint16"):
            if row["destination"] in seen:
                raise ValueError("Duplicate quantized state destination")
            seen.add(row["destination"])
            dst.fill(int(row["input_zero_point"]))
    return feeds


def advance_state(feeds, results, plan):
    # Outputs are separate allocations. Never alias a state input and output.
    for row in plan:
        src = results[row["source"]].reshape(-1)
        dst = feeds[row["destination"]].reshape(-1)
        count, width = int(row["rows"]), int(row["row_elements"])
        so, do = int(row["source_offset_elements"]), int(row["destination_offset_elements"])
        ss, ds = int(row["source_stride_elements"]), int(row["destination_stride_elements"])
        op = row["operation"]
        if op not in ("copy_float32", "copy_uint16", "requantize_uint16"):
            raise ValueError(f"Unsupported state transfer: {op}")
        if so + (count - 1) * ss + width > src.size or do + (count - 1) * ds + width > dst.size:
            raise ValueError("State transfer exceeds tensor bounds")
        for i in range(count):
            values = src[so+i*ss:so+i*ss+width]
            dst[do+i*ds:do+i*ds+width] = requantize(values, row) if op == "requantize_uint16" else values


def run(args):
    assets, out = args.assets.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "provider": args.provider, "checkpoint_revision": REVISION,
              "asset_revision": ASSET_REVISION, "source_graph_sha256": PINS[GRAPH],
              "source_weights_sha256": PINS[WEIGHTS], "runtime": ort.__version__,
              "native_w8a16_gpu_math": False if args.provider == "cuda" else None,
              "accuracy_approved": False, "benchmark_approved": False,
              "timing_scope": "diagnostic Python host loop; CPU embeddings, state transfers and sampling; first step profiled"}
    sess = None
    try:
        if ort.__version__ != "1.27.0":
            raise RuntimeError("Use ONNX Runtime 1.27.0")
        verify(assets, HOST_FILES)
        if args.provider == "cuda":
            if args.bridge is None or args.library is None or args.gpu_check is None:
                raise ValueError("CUDA requires --bridge, --library and a passed --gpu-check directory")
            gate = json.loads((args.gpu_check / "check.json").read_text())
            if gate["status"] != "gpu_quantization_primitives_passed_full_model_pending" or sha(args.library) != gate["cuda_library_sha256"]:
                raise ValueError("CUDA extension does not match a passed quantization gate")
            bridge = json.loads((args.bridge / "bridge.json").read_text())
            if bridge["source_graph_sha256"] != PINS[GRAPH] or bridge["source_weights_sha256"] != PINS[WEIGHTS]:
                raise ValueError("Bridge does not match the frozen model")
            model = args.bridge / "decode_gpu.onnx"
            if sha(model) != bridge["bridge_graph_sha256"] or sha(args.bridge / "language_weights.data") != PINS[WEIGHTS]:
                raise ValueError("Bridge graph or weights changed")
            sess = gpu_session(str(model), out / "cuda_profile", args.library)
            expected_provider = "CUDAExecutionProvider"
            report["bridge_graph_sha256"] = bridge["bridge_graph_sha256"]
            report["cuda_library_sha256"] = gate["cuda_library_sha256"]
        else:
            verify(assets, ["models/2k/decode_ctx.onnx", "models/2k/decode_ctx_qnn.bin"])
            import onnxruntime_qnn as qnn
            ort.register_execution_provider_library("QNNExecutionProvider", qnn.get_library_path())
            devices = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
            if not devices:
                raise RuntimeError("QNN device is unavailable")
            opt = session_options(out / "qnn_profile")
            opt.add_provider_for_devices(devices, {"backend_path": qnn.get_qnn_htp_path(), "offload_graph_io_quantization": "0"})
            sess = ort.InferenceSession(str(assets / "models/2k/decode_ctx.onnx"), sess_options=opt, enable_fallback=False)
            sess.disable_fallback()
            expected_provider = "QNNExecutionProvider"
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(assets / "common/tokenizer.json"))
        template = "<|im_start|>user\n" + args.prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ids = tokenizer.encode(template, add_special_tokens=False).ids
        if not ids or len(ids) + args.max_new_tokens > 2048 or args.max_new_tokens < 1:
            raise ValueError("Prompt and output must fit the 2K context")
        report.update(prompt=args.prompt, input_ids=ids, max_new_tokens=args.max_new_tokens,
                      chat_template="explicit Qwen3.5 text-only, thinking disabled", sampling="greedy")
        emb = np.load(assets / "common/embedding_fp32.npy", mmap_mode="r")
        rope = np.memmap(assets / "contracts/2k/rope_ctx2048_cos_sin_fp32.bin", dtype="<f4", mode="r", shape=(2, 2048, 64))
        contract = table(assets / "contracts/2k/tensor_contract.tsv")
        plan = table(assets / "contracts/2k/advance_plan.tsv")
        for kind, actual in [("input", sess.get_inputs()), ("output", sess.get_outputs())]:
            rows = [r for r in contract if r["kind"] == kind]
            if len(actual) != len(rows):
                raise ValueError(f"{kind} count mismatch")
            for v, row in zip(actual, rows):
                if v.name != row["name"] or v.shape != list(map(int, row["shape"].split(","))) or v.type != "tensor(" + row["dtype"].replace("float32", "float") + ")":
                    raise ValueError(f"{kind} tensor contract mismatch: {row['name']}")
        feeds = initial_state(contract, plan)
        output_names = [v.name for v in sess.get_outputs()]
        logits_at = output_names.index("logits_fp32")
        profile_finished = False
        position = 0

        def step(token):
            nonlocal profile_finished, position
            feeds["hidden_in"][...] = emb[token].reshape(feeds["hidden_in"].shape)
            feeds["rope_cos"][...] = rope[0, position].reshape(feeds["rope_cos"].shape)
            feeds["rope_sin"][...] = rope[1, position].reshape(feeds["rope_sin"].shape)
            feeds["tail_mask"].fill(-64.)
            feeds["tail_mask"].reshape(-1)[2048-position-1:] = 0.
            values = sess.run(None, feeds)
            if not profile_finished:
                profile_finished = True
                path = sess.end_profiling()
                report["profile"] = path
                report["observed_provider_counts"] = audit_profile(path, expected_provider)
            logits = values[logits_at].reshape(-1)
            if not np.isfinite(logits).all():
                raise RuntimeError("Non-finite output logits")
            advance_state(feeds, dict(zip(output_names, values)), plan)
            position += 1
            return logits

        started = time.perf_counter()
        for token in ids:
            logits = step(token)
        np.save(out / "first_logits.npy", logits)
        report["diagnostic_ttft_seconds"] = time.perf_counter() - started
        generated = []
        for i in range(args.max_new_tokens):
            token = int(np.argmax(logits))
            generated.append(token)
            if token in (248046, 248044):
                break
            if i + 1 < args.max_new_tokens:
                logits = step(token)
        report.update(status="generation_completed_accuracy_pending", output_ids=generated,
                      output_text=tokenizer.decode(generated, skip_special_tokens=True),
                      diagnostic_elapsed_seconds=time.perf_counter() - started,
                      first_logits_sha256=sha(out / "first_logits.npy"))
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(out / "run.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))


def compare(reference, candidate, output):
    a = json.loads((reference / "run.json").read_text())
    b = json.loads((candidate / "run.json").read_text())
    if {a.get("provider"), b.get("provider")} != {"qnn", "cuda"}:
        raise ValueError("Compare one QNN run and one CUDA run")
    for key in ("checkpoint_revision", "asset_revision", "source_graph_sha256",
                "source_weights_sha256", "input_ids", "max_new_tokens", "sampling", "runtime"):
        if key not in a or a[key] != b.get(key):
            raise ValueError(f"Comparison identity differs: {key}")
    if a["status"] != "generation_completed_accuracy_pending" or b["status"] != a["status"]:
        raise ValueError("Both full generation runs must complete")
    for root, report in [(reference, a), (candidate, b)]:
        if sha(root / "first_logits.npy") != report["first_logits_sha256"]:
            raise ValueError("Saved logits changed")
    x = np.load(reference / "first_logits.npy").astype(np.float64)
    y = np.load(candidate / "first_logits.npy").astype(np.float64)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Logit shape/numerical validity differs")
    result = {"status": "comparison_recorded_not_accuracy_or_benchmark_approval",
              "reference_provider": a["provider"], "candidate_provider": b["provider"],
              "max_absolute_first_logits_error": float(np.max(np.abs(x-y))),
              "relative_l2_first_logits_error": float(np.linalg.norm(x-y) / max(np.linalg.norm(x), 1e-30)),
              "first_token_matches": a["output_ids"][0] == b["output_ids"][0],
              "entire_output_ids_match": a["output_ids"] == b["output_ids"],
              "reference_text": a["output_text"], "candidate_text": b["output_text"],
              "note": "The GPU emulates the quantization grid with FP32 arithmetic; timings are not native W8A16 arithmetic comparisons."}
    write_json(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    c = commands.add_parser("check", help="CUDA quantization semantics gate; does not load the LLM")
    c.add_argument("--output", type=Path, default=Path.home() / "qwen35-w8a16-gpu-check")
    c.add_argument("--ort-include", type=Path, default=Path.home() / "ort127-benchmark/onnxruntime/include/onnxruntime/core/session")
    c.add_argument("--library", type=Path, help="Existing CUDA extension; otherwise build the adjacent source")
    c = commands.add_parser("prepare", help="Create a separate GPU bridge from the frozen full decode graph")
    c.add_argument("--assets", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c = commands.add_parser("run", help="Run the full 24-layer text model; sequential prefill, no UI change")
    c.add_argument("--assets", type=Path, required=True)
    c.add_argument("--bridge", type=Path)
    c.add_argument("--library", type=Path)
    c.add_argument("--gpu-check", type=Path)
    c.add_argument("--provider", choices=["cuda", "qnn"], required=True)
    c.add_argument("--prompt", default="What is the capital of Malaysia? Answer in one short sentence.")
    c.add_argument("--max-new-tokens", type=int, default=8)
    c.add_argument("--output", type=Path, required=True)
    c = commands.add_parser("compare", help="Compare paired full-model reports without granting accuracy approval")
    c.add_argument("--reference", type=Path, required=True)
    c.add_argument("--candidate", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "check":
        check_gpu(args.output.resolve(), args.ort_include.resolve(), args.library)
    elif args.command == "prepare":
        prepare(args.assets.resolve(), args.output.resolve())
    elif args.command == "compare":
        compare(args.reference, args.candidate, args.output)
    else:
        run(args)


if __name__ == "__main__":
    main()
