#!/usr/bin/env python3
"""Create a separate portability candidate from the original full QDQ bundle.

Preserves original weights, scales, zero points and output QDQ. This addresses
operator representation only. It does NOT repair the original model's failed
accuracy, prove QNN support, or authorize a benchmark.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def patch_graph(original):
    model = copy.deepcopy(original)
    graph = model.graph
    original_initializers = {t.name: t.SerializeToString() for t in graph.initializer}
    initializers = {t.name: t for t in graph.initializer}
    producers = {out: n for n in graph.node for out in n.output}
    consumers = {}
    for n in graph.node:
        for name in n.input:
            consumers.setdefault(name, []).append(n)
    types = {v.name: v.type.tensor_type.elem_type
             for v in [*graph.input, *graph.value_info, *graph.output]
             if v.type.HasField("tensor_type")}
    types.update({name: t.data_type for name, t in initializers.items()})
    outputs = {v.name for v in graph.output}
    used = set(initializers) | set(producers) | set(types)
    used.update(n.name for n in graph.node)
    aliases, nodes, changes, added = {}, [], [], []

    def resolve(name):
        while name in aliases:
            name = aliases[name]
        return name

    def dtype(name):
        name = resolve(name)
        if name in types:
            return types[name]
        parent = producers.get(name)
        if parent is not None and parent.op_type == "DequantizeLinear":
            scale = initializers.get(parent.input[1])
            if scale is not None:
                return scale.data_type
        return None

    for i, old in enumerate(graph.node):
        n = copy.deepcopy(old)
        n.input[:] = [resolve(name) for name in n.input]
        if n.domain in ("", "ai.onnx") and n.op_type == "Cast":
            target = next((a.i for a in n.attribute if a.name == "to"), None)
            if (target == TensorProto.FLOAT and len(n.input) == len(n.output) == 1
                    and dtype(n.input[0]) == TensorProto.FLOAT
                    and n.output[0] not in outputs):
                aliases[n.output[0]] = n.input[0]
                changes.append({"node": n.name, "change": "remove proven FP32-to-FP32 Cast"})
                continue
        if n.domain in ("", "ai.onnx") and n.op_type == "Neg":
            if len(n.input) != 1 or len(n.output) != 1 or n.attribute:
                raise ValueError(f"Unexpected Neg structure: {n.name}")
            parent = producers.get(n.input[0])
            children = consumers.get(n.output[0], [])
            if (parent is None or parent.domain not in ("", "ai.onnx")
                    or parent.op_type != "DequantizeLinear" or len(parent.input) != 3
                    or not children or n.output[0] in outputs
                    or any(c.op_type != "QuantizeLinear" or c.domain not in ("", "ai.onnx")
                           or c.input[0] != n.output[0] for c in children)):
                raise ValueError(f"Neg lacks the inspected input/output QDQ pattern: {n.name}")
            scale_t = initializers.get(parent.input[1])
            zp_t = initializers.get(parent.input[2])
            if scale_t is None or zp_t is None:
                raise ValueError(f"Non-constant quantization parameters: {n.name}")
            scale, zp = numpy_helper.to_array(scale_t), numpy_helper.to_array(zp_t)
            if (scale.dtype != np.float32 or scale.size != 1
                    or not np.isfinite(scale).all() or not np.all(scale > 0)
                    or zp.dtype != np.int8 or zp.size != 1 or np.any(zp != 0)):
                raise ValueError(f"Expected signed symmetric per-tensor INT8: {n.name}")
            stem = f"portability_zero_{i}"
            names = [stem + suffix for suffix in ("_q", "_float", "_DQ")]
            if any(name in used for name in names):
                raise ValueError("Reserved patch name collision")
            used.update(names)
            # Encoded zero uses the existing input encoding: no new scale or
            # calibration. Keep both Sub operands dequantized for QDQ grouping.
            added.append(numpy_helper.from_array(np.array(0, dtype=np.int8), names[0]))
            nodes.append(helper.make_node("DequantizeLinear",
                         [names[0], parent.input[1], parent.input[2]],
                         [names[1]], name=names[2]))
            n.op_type = "Sub"
            n.input[:] = [names[1], n.input[0]]
            changes.append({"node": n.name, "change": "Neg(x) -> Sub(exact quantized zero, x)"})
        nodes.append(n)
    graph.node.clear()
    graph.node.extend(nodes)
    graph.initializer.extend(added)
    retained_info = [v for v in graph.value_info if v.name not in aliases]
    graph.value_info.clear()
    graph.value_info.extend(retained_info)
    for t in graph.initializer:
        if t.name in original_initializers and t.SerializeToString() != original_initializers[t.name]:
            raise AssertionError(f"Original tensor changed: {t.name}")
    if not set(original_initializers).issubset(t.name for t in graph.initializer):
        raise AssertionError("Original tensor removed")
    onnx.checker.check_model(model)
    return model, changes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = Path.home() / "qwen-shared-w8a8"
    p.add_argument("--source", type=Path, default=root / "full_model_candidate_v2")
    p.add_argument("--output", type=Path, default=root / "full_model_portability_v4")
    args = p.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Source and output must be separate sibling bundles")
    if output.exists():
        raise FileExistsError(f"Will not overwrite {output}; choose a new --output")
    manifest = json.loads((source / "manifest.json").read_text())
    if "scope" in str(manifest.get("quantization", "")).lower() or "scope" in source.name:
        raise ValueError("Use the original complete QDQ candidate_v2, not scope_v3")
    files = {name: digest for name, digest in manifest["files"].items()
             if not name.startswith("fp32/")}
    models = [name for name in files if name.startswith("int8/") and name.endswith(".onnx")]
    if len(models) != 32:
        raise ValueError(f"Expected all 32 INT8 graphs; found {len(models)}")
    print("Checking source model and asset hashes...", flush=True)
    for name, expected in files.items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or sha(path) != expected:
            raise ValueError(f"Unsafe path or source checksum mismatch: {name}")
    output.mkdir(parents=True)
    report = {"status": "incomplete", "source_manifest_sha256": sha(source / "manifest.json"),
              "graphs": {}, "accuracy": "original candidate failed; NOT repaired by this patch",
              "hardware_validation": "NOT RUN", "original_tensors_unchanged": True}
    hashes = {}
    for name in sorted(files):
        src, dst = source / name, output / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if name in models:
            original = onnx.load(str(src), load_external_data=False)
            if any(t.data_location == TensorProto.EXTERNAL for t in original.graph.initializer):
                raise ValueError("External-data model needs explicit handling")
            patched, changes = patch_graph(original)
            if changes:
                onnx.save_model(patched, str(dst))
            else:
                shutil.copy2(src, dst)
            report["graphs"][name] = changes
            print(f"Patched {name}: {len(changes)} edits", flush=True)
        else:
            shutil.copy2(src, dst)
        hashes[name] = sha(dst)
    report["status"] = "portability_candidate_not_hardware_or_accuracy_validated"
    manifest["status"] = report["status"]
    manifest["files"] = hashes
    manifest["portability_patch"] = {"source_manifest_sha256": report["source_manifest_sha256"],
                                    "original_weights_scales_zero_points": "unchanged"}
    manifest.setdefault("limitations", []).append(
        "Portability-only Neg/Sub and identity-Cast rewrite. Original INT8 accuracy failure remains unresolved.")
    (output / "patch_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("All 32 graphs processed. No export or calibration performed.")
    print("Bundle:", output)
    print("This is a compatibility candidate. Accuracy is still FAILED/UNVALIDATED.")


if __name__ == "__main__":
    main()
