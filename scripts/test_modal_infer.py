import argparse
import sys

import numpy as np
import requests


def resolve_dims_from_onnx(onnx_path: str) -> tuple[int, int]:
    try:
        import onnx
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("onnx is required to infer dims from --onnx-path") from exc

    model = onnx.load(onnx_path)
    # Expect input named "node_features" with shape [B, N, Din]
    node_input = None
    for i in model.graph.input:
        if i.name == "node_features":
            node_input = i
            break
    if node_input is None:
        raise ValueError("Could not find input named 'node_features' in ONNX model")
    dims = node_input.type.tensor_type.shape.dim
    if len(dims) != 3:
        raise ValueError(f"Unexpected node_features rank: {len(dims)} (expected 3)")

    # dims[0] is batch, dims[1] is N, dims[2] is Din
    def _to_int(d) -> int | None:
        return int(d.dim_value) if d.dim_value not in (None, 0) else None

    n = _to_int(dims[1])
    din = _to_int(dims[2])
    if n is None or din is None:
        raise ValueError(
            "Symbolic or missing dims in ONNX. Provide --num-nodes and --din explicitly."
        )
    return n, din


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Modal /infer endpoint for ONNX model")
    parser.add_argument(
        "--url",
        required=True,
        help="Base function URL or full /infer URL, e.g. https://<func>.modal.run or .../infer",
    )
    parser.add_argument(
        "--onnx-path", default=None, help="Optional path to model.onnx to infer dims"
    )
    parser.add_argument("--batch", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=None,
        help="N dimension if not using --onnx-path",
    )
    parser.add_argument(
        "--din", type=int, default=None, help="Din dimension if not using --onnx-path"
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds")
    args = parser.parse_args()

    infer_url = args.url if args.url.endswith("/infer") else args.url.rstrip("/") + "/infer"

    if args.onnx_path:
        try:
            n, din = resolve_dims_from_onnx(args.onnx_path)
        except Exception as exc:
            print(f"Failed to infer dims from ONNX: {exc}", file=sys.stderr)
            sys.exit(2)
    else:
        if args.num_nodes is None or args.din is None:
            print(
                "Provide --num-nodes and --din when --onnx-path is not given",
                file=sys.stderr,
            )
            sys.exit(2)
        n, din = args.num_nodes, args.din

    b = int(args.batch)
    node_features = np.zeros((b, n, din), dtype=np.float32)
    mask = np.ones((b, n), dtype=bool)

    payload = {
        "node_features": node_features.tolist(),
        "mask": mask.tolist(),
    }

    resp = requests.post(infer_url, json=payload, timeout=args.timeout)
    try:
        resp.raise_for_status()
    except Exception:
        print(f"HTTP {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    logits = np.asarray(data.get("node_logits"))
    emb = np.asarray(data.get("graph_emb"))
    print("OK")
    print("node_logits shape:", tuple(logits.shape))
    print("graph_emb shape:", tuple(emb.shape))


if __name__ == "__main__":
    main()
