"""Send an IMC image to the deployed API (``src/deploy/modal``) and print the result.

Authentication uses a Modal proxy-auth token from the environment (``MODAL_KEY``,
``MODAL_SECRET``). ``--json`` writes the full response for further use (e.g. a UI).

Usage:
    export MODAL_KEY=wk-... MODAL_SECRET=ws-...
    uv run python scripts/deploy/query_modal.py --url https://<workspace>--pigvae-imc-imcserver-web.modal.run --info
    uv run python scripts/deploy/query_modal.py --url ... --image 7443.tiff --json out.json
    uv run python scripts/deploy/query_modal.py --url ... --image 7443.tiff --embed
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="base URL printed by `modal deploy`")
    parser.add_argument("--image", type=Path, help="TIFF or .npy [43, H, W] of raw ion counts")
    parser.add_argument("--info", action="store_true", help="print /info and exit")
    parser.add_argument("--embed", action="store_true", help="call /embed instead of /predict")
    parser.add_argument("--ig-steps", type=int, default=50, help="0 skips Integrated Gradients")
    parser.add_argument("--embeddings", action="store_true", help="include z_global per crop")
    parser.add_argument("--json", type=Path, help="write the full response here")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds (cold start ~1 min)")
    args = parser.parse_args()

    key, secret = os.environ.get("MODAL_KEY"), os.environ.get("MODAL_SECRET")
    if not key or not secret:
        sys.exit("set MODAL_KEY and MODAL_SECRET (a proxy-auth token from the Modal dashboard)")
    headers = {"Modal-Key": key, "Modal-Secret": secret}
    base = args.url.rstrip("/")

    if args.info or args.image is None:
        r = requests.get(f"{base}/info", headers=headers, timeout=args.timeout)
    else:
        endpoint = "embed" if args.embed else "predict"
        params = (
            {} if args.embed else {"ig_steps": args.ig_steps, "include_embeddings": args.embeddings}
        )
        with open(args.image, "rb") as f:
            r = requests.post(
                f"{base}/{endpoint}",
                headers=headers,
                params=params,
                files={"file": (args.image.name, f)},
                timeout=args.timeout,
            )
    if not r.ok:
        sys.exit(f"HTTP {r.status_code}: {r.text}")
    body = r.json()
    if args.json:
        args.json.write_text(json.dumps(body, indent=1))
        print(f"wrote {args.json}")

    if "prediction" in body:
        probs = ", ".join(f"{k}: {v:.3f}" for k, v in body["probabilities"].items())
        print(f"{body['label']}: {body['prediction']} ({probs}); logit {body['logit']:+.3f}")
        top = sorted(body["crops"], key=lambda c: c["attention"], reverse=True)[:5]
        print(f"{body['image']['n_crops']} crops; top attention:")
        for c in top:
            ig = c["ig_importance"]
            print(
                f"  ({c['row']:4d}, {c['col']:4d}) attention {c['attention']:.3f}"
                + ("" if ig is None else f"  IG {ig:+.3f}")
            )
        print(f"timing (ms): {body['timing_ms']}")
    elif "crops" in body:
        print(f"{body['image']['n_crops']} crops, z_global dim {len(body['crops'][0]['z_global'])}")
    else:
        print(json.dumps(body, indent=1)[:3000])


if __name__ == "__main__":
    main()
