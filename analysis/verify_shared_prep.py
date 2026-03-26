from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from shared_prep import verify_shared_prep  # noqa: E402


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="验证共享预处理产物是否完整")
    parser.add_argument("--shared-root", default="outputs/shared", help="共享产物根目录")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    summary = verify_shared_prep(shared_root=args.shared_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
