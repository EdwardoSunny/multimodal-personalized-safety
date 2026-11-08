"""Download released PRISM checkpoints into prism_monitor/checkpoints/<config>/.

After downloading, every file of the requested configs in the Hub repository is checked; if
any is missing (the Hub's rate limit can cut a download short), the script lists them and
exits with status 1. Rerunning it fetches only the missing files.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "EdwardoSunny/multimodal-personalized-safety-prism"
DEFAULT_DEST = Path(__file__).resolve().parent.parent / "prism_monitor" / "checkpoints"
SHOW_MISSING = 5


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Config names to download (e.g. default evaclip_default). Default: all.",
    )
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Output directory (default: prism_monitor/checkpoints)")
    parser.add_argument("--revision", default=None, help="Model repo revision (branch, tag, or commit)")
    args = parser.parse_args()

    files = HfApi().list_repo_files(REPO_ID, revision=args.revision)
    released = sorted({f.split("/")[0] for f in files if f.endswith("/model.safetensors")})
    if args.configs:
        unknown = sorted(set(args.configs) - set(released))
        if unknown:
            parser.error(f"not in {REPO_ID}: {', '.join(unknown)}. Released configs: {', '.join(released)}")
    configs = args.configs or released

    snapshot_download(
        repo_id=REPO_ID,
        revision=args.revision,
        local_dir=args.dest,
        allow_patterns=[f"{c}/*" for c in configs],
    )
    expected = [f for f in files if f.split("/")[0] in configs]
    missing = [f for f in expected if not (args.dest / f).is_file()]
    if missing:
        sys.exit(
            f"Error: the download is incomplete: {len(missing)} of {len(expected)} checkpoint file(s) missing "
            f"from {args.dest}, e.g. {', '.join(missing[:SHOW_MISSING])}.\n"
            "This usually means the Hugging Face Hub rate limit cut the download short. Rerun this script "
            "(already downloaded files are kept) until it reports success."
        )
    print(f"Checkpoints in {args.dest}: {', '.join(sorted(configs))} ({len(expected)} files, all present)")


if __name__ == "__main__":
    main()
