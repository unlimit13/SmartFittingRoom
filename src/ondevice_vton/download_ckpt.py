"""Download the Mobile-VTON checkpoint from HuggingFace into ./checkpoint."""
import os
from huggingface_hub import snapshot_download

REPO = "FlashStight/Mobile-VTON"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # The HF repo stores everything under a top-level "checkpoint/" folder.
    # max_workers=1 -> sequential download (the Pi's flaky link stalls on
    # parallel connections). Incomplete files resume automatically.
    path = snapshot_download(
        repo_id=REPO,
        allow_patterns=["checkpoint/*", "checkpoint/**"],
        local_dir=DEST,
        max_workers=1,
        etag_timeout=60,
    )
    print("Downloaded to:", os.path.join(DEST, "checkpoint"))
