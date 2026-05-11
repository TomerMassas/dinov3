"""Standalone HTML viewer for real-world eval clusters.

Reads:
  <OUTPUT_BASE>/test_projects.json              (GLOBAL — shared across ckpts)
  <OUTPUT_BASE>/crops/<pid>/*.jpg               (GLOBAL — shared across ckpts)
  <OUTPUT_BASE>/<ckpt_dir>/clusters/<pid>.json  (per-ckpt — model-dependent)

Writes:
  <OUTPUT_BASE>/<ckpt_dir>/ui/index.html
  <OUTPUT_BASE>/<ckpt_dir>/ui/projects/<pid>.html

Pure stdlib. No torch / no model load — fast, runs in seconds. Self-contained
output: relative paths only, no server needed. Open `index.html` in any
browser.

Usage:
    python3 -m train_pictime.finetune.realworld_eval.build_html_viewer
"""
# run this to streem the VM port
# ssh -L 8080:127.0.0.1:8080 azureuser@10.0.32.13
from __future__ import annotations

import html
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from train_pictime.finetune.realworld_eval.config import (
    FINETUNE_VERSION_DIR, OUTPUT_BASE,
)


# ---------------------------------------------------------------------------
# Output dir resolution
# ---------------------------------------------------------------------------

def find_output_dir(output_base: str, version_dir: str) -> Path:
    """Find <OUTPUT_BASE>/<Vname>_iter<N>_sil<S> matching the chosen ckpt's V<n>."""
    version_name = Path(version_dir).name
    base = Path(output_base)
    if not base.exists():
        raise FileNotFoundError(f"OUTPUT_BASE not found: {base}")
    matches = list(base.glob(f"{version_name}_iter*_sil*"))
    if not matches:
        raise FileNotFoundError(f"No subdir matches '{version_name}_iter*_sil*' under {base}. "
                                f"Run cluster_test_set.py first.")
    matches.sort(key=lambda p: p.stat().st_mtime)   # newest last (by filesystem mtime)
    if len(matches) > 1:
        print(f"Multiple matching output dirs ({len(matches)}); using newest by mtime:")
        for m in matches:
            print(f"  - {m}")
    return matches[-1]


# ---------------------------------------------------------------------------
# Cluster + crop loading
# ---------------------------------------------------------------------------

def safe_stem(filename: str) -> str:
    """Same rule as cluster_test_set.py — must match how crops were named."""
    stem = os.path.splitext(filename)[0]
    return re.sub(r"[^A-Za-z0-9._-]", "_", stem)


def load_project_clusters(clusters_root: Path, pid: str) -> dict[int, list[tuple[str, int]]]:
    """Read <clusters_root>/<pid>.json → {cluster_id: [(filename, bbox_index), ...]}."""
    path = clusters_root / f"{pid}.json"
    with open(path, "r") as f:
        data = json.load(f)
    by_cluster: dict[int, list[tuple[str, int]]] = {}
    for fname, entries in data.items():
        for entry in entries:
            cid = int(entry["cluster_id"])
            by_cluster.setdefault(cid, []).append((fname, int(entry["bbox_index"])))
    return by_cluster


def project_stats(by_cluster: dict[int, list[tuple[str, int]]]) -> dict:
    n_clusters = sum(1 for cid in by_cluster if cid != -1)
    n_noise = len(by_cluster.get(-1, []))
    n_crops = sum(len(v) for v in by_cluster.values())
    return {"n_clusters": n_clusters, "n_noise": n_noise, "n_crops": n_crops}


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 0; padding: 24px; background: #fafafa; color: #222; }
h1, h2 { font-weight: 600; }
h1 { font-size: 22px; margin: 0 0 4px 0; }
h2 { font-size: 16px; margin: 24px 0 8px 0; }
.header { margin-bottom: 24px; padding-bottom: 12px; border-bottom: 1px solid #ddd; }
.meta { font-size: 13px; color: #666; }
.nav  { font-size: 13px; margin: 8px 0; }
.nav a { color: #0366d6; text-decoration: none; margin-right: 12px; }
.nav a:hover { text-decoration: underline; }
.cluster { margin-bottom: 28px; padding: 12px; background: white; border-radius: 6px;
           border: 1px solid #e1e4e8; }
.cluster.noise { background: #fff8e1; border-color: #ffe082; }
.cluster-title { font-size: 14px; font-weight: 600; margin: 0 0 8px 0; }
.cluster-title .count { color: #666; font-weight: 400; margin-left: 8px; }
.grid { display: flex; flex-wrap: wrap; gap: 6px; }
.grid img { height: 140px; width: auto; border-radius: 3px; border: 1px solid #ddd;
            background: #f0f0f0; }
.grid figure { margin: 0; }
.grid figcaption { font-size: 10px; color: #999; max-width: 140px; overflow: hidden;
                   text-overflow: ellipsis; white-space: nowrap; margin-top: 2px; }
.card-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
             gap: 12px; }
.card { background: white; border: 1px solid #e1e4e8; border-radius: 6px;
        overflow: hidden; text-decoration: none; color: inherit; display: block; }
.card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.card .thumb { width: 100%; aspect-ratio: 1 / 1; background: #f0f0f0;
               display: flex; align-items: center; justify-content: center; }
.card .thumb img { max-width: 100%; max-height: 100%; }
.card .info { padding: 8px 10px; font-size: 12px; }
.card .pid { font-weight: 600; font-size: 12px; word-break: break-all;
             margin-bottom: 2px; }
.card .stats { color: #666; }
"""


def render_index(ckpt_meta: dict,
                 pid_to_stats: dict[str, dict],
                 pid_to_thumb: dict[str, str | None],
                ) -> str:
    pids_sorted = sorted(pid_to_stats.keys())
    n_proj = len(pids_sorted)
    total_crops = sum(s["n_crops"] for s in pid_to_stats.values())
    total_clusters = sum(s["n_clusters"] for s in pid_to_stats.values())
    total_noise = sum(s["n_noise"] for s in pid_to_stats.values())
    mean_clusters = total_clusters / n_proj if n_proj else 0.0
    noise_frac = total_noise / total_crops if total_crops else 0.0

    cards = []
    for pid in pids_sorted:
        s = pid_to_stats[pid]
        thumb = pid_to_thumb.get(pid)
        # crops/ is at OUTPUT_BASE (one level above the ckpt dir); ui/index.html
        # is two levels deep inside the ckpt dir → ../../crops/<pid>/<file>.
        thumb_html = (f'<img src="../../crops/{quote(pid)}/{quote(thumb)}" alt="">'
                      if thumb else '<span style="font-size:10px;color:#999">no crop</span>')
        cards.append(
            f'<a class="card" href="projects/{quote(pid)}.html">'
            f'  <div class="thumb">{thumb_html}</div>'
            f'  <div class="info">'
            f'    <div class="pid">{html.escape(pid)}</div>'
            f'    <div class="stats">{s["n_clusters"]} clusters &middot; '
            f'{s["n_crops"]} crops &middot; {s["n_noise"]} noise</div>'
            f'  </div>'
            f'</a>'
        )
    cards_html = "\n".join(cards)

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Real-world eval — {html.escape(ckpt_meta['version_name'])}</title>
<style>{CSS}</style></head><body>
<div class="header">
  <h1>Real-world ReID eval — {html.escape(ckpt_meta['version_name'])}</h1>
  <div class="meta">
    Ckpt: {html.escape(ckpt_meta['ckpt_path'])}<br>
    Iter {ckpt_meta['iteration']} &middot; train silhouette {ckpt_meta['train_sil']:.4f}
  </div>
  <div class="meta" style="margin-top:8px">
    {n_proj} projects &middot; {total_crops} crops &middot;
    {total_clusters} clusters total &middot;
    mean {mean_clusters:.2f} clusters/project &middot;
    noise {noise_frac*100:.1f}%
  </div>
</div>
<div class="card-grid">
{cards_html}
</div>
</body></html>
"""


def render_project_page(pid: str,
                        by_cluster: dict[int, list[tuple[str, int]]],
                        prev_pid: str | None,
                        next_pid: str | None,
                        ckpt_meta: dict,
                       ) -> str:
    stats = project_stats(by_cluster)

    nav_links = ['<a href="../index.html">&larr; index</a>']
    if prev_pid is not None:
        nav_links.append(f'<a href="{quote(prev_pid)}.html">&larr; prev: {html.escape(prev_pid)}</a>')
    if next_pid is not None:
        nav_links.append(f'<a href="{quote(next_pid)}.html">next: {html.escape(next_pid)} &rarr;</a>')
    nav_html = '<div class="nav">' + " ".join(nav_links) + "</div>"

    # Cluster sections: real cluster_ids ascending, noise (-1) last.
    real_cluster_ids = sorted(cid for cid in by_cluster if cid != -1)
    ordered_ids = real_cluster_ids + ([-1] if -1 in by_cluster else [])

    sections = []
    for cid in ordered_ids:
        entries = by_cluster[cid]
        is_noise = (cid == -1)
        title = f"Cluster {cid}" if not is_noise else "Noise (cluster_id = -1)"
        cls = "cluster noise" if is_noise else "cluster"
        imgs_html = []
        for fname, bbox_idx in entries:
            crop_file = f"{safe_stem(fname)}__bb{bbox_idx}.jpg"
            caption = html.escape(f"{fname} #{bbox_idx}")
            # ui/projects/<pid>.html → ../../../crops/<pid>/<file>
            # (up to ui/ then ckpt_dir then OUTPUT_BASE, then into crops/)
            imgs_html.append(
                f'<figure>'
                f'<img src="../../../crops/{quote(pid)}/{quote(crop_file)}" alt="{caption}" title="{caption}">'
                f'<figcaption>{caption}</figcaption>'
                f'</figure>'
            )
        sections.append(
            f'<div class="{cls}">'
            f'  <h2 class="cluster-title">{title}<span class="count">{len(entries)} crops</span></h2>'
            f'  <div class="grid">{"".join(imgs_html)}</div>'
            f'</div>'
        )
    sections_html = "\n".join(sections)

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(pid)} — real-world eval</title>
<style>{CSS}</style></head><body>
<div class="header">
  <h1>{html.escape(pid)}</h1>
  <div class="meta">
    {stats['n_clusters']} clusters &middot; {stats['n_crops']} crops &middot;
    {stats['n_noise']} noise &middot;
    ckpt iter {ckpt_meta['iteration']} (sil {ckpt_meta['train_sil']:.4f})
  </div>
  {nav_html}
</div>
{sections_html}
<div style="margin-top:24px">{nav_html}</div>
</body></html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def pick_thumb(crops_dir: Path,
               by_cluster: dict[int, list[tuple[str, int]]],
              ) -> str | None:
    """Pick a representative crop filename for the project card thumbnail."""
    # Prefer the largest non-noise cluster's first crop; fallback to anything.
    candidates = []
    for cid, entries in by_cluster.items():
        if cid == -1:
            continue
        candidates.append((len(entries), entries[0]))
    if candidates:
        candidates.sort(reverse=True)
        fname, bbox_idx = candidates[0][1]
    elif by_cluster.get(-1):
        fname, bbox_idx = by_cluster[-1][0]
    else:
        return None
    crop_file = f"{safe_stem(fname)}__bb{bbox_idx}.jpg"
    return crop_file if (crops_dir / crop_file).exists() else None


def main():
    output_dir = find_output_dir(OUTPUT_BASE, FINETUNE_VERSION_DIR)
    print(f"Building HTML viewer for: {output_dir}")

    # test_projects.json + crops/ live at OUTPUT_BASE (global, shared across ckpts).
    test_projects_path = Path(OUTPUT_BASE) / "test_projects.json"
    if not test_projects_path.exists():
        raise FileNotFoundError(f"Global test_projects.json not found at {test_projects_path}. "
                                f"Run cluster_test_set.py first.")
    with open(test_projects_path, "r") as f:
        test_data = json.load(f)
    pids: list[str] = sorted(test_data["project_ids"])

    clusters_root = output_dir / "clusters"
    crops_root = Path(OUTPUT_BASE) / "crops"
    ui_dir = output_dir / "ui"
    projects_dir = ui_dir / "projects"
    ui_dir.mkdir(parents=True, exist_ok=True)
    projects_dir.mkdir(parents=True, exist_ok=True)

    # Parse ckpt identity from dir name: V<n>_iter<N>_sil<S>
    m = re.match(r"(V\d+)_iter(\d+)_sil([-\d.]+)", output_dir.name)
    if not m:
        raise ValueError(f"Cannot parse ckpt identity from dir name: {output_dir.name}")
    ckpt_meta = {
        "version_name": m.group(1),
        "iteration": int(m.group(2)),
        "train_sil": float(m.group(3)),
        "ckpt_path": str(Path(FINETUNE_VERSION_DIR) / "ckpt"),
    }

    # Per-project: load clusters, pick thumb, render page
    pid_to_stats: dict[str, dict] = {}
    pid_to_thumb: dict[str, str | None] = {}
    pid_to_by_cluster: dict[str, dict] = {}

    available_pids: list[str] = []
    for pid in pids:
        clusters_path = clusters_root / f"{pid}.json"
        if not clusters_path.exists():
            print(f"  [{pid}] no clusters json — skipping")
            continue
        by_cluster = load_project_clusters(clusters_root, pid)
        pid_to_by_cluster[pid] = by_cluster
        pid_to_stats[pid] = project_stats(by_cluster)
        pid_to_thumb[pid] = pick_thumb(crops_root / pid, by_cluster)
        available_pids.append(pid)

    # Render per-project pages with prev/next nav
    for i, pid in enumerate(available_pids):
        prev_pid = available_pids[i - 1] if i > 0 else None
        next_pid = available_pids[i + 1] if i < len(available_pids) - 1 else None
        page = render_project_page(pid=pid,
                                   by_cluster=pid_to_by_cluster[pid],
                                   prev_pid=prev_pid,
                                   next_pid=next_pid,
                                   ckpt_meta=ckpt_meta,
                                  )
        with open(projects_dir / f"{pid}.html", "w", encoding="utf-8") as f:
            f.write(page)

    # Render index
    index_html = render_index(ckpt_meta=ckpt_meta,
                              pid_to_stats=pid_to_stats,
                              pid_to_thumb=pid_to_thumb,
                             )
    with open(ui_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(index_html)

    print(f"\nWrote {len(available_pids)} project pages + index.html")
    print(f"Open: {ui_dir / 'index.html'}")


if __name__ == "__main__":
    main()
