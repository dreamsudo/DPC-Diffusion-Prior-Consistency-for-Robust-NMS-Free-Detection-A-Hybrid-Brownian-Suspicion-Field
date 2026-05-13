#!/usr/bin/env python3
"""
patch_run_full_evaluation.py — applies the dual-combiner + label-overlay
edits to tools/run_full_evaluation.py in place.

Idempotent: running it twice is safe (it checks before patching).

Usage:
  python patch_run_full_evaluation.py
"""
from pathlib import Path
import sys

TARGET = Path(__file__).resolve().parent / "tools" / "run_full_evaluation.py"

if not TARGET.exists():
    print(f"ERROR: {TARGET} not found.")
    sys.exit(1)

src = TARGET.read_text()
orig_len = len(src)

changes = []

# ── Patch 1: add --dual-combiner argparse flag ───────────────────────────────
old_arg = '    p.add_argument("--alpha", type=float, default=50.0)'
new_arg = '''    p.add_argument("--alpha", type=float, default=50.0)
    p.add_argument("--dual-combiner", type=str, default="max",
                   choices=["max", "min", "mean", "product"],
                   help="How to fuse Phase 1 and Phase 2 deployed fields for DUAL signal")'''
if "--dual-combiner" in src:
    changes.append("[skip] --dual-combiner already present")
elif old_arg in src:
    src = src.replace(old_arg, new_arg, 1)
    changes.append("[ok]   added --dual-combiner argparse flag")
else:
    print("ERROR: could not find argparse line to patch.")
    print("Looking for:", old_arg)
    sys.exit(2)

# ── Patch 2: replace torch.maximum with combiner branch ──────────────────────
old_fuse = """            # Phase 3 — DUAL SIGNAL
            t0 = time.perf_counter()
            deployed_p1 = out_p1.suspicion_field   # [1,1,H,W]
            deployed_p2 = out_p2.suspicion_field
            deployed_dual = torch.maximum(deployed_p1, deployed_p2)"""
new_fuse = """            # Phase 3 — DUAL SIGNAL (combiner chosen by --dual-combiner)
            t0 = time.perf_counter()
            deployed_p1 = out_p1.suspicion_field   # [1,1,H,W]
            deployed_p2 = out_p2.suspicion_field
            if args.dual_combiner == "max":
                deployed_dual = torch.maximum(deployed_p1, deployed_p2)
            elif args.dual_combiner == "min":
                deployed_dual = torch.minimum(deployed_p1, deployed_p2)
            elif args.dual_combiner == "mean":
                deployed_dual = 0.5 * (deployed_p1 + deployed_p2)
            elif args.dual_combiner == "product":
                deployed_dual = deployed_p1 * deployed_p2
            else:
                raise ValueError(f"unknown combiner: {args.dual_combiner}")"""
if "args.dual_combiner ==" in src:
    changes.append("[skip] combiner branch already present")
elif old_fuse in src:
    src = src.replace(old_fuse, new_fuse, 1)
    changes.append("[ok]   replaced torch.maximum with combiner branch")
else:
    print("ERROR: could not find torch.maximum block to patch.")
    sys.exit(3)

# ── Patch 3: log combiner at startup ─────────────────────────────────────────
old_log = '    info(f"alpha (cls=obj): {args.alpha}")'
new_log = '''    info(f"alpha (cls=obj): {args.alpha}")
    info(f"dual combiner: {args.dual_combiner}")'''
if 'info(f"dual combiner:' in src:
    changes.append("[skip] dual combiner log already present")
elif old_log in src:
    src = src.replace(old_log, new_log, 1)
    changes.append("[ok]   added dual combiner startup log")
else:
    changes.append("[warn] could not find startup log line")

# ── Patch 4: record combiner in aggregate.json ───────────────────────────────
old_agg = '''    aggregate = {
        "n_images": len(per_image_records),
        "alpha": args.alpha,
        "score_threshold": args.score_threshold,'''
new_agg = '''    aggregate = {
        "n_images": len(per_image_records),
        "alpha": args.alpha,
        "dual_combiner": args.dual_combiner,
        "score_threshold": args.score_threshold,'''
if '"dual_combiner": args.dual_combiner,' in src:
    changes.append("[skip] aggregate dual_combiner already present")
elif old_agg in src:
    src = src.replace(old_agg, new_agg, 1)
    changes.append("[ok]   added dual_combiner to aggregate.json")
else:
    changes.append("[warn] could not patch aggregate dict")

# ── Patch 5: record combiner in manifest.json ────────────────────────────────
old_man = '''    write_manifest_local(out_dir, run_config={
        "alpha": args.alpha,
        "score_threshold": args.score_threshold,'''
new_man = '''    write_manifest_local(out_dir, run_config={
        "alpha": args.alpha,
        "dual_combiner": args.dual_combiner,
        "score_threshold": args.score_threshold,'''
if old_man not in src and '"dual_combiner": args.dual_combiner,\n        "score_threshold"' in src:
    changes.append("[skip] manifest dual_combiner already present")
elif old_man in src:
    src = src.replace(old_man, new_man, 1)
    changes.append("[ok]   added dual_combiner to manifest.json")
else:
    changes.append("[warn] could not patch manifest dict")

# ── Patch 6: add _draw_dets_on_axis helper above render_panel ────────────────
old_renderdef = "def render_panel(img_np, patch_boxes, fields, record, name, out_path):"
new_helper_plus_def = '''def _draw_dets_on_axis(ax, dets, color, *, on_dark_bg=False, label_size=6):
    """Draw detection bboxes + class+conf labels on an axis.

    on_dark_bg=True styles labels for visibility on hot-colormap heatmaps.
    """
    for det in dets:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.4, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        txt = f"{det['class_name']} {det['conf']:.2f}"
        if on_dark_bg:
            ax.text(x1, y1 - 2, txt, fontsize=label_size, color="white",
                    bbox=dict(facecolor="black", edgecolor=color,
                              alpha=0.7, pad=0.5, linewidth=0.5))
        else:
            ax.text(x1, y1 - 2, txt, fontsize=label_size, color=color,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.7, pad=0.5))


def render_panel(img_np, patch_boxes, fields, record, name, out_path):'''
if "_draw_dets_on_axis" in src:
    changes.append("[skip] _draw_dets_on_axis helper already present")
elif old_renderdef in src:
    src = src.replace(old_renderdef, new_helper_plus_def, 1)
    changes.append("[ok]   added _draw_dets_on_axis helper")
else:
    print("ERROR: could not find render_panel definition.")
    sys.exit(4)

# ── Patch 7: Panel 1 — add baseline labels ───────────────────────────────────
old_p1 = '''    # Panel 1: original with patch bbox
    axes[0, 0].imshow(img_np)
    for box in patch_boxes:
        x1, y1, x2, y2 = box
        rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2, edgecolor="lime",
                                  facecolor="none")
        axes[0, 0].add_patch(rect)
    axes[0, 0].set_title(name, fontsize=11)
    axes[0, 0].axis("off")'''
new_p1 = '''    # Panel 1: original with patch bbox AND Phase 0 baseline detections
    axes[0, 0].imshow(img_np)
    for box in patch_boxes:
        x1, y1, x2, y2 = box
        rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2, edgecolor="lime",
                                  facecolor="none")
        axes[0, 0].add_patch(rect)
    _draw_dets_on_axis(axes[0, 0], record["phase0"]["detections"],
                       color="#00BFFF", on_dark_bg=False, label_size=7)
    n_p0 = record["phase0"]["n_dets"]
    axes[0, 0].set_title(f"{name}  |  Phase 0 baseline ({n_p0} dets)",
                          fontsize=11)
    axes[0, 0].axis("off")'''
if 'Phase 0 baseline ({n_p0} dets)' in src:
    changes.append("[skip] Panel 1 already patched")
elif old_p1 in src:
    src = src.replace(old_p1, new_p1, 1)
    changes.append("[ok]   patched Panel 1 (baseline labels)")
else:
    changes.append("[warn] Panel 1 patch did not match — may be already patched or different version")

# ── Patch 8: Panel 5 — overlay P1 detections ─────────────────────────────────
old_p5 = '''    im5 = axes[1, 0].imshow(fields["deployed_p1"], cmap="hot", vmin=0, vmax=1)
    axes[1, 0].set_title(title5, fontsize=10)
    axes[1, 0].axis("off")
    plt.colorbar(im5, ax=axes[1, 0], fraction=0.046, pad=0.04)'''
new_p5 = '''    im5 = axes[1, 0].imshow(fields["deployed_p1"], cmap="hot", vmin=0, vmax=1)
    _draw_dets_on_axis(axes[1, 0], record["phase1"]["detections"],
                       color="#D62728", on_dark_bg=True, label_size=6)
    axes[1, 0].set_title(title5, fontsize=10)
    axes[1, 0].axis("off")
    plt.colorbar(im5, ax=axes[1, 0], fraction=0.046, pad=0.04)'''
if 'record["phase1"]["detections"]' in src and 'color="#D62728"' in src:
    changes.append("[skip] Panel 5 already patched")
elif old_p5 in src:
    src = src.replace(old_p5, new_p5, 1)
    changes.append("[ok]   patched Panel 5 (P1 dets on heatmap)")
else:
    changes.append("[warn] Panel 5 patch did not match")

# ── Patch 9: Panel 6 — overlay P2 detections ─────────────────────────────────
old_p6 = '''    im6 = axes[1, 1].imshow(fields["deployed_p2"], cmap="hot", vmin=0, vmax=1)
    axes[1, 1].set_title(title6, fontsize=10)
    axes[1, 1].axis("off")
    plt.colorbar(im6, ax=axes[1, 1], fraction=0.046, pad=0.04)'''
new_p6 = '''    im6 = axes[1, 1].imshow(fields["deployed_p2"], cmap="hot", vmin=0, vmax=1)
    _draw_dets_on_axis(axes[1, 1], record["phase2"]["detections"],
                       color="#2CA02C", on_dark_bg=True, label_size=6)
    axes[1, 1].set_title(title6, fontsize=10)
    axes[1, 1].axis("off")
    plt.colorbar(im6, ax=axes[1, 1], fraction=0.046, pad=0.04)'''
if 'record["phase2"]["detections"]' in src and 'color="#2CA02C"' in src:
    changes.append("[skip] Panel 6 already patched")
elif old_p6 in src:
    src = src.replace(old_p6, new_p6, 1)
    changes.append("[ok]   patched Panel 6 (P2 dets on heatmap)")
else:
    changes.append("[warn] Panel 6 patch did not match")

# ── Patch 10: Panel 7 — overlay DUAL detections ──────────────────────────────
old_p7 = '''    im7 = axes[1, 2].imshow(fields["deployed_dual"], cmap="hot", vmin=0, vmax=1)
    axes[1, 2].set_title(title7, fontsize=10)
    axes[1, 2].axis("off")
    plt.colorbar(im7, ax=axes[1, 2], fraction=0.046, pad=0.04)'''
new_p7 = '''    im7 = axes[1, 2].imshow(fields["deployed_dual"], cmap="hot", vmin=0, vmax=1)
    _draw_dets_on_axis(axes[1, 2], record["phase3"]["detections"],
                       color="#1F77B4", on_dark_bg=True, label_size=6)
    axes[1, 2].set_title(title7, fontsize=10)
    axes[1, 2].axis("off")
    plt.colorbar(im7, ax=axes[1, 2], fraction=0.046, pad=0.04)'''
if 'record["phase3"]["detections"]' in src and 'color="#1F77B4"' in src and 'on_dark_bg=True' in src:
    changes.append("[skip] Panel 7 already patched")
elif old_p7 in src:
    src = src.replace(old_p7, new_p7, 1)
    changes.append("[ok]   patched Panel 7 (DUAL dets on heatmap)")
else:
    changes.append("[warn] Panel 7 patch did not match")

# ── Validate the patched file parses ─────────────────────────────────────────
import ast
try:
    ast.parse(src)
except SyntaxError as e:
    print(f"ERROR: patched file has syntax error: {e}")
    print("NOT writing changes.")
    sys.exit(5)

# ── Write back ───────────────────────────────────────────────────────────────
TARGET.write_text(src)

print(f"\nPatched {TARGET}")
print(f"  size: {orig_len} → {len(src)} bytes (+{len(src) - orig_len})")
print(f"\nChanges:")
for c in changes:
    print(f"  {c}")

print(f"\nVerify with:")
print(f"  grep -c '_draw_dets_on_axis' {TARGET}    # should be >= 4")
print(f"  grep -c 'dual_combiner'      {TARGET}    # should be >= 5")