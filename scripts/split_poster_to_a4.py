#!/usr/bin/env python3
"""
Split a large poster PDF into tiled A4 pages for regular printer output.

Takes the 30"x20" poster and splits it into a grid of A4 pages (landscape),
each showing a portion of the poster with optional overlap for taping.

Output: individual A4 PDFs in reports/poster_pages/
"""

import subprocess
import sys
from pathlib import Path


def main():
    poster_path = Path("reports/poster.pdf")
    out_dir = Path("reports/poster_pages")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not poster_path.exists():
        print(f"Error: {poster_path} not found. Compile poster.tex first.")
        sys.exit(1)

    # Poster dimensions: 30in x 20in (landscape)
    # A4 landscape: 11.69in x 8.27in
    # With margins/overlap, effective print area ~10.5in x 7.5in
    # Grid: 3 columns x 3 rows = 9 pages covers 31.5in x 22.5in (with overlap)

    poster_w_in = 30.0
    poster_h_in = 20.0
    a4_w_in = 11.69
    a4_h_in = 8.27

    # Overlap for taping (0.5 inch on each shared edge)
    overlap_in = 0.5

    # Effective tile size (what each A4 page covers of the poster)
    tile_w = a4_w_in - overlap_in
    tile_h = a4_h_in - overlap_in

    # Number of tiles needed
    import math
    n_cols = math.ceil(poster_w_in / tile_w)
    n_rows = math.ceil(poster_h_in / tile_h)

    print(f"Poster: {poster_w_in}\" x {poster_h_in}\" (landscape)")
    print(f"A4 landscape: {a4_w_in}\" x {a4_h_in}\"")
    print(f"Tile size (with {overlap_in}\" overlap): {tile_w}\" x {tile_h}\"")
    print(f"Grid: {n_cols} cols x {n_rows} rows = {n_cols * n_rows} pages")
    print()

    # Points per inch
    ppi = 72
    poster_w_pt = poster_w_in * ppi
    poster_h_pt = poster_h_in * ppi
    a4_w_pt = a4_w_in * ppi
    a4_h_pt = a4_h_in * ppi
    tile_w_pt = tile_w * ppi
    tile_h_pt = tile_h * ppi

    # Generate one LaTeX file per tile
    pages = []
    for row in range(n_rows):
        for col in range(n_cols):
            page_num = row * n_cols + col + 1

            # Offset into the poster (how much to shift)
            x_offset = col * tile_w_pt
            y_offset = row * tile_h_pt

            # Clamp to not exceed poster
            view_w = min(a4_w_pt, poster_w_pt - x_offset + overlap_in * ppi)
            view_h = min(a4_h_pt, poster_h_pt - y_offset + overlap_in * ppi)

            label = f"row{row+1}_col{col+1}"
            pages.append({
                "page_num": page_num,
                "label": label,
                "row": row + 1,
                "col": col + 1,
                "x_offset": x_offset,
                "y_offset": y_offset,
            })

    # Create a single LaTeX file that produces all tiles
    tex_path = out_dir / "poster_tiled.tex"
    tex_content = r"""\documentclass[landscape]{article}
\usepackage[a4paper, landscape, margin=0pt]{geometry}
\usepackage{graphicx}
\usepackage{eso-pic}
\pagestyle{empty}

\begin{document}
"""

    for p in pages:
        # Use trimming: we include the full poster and trim to show only our tile
        # trim={left} {bottom} {right} {top}
        left_trim = p["x_offset"]
        bottom_trim = poster_h_pt - p["y_offset"] - a4_h_pt
        right_trim = poster_w_pt - p["x_offset"] - a4_w_pt
        top_trim = p["y_offset"]

        # Clamp negative trims to 0 (edge pages may be smaller)
        left_trim = max(0, left_trim)
        bottom_trim = max(0, bottom_trim)
        right_trim = max(0, right_trim)
        top_trim = max(0, top_trim)

        tex_content += f"""% Page {p['page_num']}: Row {p['row']}, Col {p['col']}
\\begin{{center}}
\\includegraphics[trim={{{left_trim}pt {bottom_trim}pt {right_trim}pt {top_trim}pt}}, clip, width=\\paperwidth, height=\\paperheight, keepaspectratio]{{../poster.pdf}}
\\end{{center}}
\\newpage
"""

    tex_content += r"\end{document}" + "\n"

    with open(tex_path, "w") as f:
        f.write(tex_content)

    print(f"Generated {tex_path}")

    # Compile
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "poster_tiled.tex"],
        cwd=str(out_dir),
        capture_output=True, text=True
    )

    if "Output written" in result.stdout:
        print("Compiled poster_tiled.pdf")
    else:
        print("Compilation output:")
        for line in result.stdout.split("\n"):
            if "Error" in line or "Output" in line:
                print(f"  {line}")

    # Also split into individual page PDFs using pdftk or python
    tiled_pdf = out_dir / "poster_tiled.pdf"
    if tiled_pdf.exists():
        try:
            # Try using ghostscript to split pages
            for i, p in enumerate(pages):
                out_file = out_dir / f"page_{p['page_num']:02d}_{p['label']}.pdf"
                gs_result = subprocess.run(
                    ["gs", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pdfwrite",
                     f"-dFirstPage={i+1}", f"-dLastPage={i+1}",
                     f"-sOutputFile={out_file}", str(tiled_pdf)],
                    capture_output=True, text=True
                )
                if gs_result.returncode == 0:
                    print(f"  Extracted {out_file.name}")
                else:
                    print(f"  Failed to extract page {i+1}")
        except FileNotFoundError:
            print("  ghostscript (gs) not found; individual pages not split.")
            print(f"  Use the combined file: {tiled_pdf}")

    # Print assembly guide
    print(f"\n{'='*50}")
    print("ASSEMBLY GUIDE")
    print(f"{'='*50}")
    print(f"Print all pages from poster_tiled.pdf on A4 LANDSCAPE.")
    print(f"Arrange in a {n_cols}x{n_rows} grid:")
    print()
    for row in range(n_rows):
        row_labels = []
        for col in range(n_cols):
            page_num = row * n_cols + col + 1
            row_labels.append(f"[Page {page_num:2d}]")
        print("  " + "  ".join(row_labels))
    print()
    print(f"Overlap: {overlap_in}\" on shared edges. Tape/glue together.")
    print(f"Final size: ~{poster_w_in}\" x {poster_h_in}\" (landscape)")

    # Write assembly guide to file
    with open(out_dir / "ASSEMBLY_GUIDE.txt", "w") as f:
        f.write("POSTER ASSEMBLY GUIDE\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Poster: {poster_w_in}\" x {poster_h_in}\" landscape\n")
        f.write(f"Printed on: {n_cols * n_rows} A4 landscape pages\n")
        f.write(f"Grid: {n_cols} columns x {n_rows} rows\n")
        f.write(f"Overlap: {overlap_in}\" per shared edge\n\n")
        f.write("Arrangement (looking at poster from front):\n\n")
        for row in range(n_rows):
            row_labels = []
            for col in range(n_cols):
                page_num = row * n_cols + col + 1
                row_labels.append(f"[Page {page_num:2d}]")
            f.write("  " + "  ".join(row_labels) + "\n")
        f.write("\nInstructions:\n")
        f.write("1. Print poster_tiled.pdf on A4 paper, LANDSCAPE orientation.\n")
        f.write("2. Lay out pages in the grid above.\n")
        f.write(f"3. Overlap shared edges by {overlap_in}\" and tape from behind.\n")
        f.write("4. Trim excess white edges if needed.\n")

    print(f"\nAssembly guide saved to {out_dir / 'ASSEMBLY_GUIDE.txt'}")


if __name__ == "__main__":
    main()
