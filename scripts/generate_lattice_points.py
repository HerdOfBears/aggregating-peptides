#!/usr/bin/env python3
"""
Generate a positions.dat file of lattice points for use with
gmx insert-molecules -ip.

Usage:
    python generate_lattice_points.py <nmol> <box_l_nm> <output_file>

Example:
    python generate_lattice_points.py 50 15 positions.dat
"""

import sys
import math


def main():
    if len(sys.argv) != 4:
        print("Usage: python generate_lattice_points.py <nmol> <box_l_nm> <output_file>")
        sys.exit(1)

    nmol = int(sys.argv[1])
    box_l = float(sys.argv[2])
    output_file = sys.argv[3]

    # smallest n such that n^3 >= nmol
    n = math.ceil(nmol ** (1.0 / 3.0))
    spacing = box_l / n

    print(f"Lattice: {n}^3 grid, spacing = {spacing:.3f} nm")
    print(f"Placing {nmol} points in a {box_l} nm cubic box")

    count = 0
    with open(output_file, "w") as f:
        f.write(f"# {nmol} lattice points in {box_l} nm cubic box\n")
        f.write(f"# grid: {n}x{n}x{n}, spacing: {spacing:.3f} nm\n")
        for ix in range(n):
            for iy in range(n):
                for iz in range(n):
                    if count >= nmol:
                        break
                    x = (ix + 0.5) * spacing
                    y = (iy + 0.5) * spacing
                    z = (iz + 0.5) * spacing
                    f.write(f"{x:.3f}  {y:.3f}  {z:.3f}\n")
                    count += 1
                if count >= nmol:
                    break
            if count >= nmol:
                break

    print(f"Wrote {count} positions to {output_file}")


if __name__ == "__main__":
    main()
