#!/usr/bin/env vmd
# -*- mode: tcl -*-
# =====================================================================
# neutral_termini.tcl
#
# Swap the termini of a protein structure for NEUTRAL termini using
# VMD + psfgen (CHARMM36):
#
#     C-terminus  ->  CT2          amidation, -C(=O)NH2,  charge 0
#     N-terminus  ->  NNEU / NNEG  free amine, -NH2,      charge 0   (default)
#               or ->  ACE / ACP   acetyl cap, CH3CO-,    charge 0   (NTERM_STYLE=acetyl)
#
# Glycine and proline at the N-terminus are special-cased, because the
# standard CHARMM36 topology has NO valid neutral-amine patch for them:
#   * GLY : NNEU assumes a CT1 alpha carbon; glycine's is CT2, so plain
#           NNEU fails. We inject a custom NNEG patch (see step 3).
#   * PRO : no neutral proline amine patch exists in standard CHARMM36.
#           In amine mode we STOP; use NTERM_STYLE=acetyl (ACP) instead.
#
# Run (text mode, no GUI):
#   vmd -dispdev text -e neutral_termini.tcl -args input.pdb out
#     input.pdb : starting structure (PDB; hydrogens optional)
#     out       : basename -> out.psf / out.pdb
#
# ASSUMPTIONS: single protein chain. For multi-chain systems, loop the
# segment block over each chain with its own first/last residue check.
# =====================================================================

package require psfgen
psfcontext reset

# ----------------------- user settings -------------------------------
set TOPO        "data/top_all36_prot.rtf" ;# path to your CHARMM36 protein RTF
set NTERM_STYLE "amine"              ;# "amine" (NNEU/NNEG) or "acetyl" (ACE/ACP)
set SEGID       "PROA"               ;# segment id for the protein
# ---------------------------------------------------------------------

# ---- command-line args ----
if {[llength $argv] < 2} {
    puts "usage: vmd -dispdev text -e neutral_termini.tcl -args <input.pdb> <out_basename>"
    exit 1
}
set inpdb   [lindex $argv 0]
set outbase [lindex $argv 1]

# =====================================================================
# 1. Inspect the structure: find the first and last protein residue.
#    A throwaway VMD molecule is used just to read resname/resid.
# =====================================================================
set mol [mol new $inpdb type pdb waitfor all]
set sel [atomselect $mol "protein"]
if {[$sel num] == 0} { puts "ERROR: no protein atoms in $inpdb"; exit 1 }

set resids   [lsort -integer -unique [$sel get resid]]
set firstres [lindex $resids 0]
set lastres  [lindex $resids end]

set fsel [atomselect $mol "protein and resid $firstres"]
set lsel [atomselect $mol "protein and resid $lastres"]
set firstname [lindex [$fsel get resname] 0]
set lastname  [lindex [$lsel get resname] 0]
$fsel delete; $lsel delete; $sel delete

puts "First residue: $firstname $firstres"
puts "Last  residue: $lastname $lastres"

# =====================================================================
# 2. Choose terminal patches.  C-term is always CT2 (amidation).
# =====================================================================
set CTER_PATCH  "CT2"
set need_custom_gly 0

if {$NTERM_STYLE eq "acetyl"} {
    # Acetyl caps exist for every residue, including proline.
    set NTER_PATCH [expr {$firstname eq "PRO" ? "ACP" : "ACE"}]
} else {
    switch -- $firstname {
        PRO {
            puts "============================================================"
            puts "ERROR: first residue is PROLINE."
            puts "  No validated neutral-amine N-terminal patch exists for"
            puts "  proline in standard CHARMM36 (only PROP, which is +1)."
            puts "  Set NTERM_STYLE to \"acetyl\" (uses ACP), or supply your"
            puts "  own QM-parameterized proline patch."
            puts "============================================================"
            exit 1
        }
        GLY {
            # plain NNEU breaks on glycine -> use custom NNEG (step 3)
            set NTER_PATCH "NNEG"
            set need_custom_gly 1
        }
        default { set NTER_PATCH "NNEU" }
    }
}
puts "N-terminal patch: $NTER_PATCH"
puts "C-terminal patch: $CTER_PATCH"

# =====================================================================
# 3. Custom neutral-glycine patch (NNEG), written only if needed.
#    This is NNEU with the two changes the topology file's own comment
#    calls for: retype CA (CT1 -> CT2) and add glycine's second alpha
#    hydrogen. Charges keep NNEU's partitioning (alpha group = +0.28),
#    so the patch stays charge-neutral.
#
#    *** VALIDATE *** against YOUR toppar release before production:
#    copy NNEU's exact atom types from your top_all36_prot.rtf (the
#    H vs HC and HB vs HB2 names are version dependent), change ONLY
#    CA to CT2 + duplicate the alpha H, and confirm psfgen reports no
#    missing parameters.
# =====================================================================
if {$need_custom_gly} {
    set patchfile "nneg_patch.str"
    set fh [open $patchfile w]
    puts $fh "* Custom neutral glycine N-terminus patch (NNEG)"
    puts $fh "* NNEU analog with alpha carbon retyped CT1 -> CT2"
    puts $fh "*"
    puts $fh "   36  1"
    puts $fh ""
    puts $fh "PRES NNEG         0.00 ! neutral Gly N-terminus (NNEU analog, CA=CT2)"
    puts $fh "GROUP"
    puts $fh "ATOM N    NH2   -0.96"
    puts $fh "ATOM HT1  H      0.34"
    puts $fh "ATOM HT2  H      0.34"
    puts $fh "ATOM CA   CT2    0.10"
    puts $fh "ATOM HA1  HB2    0.09"
    puts $fh "ATOM HA2  HB2    0.09"
    puts $fh "DELETE ATOM HN"
    puts $fh "BOND HT1 N"
    puts $fh "BOND HT2 N"
    puts $fh "DONOR HT1 N"
    puts $fh "DONOR HT2 N"
    puts $fh "IC HT1  N    CA   C     0.0000 0.0000 180.0000 0.0000 0.0000"
    puts $fh "IC HT2  CA   *N   HT1   0.0000 0.0000 120.0000 0.0000 0.0000"
    puts $fh ""
    puts $fh "END"
    close $fh
    puts "Wrote custom glycine patch -> $patchfile  (VALIDATE before production)"
}

# =====================================================================
# 4. Build the PSF/PDB with psfgen.
# =====================================================================
topology $TOPO
if {$need_custom_gly} { topology $patchfile }

# Residue/atom name aliases: PDB convention -> CHARMM convention.
pdbalias residue HIS HSD
pdbalias residue HID HSD
pdbalias residue HIE HSE
pdbalias residue HIP HSP
pdbalias residue MSE MET
pdbalias atom ILE CD1 CD
pdbalias atom SER HG  HG1
pdbalias atom CYS HG  HG1

# Feed psfgen a protein-only, heavy-atom PDB with terminal carboxylate
# oxygens removed (CT2 rebuilds the C-terminus; psfgen rebuilds all H).
set bsel [atomselect $mol "protein and noh and not name OXT OT1 OT2"]
$bsel writepdb _protein_heavy.pdb
$bsel delete
mol delete $mol

segment $SEGID {
    pdb _protein_heavy.pdb
    first $NTER_PATCH
    last  $CTER_PATCH
}
coordpdb _protein_heavy.pdb $SEGID

guesscoord                       ;# place atoms added by the patches (amide H, etc.)
regenerate angles dihedrals      ;# rebuild angle/dihedral lists after patching

writepsf "${outbase}.psf"
writepdb "${outbase}.pdb"
puts "Wrote ${outbase}.psf and ${outbase}.pdb"

# =====================================================================
# 5. Sanity check: reload and report the total charge.
#    Termini are neutral, so this should equal the side-chain-only net
#    charge (e.g. #Arg + #Lys - #Asp - #Glu, +His if protonated), NOT
#    necessarily zero. If it is off by +/-1 a terminus didn't neutralize.
# =====================================================================
set chk [mol new "${outbase}.psf" type psf waitfor all]
mol addfile "${outbase}.pdb" type pdb waitfor all molid $chk
set allsel [atomselect $chk "all"]
set q 0.0
foreach c [$allsel get charge] { set q [expr {$q + $c}] }
puts [format "Total system charge: %.4f  (should be an integer; termini ~ 0)" $q]
$allsel delete
mol delete $chk

puts "Done."
exit