#!/usr/bin/env vmd
# -*- mode: tcl -*-
# =====================================================================
# solvate_fibril.tcl
#
# Solvate + ionize a PERIODIC fibril assembly.
#
# The fibril axis is NOT padded: its box length is set to exactly
#     nlevels x rise
# so that the peptide stack continues seamlessly into its own periodic
# image (an "infinite" fibril).  The other two axes get normal padding.
#
# Because solvate knows nothing about periodicity, it will place waters
# at the axial seam that clash with the periodic IMAGE of the peptide.
# Step 4 finds those with VMD's `pbwithin` keyword and deletes them via
# psfgen, then re-ionizes.
#
# The exact box is written to <out>.box -- a PSF stores no box, and the
# axial length must NOT be re-measured from water extents downstream
# (that would silently change the fibril rise).  Feed <out>.box to
# psf.setBox() in OpenMM.
#
# Run:
#   vmd -dispdev text -e solvate_fibril.tcl -args \
#       -psf full.psf -pdb full.pdb -o ionized \
#       -axis z -pad 15 -sc 0.15 -cation SOD [-rise 4.8]
#
#   -axis    fibril axis: x | y | z            (default z)
#   -pad     padding on the two NON-fibril axes, Angstrom (default 15)
#   -rise    inter-peptide rise along the axis, Angstrom.
#            If omitted it is MEASURED from the per-copy centers -- always
#            check the printed value against what you built.
#   -sc      salt concentration, mol/L          (default 0.15)
#   -cation  SOD | SOD                          (default SOD)
#   -b       water/solute min distance, Ang     (default 2.4)
# =====================================================================

package require solvate
package require autoionize
package require pbctools
package require psfgen

# ------------------------- defaults / args ---------------------------
array set opt {
    -psf "" -pdb "" -o "ionized" -axis z -pad 15
    -rise "" -sc 0.15 -cation SOD -b 2.4
}
foreach {k v} $argv {
    if {![info exists opt($k)]} { puts "ERROR: unknown option '$k'"; exit 1 }
    set opt($k) $v
}
foreach req {-psf -pdb} {
    if {$opt($req) eq ""} { puts "ERROR: $req is required"; exit 1 }
}
set AX [string tolower $opt(-axis)]
set ai [lsearch {x y z} $AX]
if {$ai < 0} { puts "ERROR: -axis must be x, y or z"; exit 1 }
set PAD  $opt(-pad)
set OUT  $opt(-o)
puts "fibril axis: $AX   lateral padding: $PAD A"

# =====================================================================
# 1. Load the solute; group it into axial "levels" (one level per rung
#    of the stack; with two stacks of ten there are 2 copies per level).
# =====================================================================
set mol [mol new $opt(-psf) type psf waitfor all]
mol addfile $opt(-pdb) type pdb waitfor all molid $mol

set all  [atomselect $mol all]
set segs [lsort -unique [$all get segid]]
puts "solute: [$all num] atoms in [llength $segs] segments"

# per-copy center, projected onto the fibril axis
set proj {}
foreach s $segs {
    set sel [atomselect $mol "segid $s"]
    lappend proj [lindex [measure center $sel weight mass] $ai]
    $sel delete
}
set proj [lsort -real $proj]

# cluster projected centers into levels (1.0 A tolerance)
set levels {}
set acc {}
foreach p $proj {
    if {[llength $acc] && [expr {$p - [lindex $acc end]}] > 1.0} {
        set sum 0.0; foreach a $acc { set sum [expr {$sum + $a}] }
        lappend levels [expr {$sum / [llength $acc]}]
        set acc {}
    }
    lappend acc $p
}
if {[llength $acc]} {
    set sum 0.0; foreach a $acc { set sum [expr {$sum + $a}] }
    lappend levels [expr {$sum / [llength $acc]}]
}
set nlev [llength $levels]
if {$nlev < 2} { puts "ERROR: found $nlev axial level(s); cannot infer a rise."; exit 1 }

# rise: supplied, or measured as the mean level spacing
set measured [expr {([lindex $levels end] - [lindex $levels 0]) / ($nlev - 1)}]
if {$opt(-rise) ne ""} {
    set RISE $opt(-rise)
    puts [format "levels: %d   rise: %.3f A (supplied; measured %.3f A)" $nlev $RISE $measured]
    if {abs($RISE - $measured) > 0.5} {
        puts "WARNING: supplied rise differs from measured by >0.5 A -- check -axis."
    }
} else {
    set RISE $measured
    puts [format "levels: %d   rise: %.3f A (MEASURED -- verify against your build)" $nlev $RISE]
}

# =====================================================================
# 2. Box.  Axial: exactly nlev x RISE, centered so each level owns a
#    slab of thickness RISE.  Lateral: solute extent + 2 x PAD.
# =====================================================================
set mm  [measure minmax $all]
set lo  [lindex $mm 0]
set hi  [lindex $mm 1]

set blo {}; set bhi {}
for {set d 0} {$d < 3} {incr d} {
    if {$d == $ai} {
        lappend blo [expr {[lindex $levels 0]   - $RISE/2.0}]
        lappend bhi [expr {[lindex $levels end] + $RISE/2.0}]
    } else {
        lappend blo [expr {[lindex $lo $d] - $PAD}]
        lappend bhi [expr {[lindex $hi $d] + $PAD}]
    }
}
set LX [expr {[lindex $bhi 0] - [lindex $blo 0]}]
set LY [expr {[lindex $bhi 1] - [lindex $blo 1]}]
set LZ [expr {[lindex $bhi 2] - [lindex $blo 2]}]
puts [format "box: %.3f x %.3f x %.3f A   (axial %s = %d x %.3f = %.3f)" \
      $LX $LY $LZ $AX $nlev $RISE [expr {$nlev*$RISE}]]

# minimum-image warning against a 12 A cutoff
foreach L [list $LX $LY $LZ] n {x y z} {
    if {$L < 24.0} { puts "WARNING: $n edge ${L} A < 2 x 12 A cutoff." }
}
$all delete
mol delete $mol

# =====================================================================
# 3. Solvate into that exact box.
# =====================================================================
solvate $opt(-psf) $opt(-pdb) -minmax [list $blo $bhi] -b $opt(-b) -o ${OUT}_wet
puts "solvate done -> ${OUT}_wet.psf/.pdb"

# =====================================================================
# 4. Trim waters clashing with the PERIODIC IMAGE of the solute.
#    `pbwithin` needs the unit cell, so set it first.  Only the axial
#    seam can produce these (the lateral axes are padded), but the
#    selection is general.
# =====================================================================
set wm [mol new ${OUT}_wet.psf type psf waitfor all]
mol addfile ${OUT}_wet.pdb type pdb waitfor all molid $wm
pbc set [list $LX $LY $LZ 90 90 90] -all

set solute "not water and not resname SOD SOD CLA"
set bad [atomselect $wm "water and pbwithin $opt(-b) of ($solute)"]
set doomed [lsort -unique [$bad get {segid resid}]]
puts "seam clashes: [llength $doomed] water residues to delete"
$bad delete
mol delete $wm

if {[llength $doomed]} {
    resetpsf
    readpsf  ${OUT}_wet.psf
    coordpdb ${OUT}_wet.pdb
    foreach sr $doomed { delatom [lindex $sr 0] [lindex $sr 1] }
    writepsf ${OUT}_trim.psf
    writepdb ${OUT}_trim.pdb
    resetpsf
} else {
    file copy -force ${OUT}_wet.psf ${OUT}_trim.psf
    file copy -force ${OUT}_wet.pdb ${OUT}_trim.pdb
}

# =====================================================================
# 5. Ionize.
# =====================================================================
autoionize -psf ${OUT}_trim.psf -pdb ${OUT}_trim.pdb \
           -sc $opt(-sc) -cation $opt(-cation) -o $OUT

foreach f [list ${OUT}.psf ${OUT}.pdb] {
    if {![file exists $f]} {
        puts "ERROR: autoionize did not produce $f -- check its output above."
        exit 1
    }
}

# =====================================================================
# 6. Report + write the exact box for OpenMM.
# =====================================================================
set fm [mol new ${OUT}.psf type psf waitfor all]
mol addfile ${OUT}.pdb type pdb waitfor all molid $fm
pbc set [list $LX $LY $LZ 90 90 90] -all
set fs [atomselect $fm all]
set q 0.0; foreach c [$fs get charge] { set q [expr {$q + $c}] }
set nw [llength [lsort -unique [[atomselect $fm "water and name OH2"] get {segid resid}]]]

# Ion resnames come from the cation actually requested, so this cannot
# fall out of sync with autoionize (a hard-coded list silently matches
# nothing if -cation changes).
set ionsel "resname $opt(-cation) CLA"
set nion [[atomselect $fm $ionsel] num]
set ib [atomselect $fm "$ionsel and pbwithin 4.0 of ($solute)"]
if {[$ib num]} { puts "WARNING: [$ib num] ion(s) within 4 A of a periodic image of the solute." }
$ib delete

puts "-----------------------------------------------------------"
puts [format "atoms       : %d" [$fs num]]
puts [format "waters      : %d" $nw]
puts [format "ions        : %d  (%s / CLA)" $nion $opt(-cation)]
puts [format "net charge  : %+.4f  (should be ~0)" $q]
puts [format "box (A)     : %.4f %.4f %.4f" $LX $LY $LZ]
puts [format "box (nm)    : %.5f %.5f %.5f" [expr {$LX/10.0}] [expr {$LY/10.0}] [expr {$LZ/10.0}]]
puts "-----------------------------------------------------------"
$fs delete; mol delete $fm

set bf [open ${OUT}.box w]
puts $bf [format "%.6f %.6f %.6f" [expr {$LX/10.0}] [expr {$LY/10.0}] [expr {$LZ/10.0}]]
close $bf
puts "wrote ${OUT}.psf / ${OUT}.pdb / ${OUT}.box  (box in nm; axis $AX index $ai)"

# Intermediates kept for validation. Set CLEAN to 1 to remove them.
set CLEAN 0
if {$CLEAN} {
    foreach f [list ${OUT}_wet.psf ${OUT}_wet.pdb ${OUT}_trim.psf ${OUT}_trim.pdb] {
        file delete $f
    }
}
exit