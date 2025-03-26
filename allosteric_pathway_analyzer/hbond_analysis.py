from subprocess import run

top = "/home/yehor/Desktop/AllostericPathwayAnalyzer/untracked/MD_data/p53_WT_nowat.prmtop"
traj = "/home/yehor/Desktop/AllostericPathwayAnalyzer/untracked/MD_data/p53_WT_md1000_str.nc"
cpptraj = "/home/yehor/miniforge3/envs/AmberTools23/bin/cpptraj"

command = (
    f"echo 'parm {top}\n"
    f"trajin {traj}\n"
    f"hbond HB out hb.dat avgout avg_hb.dat "
    f"donormask :1-393 acceptormask :1-393 distance 3.5 angle 120\n"
    f"run' | {cpptraj}"
)

run(command, check=True, shell=True)

"""
The values of 3.5 Å for distance and 120° for the angle are commonly used default criteria for identifying hydrogen bonds in molecular dynamics analyses.
Distance Criterion: 3.5 Å

    Why 3.5 Å?
    Hydrogen bonds are typically characterized by a close proximity between the hydrogen bond donor and acceptor atoms.
    Empirical studies and crystallographic data have shown that most hydrogen bonds have donor-acceptor distances of around 2.7-3.5 Å.
    A 3.5 Å cutoff ensures that you capture most physiologically relevant hydrogen bonds without including interactions that are too weak
    or are just due to van der Waals contacts.

    Is it Optimal?
    For many proteins and nucleic acids, 3.5 Å works well as a balance between sensitivity and specificity.
    However, in systems with unusual environments (e.g., membrane proteins, highly flexible regions, or at high temperatures),
    you might consider testing a range of distance cutoffs to see if 3.5 Å appropriately captures the interactions of interest.

Angle Criterion: 120°

    Why 120°?
    The angle parameter refers to the hydrogen-donor-acceptor angle. In an ideal hydrogen bond, this angle is close to 180°(a straight line),
    which maximizes the interaction strength. However, biological systems often exhibit deviations from perfect linearity.
    A cutoff of 120° is commonly used because it allows for some flexibility while still ensuring that the geometry is favorable
    for hydrogen bonding. Angles below 120° are typically considered too bent to form a strong hydrogen bond.

    Is it Optimal?
    The 120° cutoff is a widely accepted compromise that captures the majority of meaningful hydrogen bonds.
    That said, if your system is known to have particularly linear or particularly bent hydrogen bonds
    (or if you're interested in more transient, weak interactions), you might adjust the angle criterion.
    For instance, a stricter cutoff (e.g., 135° or 150°) would capture only the most linear and likely stronger bonds,
    while a lower cutoff might include weaker or non-canonical interactions.
"""

"""
Columns in the Output

    Acceptor:
    This column lists the acceptor atom involved in the hydrogen bond.
    For example, "ASN_310@OD1" indicates the oxygen atom (OD1) of asparagine at residue 310 is acting as the acceptor.

    DonorH:
    This is the hydrogen atom that participates as the donor.
    In "SER_315@HG", the hydrogen (HG) on serine residue 315 is the one being donated.

    Donor:
    This represents the heavy atom (usually oxygen or nitrogen) to which the donor hydrogen is covalently bonded. 
    In the same row, "SER_315@OG" is the oxygen atom of serine that holds the hydrogen.

    Frames:
    This shows the number of simulation frames (out of the total, e.g. 1000) in which the hydrogen bond was detected. 
    A value of 1000 means the bond was observed in every frame, suggesting a very stable interaction.

    Frac:
    The fraction (or occupancy) of frames where the bond is present. 
    A value of 1.0000 indicates the bond exists in 100% of the frames, whereas a value slightly less than 1 (e.g., 0.9980) 
    implies it's nearly always present but may occasionally be absent.

    AvgDist:
    This is the average distance (usually in Ångstroms) between the donor and acceptor atoms over the frames where the bond is present.
    Typical hydrogen bonds are around 2.7-3.5 Å; values like 2.81 or 2.90 Å indicate a strong, close interaction.

    AvgAng:
    The average angle (in degrees) of the hydrogen bond, usually measured as the donor-hydrogen-acceptor angle. 
    Ideally, this angle is close to 180° for a strong, linear hydrogen bond. Values like 163° or 164° are quite favorable, 
    indicating a nearly linear arrangement.
"""; "I think I can probably run the analysis only on a subset of frames to get average values per batch"

"""
Hydrogen Bond Strength (HBS) = Frac * 1/AvgDist * AvgAng/180
^
| This is not a universally accepted formula, however.
"""