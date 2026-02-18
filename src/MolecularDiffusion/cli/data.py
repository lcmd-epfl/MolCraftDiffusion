
"""
Data CLI module for MolCraft.
Exposes data preparation, augmentation, and ASE operations via CLI commands.
"""

import click
import logging
from pathlib import Path

# Implements the structure defined in the plan
from MolecularDiffusion.runmodes.data import preparation as prep
from MolecularDiffusion.runmodes.data import augmentation as aug
from MolecularDiffusion.runmodes.data import ase_ops
from MolecularDiffusion.runmodes.data import featurization as feat

logger = logging.getLogger(__name__)

# Verify asset existence at startup

class VariadicOption(click.Option):
    """
    A Click Option that consumes all remaining arguments until the next flag.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_to_parser(self, parser, ctx):
        def parser_process(value, state):
            # method to hook to the parser.process
            done = False
            value = [value]
            # grab everything up to the next option
            while state.rargs and not done:
                for prefix in self._eat_all_parser.prefixes:
                    if state.rargs[0].startswith(prefix):
                        done = True
                if not done:
                    value.append(state.rargs.pop(0))
            value = tuple(value)
            self._previous_parser_process(value, state)

        retval = super(VariadicOption, self).add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = parser._long_opt.get(name) or parser._short_opt.get(name)
            if our_parser:
                self._eat_all_parser = our_parser
                # preventing infinite loops when aliases are present
                if our_parser.process != parser_process:
                     self._previous_parser_process = our_parser.process
                     our_parser.process = parser_process
        return retval

@click.group()
def data(): 
    """Data processing utilities."""
    pass

# --- Preparation Commands ---

@data.group()
def prepare():
    """Data preparation commands."""
    pass

@prepare.command("compile")
@click.option("--source", "-s", "--src", required=True, type=click.Path(exists=True), help="Input source (XYZ dir or NPY file)")
@click.option("--db", "-d", required=True, type=click.Path(), help="Output ASE DB path")
@click.option("--natoms", "-n", "--nat", type=click.Path(exists=True), help="Number of atoms file (for NPY)")
@click.option("--csv", "-c", type=click.Path(), help="Metadata CSV file")
@click.option("--sdf", "-sd", type=click.Path(), help="SDF file/dir for merging")
@click.option("--fraction", "-f", "--frac", default=1.0, show_default=True, help="Fraction of data to process")
@click.option("--seed", "-se", type=int, help="Random seed")
def compile_cmd(source, db, natoms, csv, sdf, fraction, seed):
    """Compile molecular data into an ASE database."""
    prep.compile_to_asedb(
        input_source=source,
        db_path=Path(db),
        input_type=None,
        natoms_file=Path(natoms) if natoms else None,
        csv_file=Path(csv) if csv else None,
        sdf_path=Path(sdf) if sdf else None,
        fraction=fraction,
        seed=seed
    )

@prepare.command("annotate")
@click.option("--db", "-d", required=True, type=click.Path(exists=True), help="ASE DB path")
@click.option("--tag", "-t", required=True, cls=VariadicOption, help="Tag name(s)")
@click.option("--value", "-v", "--val", required=True, cls=VariadicOption, help="Tag value(s)")
def annotate_cmd(db, tag, value):
    """Annotate an ASE database with a tag."""
    prep.annotate_db(Path(db), tag, value, verbose=True)

@prepare.command("generate-blocks")
@click.option("--source", "-s", "--src", required=True, type=click.Path(exists=True), help="Input source (DB, XYZ dir, or NPY)")
@click.option("--sdf", "-sd", type=click.Path(), help="Output SDF file (required for XYZ/NPY)")
@click.option("--natoms", "-n", "--nat", type=click.Path(exists=True), help="Number of atoms file (for NPY)")
@click.option("--csv", "-c", type=click.Path(exists=True), help="Metadata CSV file")
@click.option("--fraction", "-f", "--frac", default=1.0, show_default=True, help="Fraction to process")
@click.option("--indices", type=str, help='Process a slice of data, e.g., "0-10000".')
@click.option("--method", "-m", "--met", default="hybrid", show_default=True, type=click.Choice(['hybrid', 'openbabel', 'cell2mol']), help="Smilification method")
def generate_blocks_cmd(source, sdf, natoms, csv, fraction, indices, method):
    """Generate Mol Blocks, SMILES, and properties (SA, SC)."""
    # Logic is handled inside preparation.py
    prep.generate_mol_blocks_and_sdf(
        input_source=source,
        output_sdf=Path(sdf) if sdf else None,
        natoms_file=Path(natoms) if natoms else None,
        csv_file=Path(csv) if csv else None,
        fraction=fraction,
        smilify_method=method,
        indices=indices
    )

@data.command("featurize")
@click.option("--method", "-m", default="morgan", show_default=True, type=click.Choice(['morgan', 'soap']), help="Featurization method")
@click.option("--input", "-i", required=True, type=click.Path(exists=True), help="Input source (XYZ dir or DB)")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output directory")
@click.option("--format", "-f", default="npy", show_default=True, type=click.Choice(['npy', 'safetensors']), help="Output format")
@click.option("--readout", "-ro", default="mean", show_default=True, type=click.Choice(['mean', 'sum']), help="Readout mode for SOAP")
@click.option("--smilify-method", "-sm", default="hybrid", show_default=True, type=click.Choice(['hybrid', 'openbabel', 'cell2mol']), help="Smilification method for Morgan")
@click.option("--radius", "-r", default=2, show_default=True, help="Morgan radius")
@click.option("--nbits", "-n", default=2048, show_default=True, help="Morgan nbits")
@click.option("--rcut", "-rc", default=5.0, show_default=True, help="SOAP r_cut")
@click.option("--nmax", "-nm", default=8, show_default=True, help="SOAP n_max")
@click.option("--lmax", "-lm", default=6, show_default=True, help="SOAP l_max")
def featurize_cmd(method, input, output, format, readout, smilify_method, radius, nbits, rcut, nmax, lmax):
    """Featurize 3D molecules into vectors."""
    feat.featurize(
        input_source=input,
        output_dir=output,
        method=method,
        format=format,
        readout=readout,
        smilify_method=smilify_method,
        radius=radius,
        nbits=nbits,
        rcut=rcut,
        nmax=nmax,
        lmax=lmax
    )

# --- Augmentation Commands ---

@data.group()
def augment():
    """Data augmentation commands."""
    pass

@augment.command("charge")
@click.option("--input", "-i", "--inp", required=True, type=click.Path(exists=True), help="Input directory (XYZ) or DB")
@click.option("--output", "-o", "--out", required=True, type=click.Path(), help="Output directory or DB")
@click.option("--max-h", "-mh", default=1, show_default=True, help="Max H atoms to change")
@click.option("--fraction", "-f", "--frac", default=1.0, show_default=True, help="Fraction to process")
@click.option("--db", "-d", is_flag=True, help="Enable DB mode")
def charge_cmd(input, output, max_h, fraction, db):
    """Augment data by random charge modification."""
    aug.augment_charge(input, output, max_h_change=max_h, fraction=fraction, is_db_mode=db)

@augment.command("distortion")
@click.option("--input", "-i", "--inp", required=True, type=click.Path(exists=True), help="Input directory (XYZ) or DB")
@click.option("--output", "-o", "--out", required=True, type=click.Path(), help="Output directory or DB")
@click.option("--sigma", "-s", "--sig", default=0.1, show_default=True, help="Noise standard deviation (d_max)")
@click.option("--fraction", "-f", "--frac", default=1.0, show_default=True, help="Fraction to process")
@click.option("--freeze", "-fr", cls=VariadicOption, help="Indices of atoms to freeze (0-indexed)")
def distortion_cmd(input, output, sigma, fraction, freeze):
    """Augment data by random coordinate distortion."""
    import ast
    if freeze and isinstance(freeze, str) and freeze.startswith("("):
        try:
            freeze = ast.literal_eval(freeze)
        except (ValueError, SyntaxError):
            pass # Fallback or error

    freeze_indices = [int(i) for i in freeze] if freeze else None
    aug.augment_distortion(Path(input), Path(output), sigma=sigma, fraction=fraction, is_db_mode=None, freeze_indices=freeze_indices)

@augment.command("size")
@click.option("--input", "-i", "--inp", required=True, type=click.Path(exists=True), help="Input ASE DB")
@click.option("--output", "-o", "--out", required=True, type=click.Path(), help="Output ASE DB")
@click.option("--s-start", "-ss", default=60, show_default=True, help="Start size")
@click.option("--t-start", "-ts", default=50000, show_default=True, help="Target count at start")
@click.option("--s-end", "-se", default=120, show_default=True, help="End size")
@click.option("--t-end", "-te", default=5000, show_default=True, help="Target count at end")
@click.option("--strength", "-str", default=1.0, show_default=True, help="Augmentation strength")
@click.option("--decay", "-dec", default=1.0, show_default=True, help="Decay power")
@click.option("--invert", "-inv", default=0.5, show_default=True, help="Inversion probability")
@click.option("--plot-prefix", "-pp", type=click.Path(), help="Prefix for plots")
def size_cmd(input, output, s_start, t_start, s_end, t_end, strength, decay, invert, plot_prefix):
    """Augment data to balance molecule sizes."""
    aug.augment_size(
        Path(input), Path(output),
        s_start, t_start, s_end, t_end,
        strength, decay_power=decay, invert_prob=invert,
        plot_prefix=Path(plot_prefix) if plot_prefix else None
    )


# --- ASE Operations Commands ---

@data.group("ase-ops")
def ase_ops_group():
    """ASE database operations."""
    pass

@ase_ops_group.command("merge")
@click.option("--input", "-i", "--inp", required=True, type=click.Path(exists=True), help="Input directory containing DBs")
@click.option("--output", "-o", "--out", required=True, type=click.Path(), help="Output merged DB path")
@click.option("--recursive", "-r", "--rec", is_flag=True, help="Search recursively")
def merge_cmd(input, output, recursive):
    """Merge multiple ASE databases."""
    ase_ops.merge_dbs(Path(input), Path(output), recursive=recursive)

@ase_ops_group.command("inspect")
@click.option("--db", "-d", required=True, type=click.Path(exists=True), help="ASE DB path")
@click.option("--output", "-o", "--out", type=click.Path(), help="Directory to save plots")
@click.option("--keys", "-k", multiple=True, help="Specific keys to plot")
@click.option("--limit", "-l", "--lim", default=10, show_default=True, help="Number of entries to print")
def inspect_cmd(db, output, keys, limit):
    """Inspect an ASE database and optionally plot statistics."""
    ase_ops.inspect_db(Path(db), output_dir=Path(output) if output else None, keys_to_plot=list(keys), limit_print=limit)

@ase_ops_group.command("split")
@click.option("--db", "-d", required=True, type=click.Path(exists=True), help="ASE DB path")
@click.option("--output", "-o", "--out", required=True, type=click.Path(), help="Output directory")
@click.option("--n", "-num", default=2, show_default=True, help="Number of splits")
def split_cmd(db, output, n):
    """Split an ASE database."""
    ase_ops.split_db(Path(db), Path(output), n_splits=n)

@ase_ops_group.command("sample")
@click.option("--input", "-i", "--inp", required=True, type=click.Path(exists=True), help="Input ASE DB")
@click.option("--output", "-o", "--out", required=True, type=click.Path(), help="Output file/dir")
@click.option("--fraction", "-f", "--frac", type=float, help="Fraction to sample")
@click.option("--number", "-num", type=int, help="Number to sample")
@click.option("--seed", "-s", "-se", type=int, help="Random seed")
@click.option("--verify", "-v", "--ver", is_flag=True, help="Verify atom order matches RDKit")
def sample_cmd(input, output, fraction, number, seed, verify):
    """Sample entries from an ASE database."""
    ase_ops.sample_db(
        Path(input), Path(output),
        output_type=None,
        fraction=fraction, number=number,
        seed=seed, verify_clean=verify
    )
