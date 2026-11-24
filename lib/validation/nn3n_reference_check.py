"""Check the published NN+3N operator against an independent NuQuLib mapping.

The published e_max=1 NN+3N operator (small_operators.npz, key
3nf_e1_compact_common) was produced by the factored three-body recoupling and
coefficient accumulation used for all NN+3N spaces. This script maps the same
SRG 2.0 fm^-1 NN and 3N inputs with NuQuLib's direct Python implementation
and compares the two operators term by term in the Majorana representation.

Requires NuQuLib (https://github.com/SotaYoshida/NuQuLib) and its test
interaction files. From this repository's root:

    python lib/validation/nn3n_reference_check.py /path/to/NuQuLib/tests/interaction_file

The result is written to data/validation/nn3n_reference_check.json.
"""
import sys, json, time, hashlib, platform
from importlib.metadata import version
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'lib'/'finite_shot'))
from measurement import paulis_to_support
import nuqulib

NN = 'TwBME-HO_NN-only_N3LO_EM500_srg2.0_hw20_emax1_e2max2.kshell.snt'
NNN = ('ThBME_srg2.0_ramp40-5-36-7-32-9-28-11-24_N3LO_EM500_c1_-0.81_c3_-3.2_c4_5.4_'
       'cD_0.7_cE_-0.06_LNL2_650_500_IS_hw20from30_ms1_2_3.me3j.gz')
KEY = '3nf_e1_compact_common'
CUTOFF = 1e-8  # MeV, as for the published operators


def hermitian_sign(support, degree):
    """Sign relating a Hermitian Majorana monomial to its Jordan-Wigner Pauli string."""
    signs = np.ones(len(degree))
    for i in range(len(degree)):
        d = degree[i]; odd = 0; ny = 0
        maxq = support[i, d-1]//2 if d else 0
        for q in range(maxq+1):
            x = z = 0
            for a in range(d):
                p = support[i, a]//2; y = support[i, a] % 2
                if p == q: x ^= 1; z ^= y
                elif p > q: z ^= 1
            if x and z: ny += 1
        for a in range(d): odd += support[i, a] % 2
        phase = (d//2 + odd - ny) % 4
        assert phase % 2 == 0
        signs[i] = 1 if phase == 0 else -1
    return signs


def majorana_terms(op):
    op = op.simplify(atol=0)
    x, z, c = op.paulis.x, op.paulis.z, op.coeffs
    assert np.max(np.abs(c.imag)) < 1e-12
    n, q = x.shape; w = (q+7)//8
    packed = np.zeros((n, 2*w), np.uint8)
    for j in range(q):
        packed[:, j//8] |= x[:, j].astype(np.uint8) << (j % 8)
        packed[:, w + j//8] |= z[:, j].astype(np.uint8) << (j % 8)
    s, d = paulis_to_support(packed, q)
    h = c.real * hermitian_sign(s, d)
    ident = d == 0
    keep = ~ident & (np.abs(h) > CUTOFF)
    terms = {tuple(int(v) for v in s[i, :d[i]]): h[i] for i in np.flatnonzero(keep)}
    return q, float(h[ident].sum()), terms


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(interactions):
    interactions = Path(interactions)
    start = time.monotonic()
    _, op, _, _ = nuqulib.get_Hamiltonian(str(interactions/NN), 8, 8, fn_3NF=str(interactions/NNN),
                                         emax=1, e3max=3, ncsm=True, mapping_method='Jordan-Wigner')
    q, identity, mine = majorana_terms(op)
    elapsed = time.monotonic() - start

    archive = np.load(ROOT/'data'/'small_operators.npz')
    s, d, h = (archive[f'{KEY}__{f}'] for f in ('supports', 'degree', 'coeffs_majorana'))
    published = {tuple(int(v) for v in s[i, :d[i]]): h[i] for i in range(len(h))}
    summary = next(r for r in json.loads((ROOT/'data'/'operator_summaries.json').read_text())['rows']
                   if r['id'] == KEY)
    common = sorted(set(mine) & set(published))
    absdiff = np.array([abs(mine[t] - published[t]) for t in common])
    reldiff = absdiff / np.array([abs(published[t]) for t in common])
    record = dict(
        purpose='Independent check of the factored NN+3N recoupling and coefficient accumulation',
        published_operator=KEY, nqubits=int(q), emax=1, E3max=3, SRG_lambda_fm_inv=2.0, hw_MeV=20,
        input_nn=NN, input_nn_sha256=sha256(interactions/NN),
        input_3nf=NNN, input_3nf_sha256=sha256(interactions/NNN),
        coefficient_cutoff_MeV=CUTOFF,
        reference_nonidentity_terms=len(mine), published_nonidentity_terms=len(published),
        terms_only_in_reference=len(set(mine) - set(published)),
        terms_only_in_published=len(set(published) - set(mine)),
        max_abs_coefficient_difference_MeV=float(absdiff.max()),
        max_rel_coefficient_difference=float(reldiff.max()),
        reference_identity_coefficient_MeV=identity,
        published_identity_coefficient_MeV=float(summary['identity_coefficient']),
        reference_lambda_nonidentity_MeV=float(sum(abs(v) for v in mine.values())),
        published_lambda_nonidentity_MeV=float(summary['lambda_nonidentity']),
        mapping_seconds=round(elapsed, 1),
        environment=dict(python=platform.python_version(), numpy=version('numpy'), qiskit=version('qiskit')),
    )
    out = ROOT/'data'/'validation'/'nn3n_reference_check.json'
    out.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
