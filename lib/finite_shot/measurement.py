"""Majorana-pairing measurement settings used by the finite-shot replay.

Extracted from the measurement-grouping code used for the manuscript. Only the
functions needed to rebuild the greedy Majorana groups and their readout
circuits for the six- and 12-qubit finite-shot Hamiltonians (Table VII) are kept.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def paulis_to_support(packed, nq, max_degree=6):
    n = len(packed); width = (nq+7)//8
    support = np.full((n,max_degree),-1,np.int16)
    degree = np.zeros(n,np.int8)
    for i in range(n):
        suffix = 0
        # Descending qubits produces Majorana indices in descending order.
        for q in range(nq-1,-1,-1):
            x = (packed[i,q//8] >> (q%8)) & 1
            z = (packed[i,width+q//8] >> (q%8)) & 1
            my = z ^ suffix
            mx = x ^ my
            if my:
                if degree[i]>=max_degree: raise ValueError('Majorana degree exceeds configured maximum')
                support[i,degree[i]]=2*q+1; degree[i]+=1
            if mx:
                if degree[i]>=max_degree: raise ValueError('Majorana degree exceeds configured maximum')
                support[i,degree[i]]=2*q; degree[i]+=1
            suffix ^= x
        d=degree[i]
        if d%2: raise ValueError('Odd Majorana degree: parity-preserving Hamiltonian required')
        for a in range(d//2):
            tmp=support[i,a]; support[i,a]=support[i,d-1-a]; support[i,d-1-a]=tmp
    return support,degree

@njit(cache=True)
def operator_bits(s, d, q):
    m = len(d); w = (q+63)//64
    a = np.zeros((m, (2*q+63)//64), np.uint64)
    x = np.zeros((m, w), np.uint64); z = np.zeros_like(x)
    for i in range(m):
        for j in range(d[i]):
            k = int(s[i,j]); p = k//2
            a[i,k//64] ^= np.uint64(1) << np.uint64(k%64)
            x[i,p//64] ^= np.uint64(1) << np.uint64(p%64)
            for t in range(p): z[i,t//64] ^= np.uint64(1) << np.uint64(t%64)
            if k%2: z[i,p//64] ^= np.uint64(1) << np.uint64(p%64)
    return a, x, z

@njit(cache=True, inline='always')
def matching_compatible(row, degree, mate):
    for j in range(degree):
        partner = mate[row[j]]
        if partner < 0: continue
        found = False
        for k in range(degree):
            if row[k] == partner: found = True; break
        if not found: return False
    return True

@njit(cache=True)
def pairing_insertion(s, d, order, q):
    """First-fit partial matchings; pair free labels in sorted order.

    This is a complete support-aware heuristic for degrees 2, 4, AND 6.
    It is not the published universal two-body schedule.
    """
    capacity = 32; ng = 0; mates = np.full((capacity, 2*q), -1, np.int16)
    ids = np.full(len(d), -1, np.int32)
    for i in order:
        g = 0
        while g < ng and not matching_compatible(s[i], d[i], mates[g]): g += 1
        if g == ng:
            if ng == capacity:
                new = np.full((2*capacity,2*q), -1, np.int16); new[:capacity] = mates; mates = new; capacity *= 2
            ng += 1
        ids[i] = g; previous = -1
        for j in range(d[i]):
            v = s[i,j]
            if mates[g,v] >= 0: continue
            if previous < 0: previous = v
            else: mates[g,previous] = v; mates[g,v] = previous; previous = -1
        assert previous == -1
    for g in range(ng):
        previous = -1
        for v in range(2*q):
            if mates[g,v] >= 0: continue
            if previous < 0: previous = v
            else: mates[g,previous] = v; mates[g,v] = previous; previous = -1
    return ids, mates[:ng]

def matching_readout(s, d, ids, mates):
    """G_A = sign * product of selected B_l, B_l=i gamma_a gamma_b (a<b)."""
    pairs = np.array([[(a,int(b)) for a,b in enumerate(row) if a < b] for row in mates], np.int16)
    slots = np.empty_like(mates)
    for g,row in enumerate(pairs):
        for k,(a,b) in enumerate(row): slots[g,a] = slots[g,b] = k
    pair_indices, signs = _readout(s,d,ids,mates,slots)
    return pairs, pair_indices, signs

@njit(cache=True)
def _readout(s, d, ids, mates, slots):
    selections = np.full((len(d),3), -1, np.int16); signs = np.ones(len(d), np.int8)
    for i in range(len(d)):
        g = ids[i]; seq = np.empty(6, np.int16); count = 0
        for j in range(d[i]):
            a = s[i,j]; b = mates[g,a]
            assert b >= 0
            if a < b:
                selections[i,count] = slots[g,a]
                seq[2*count] = a; seq[2*count+1] = b; count += 1
        assert 2*count == d[i]
        inv = 0
        for j in range(d[i]):
            for k in range(j):
                if seq[k] > seq[j]: inv += 1
        if inv%2: signs[i] = -1
    return selections, signs

def labels_from_bits(x,z,q):
    labels = []
    for xx,zz in zip(x,z):
        chars = []
        for k in range(q):
            xb = (int(xx[k//64]) >> (k%64)) & 1; zb = (int(zz[k//64]) >> (k%64)) & 1
            chars.append('IXZY'[xb+2*zb])
        labels.append(''.join(chars[::-1]))
    return labels

def basis_jw(bits,q):
    shape = bits.shape[:-1]
    x,z = _basis_jw(np.ascontiguousarray(bits).reshape(-1,bits.shape[-1]),q)
    return x.reshape(*shape,x.shape[-1]),z.reshape(*shape,z.shape[-1])

@njit(cache=True)
def _basis_jw(bits,q):
    x = np.zeros((len(bits),(q+63)//64),np.uint64); z = np.zeros_like(x)
    for i in range(len(bits)):
        for k in range(2*q):
            if (bits[i,k//64] >> np.uint64(k%64)) & np.uint64(1):
                p = k//2; x[i,p//64] ^= np.uint64(1) << np.uint64(p%64)
                for b in range(p//64): z[i,b] ^= np.uint64(0xffffffffffffffff)
                z[i,p//64] ^= (np.uint64(1)<<np.uint64(p%64))-np.uint64(1)
                if k%2: z[i,p//64] ^= np.uint64(1) << np.uint64(p%64)
    return x,z

def setting_circuit(data, ident, method, group):
    """Return C: apply C to the state, then measure all qubits in Z.

    Use PauliList.evolve(Clifford(C), frame='s') for signed Z readout masks.
    Qiskit synthesis here is generic; no optimal gate-count/depth claim is made.
    """
    from qiskit import QuantumCircuit
    from qiskit.synthesis import synth_circuit_from_stabilizers
    q = int(np.asarray(data[ident+'__nqubits'])); prefix = ident+'__'+method+'__'
    if method == 'qwc':
        bx,bz = data[prefix+'basis_x'][group],data[prefix+'basis_z'][group]
        c = QuantumCircuit(q)
        for k in range(q):
            x = (int(bx[k//64]) >> (k%64)) & 1; z = (int(bz[k//64]) >> (k%64)) & 1
            if x and z: c.sdg(k)
            if x: c.h(k)
        return c
    if method == 'fc':
        rank = int(data[prefix+'ranks'][group])
        x,z = basis_jw(data[prefix+'generators'][group,:rank],q)
    else:
        pairs = data[prefix+'pairs'][group]
        s = np.full((q,6),-1,np.int16); s[:,:2] = pairs
        _,x,z = operator_bits(s,np.full(q,2,np.int8),q)
    return synth_circuit_from_stabilizers(labels_from_bits(x,z,q), allow_underconstrained=True).inverse()

@njit(cache=True)
def verify_matching(s,d,ids,mates):
    for i in range(len(d)):
        for j in range(d[i]):
            partner=mates[ids[i],s[i,j]];count=0
            for k in range(d[i]):
                if s[i,k]==partner:count+=1
            assert count==1
