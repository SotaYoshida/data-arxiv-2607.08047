"""Replay the six fixed-budget finite-shot protocols of Table VII.

Run from the repository root:  python lib/finite_shot/replay.py
For each nucleus and method, the 1,200 energy estimates are recomputed from the
Hamiltonians in data/finite_shot_inputs.json and the protocols in
data/finite_shot.json, and checked against data/finite_shot_energies.npz.
Requires qiskit in addition to requirements.txt.
"""
import sys
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import linalg, stats
from numba import njit
from qiskit import transpile
from qiskit.quantum_info import Statevector, PauliList, Clifford

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data'
sys.path.insert(0, str(HERE))
import measurement as core
CASES = json.loads((DATA / 'finite_shot_inputs.json').read_text())
PROTOCOLS = json.loads((DATA / 'finite_shot.json').read_text())['cases']
R = 4
CONFIG = dict(seed=4192206, validation=1200, calibration_goal=.985)


@njit
def paths(perm,left,right,psi,r,size):
    old=psi.copy();new=psi.copy()
    out=np.empty((size,len(psi)),np.complex128)
    for k in range(size):
        out[k]=old
        if k+1==size:break
        for _ in range(r):
            for g in range(len(perm)):
                for i in range(len(psi)):
                    new[i]=left[g,i]*old[i]+right[g,i]*old[perm[g,i]]
                old,new=new,old
    return out

def setup(name):
    case=CASES[name];q=case['q'];D=2**q;terms=case['terms']
    occ=case['occupied'];det=sum(1<<i for i in occ)
    sector=np.array([i for i in range(D) if i.bit_count()==len(occ) and
                     sum(x for j,x in enumerate(case['m2']) if i>>j&1)==0])
    def mask(p,letters):return sum(1<<j for j,c in enumerate(reversed(p)) if c in letters)
    orbit={det}
    for x in sorted({mask(p,'XY') for p,c in terms}):orbit.update([i^x for i in orbit])
    orbit=np.array(sorted(orbit));where=np.full(D,-1);where[orbit]=np.arange(len(orbit))
    parity=np.array([(-1)**i.bit_count() for i in range(D)])
    blocks=[];rows={};perms={}
    for p,c in terms:
        x=mask(p,'XY');perm=where[orbit^x];phase=(1j**p.count('Y'))*parity[(orbit^x)&mask(p,'YZ')]
        if not blocks or blocks[-1][0]!=x:blocks.append((x,[]))
        blocks[-1][1].append((c,phase))
        if x not in rows:rows[x]=np.zeros(len(orbit),complex);perms[x]=perm
        rows[x]+=c*phase
    inds=where[sector];assert min(inds)>=0
    reverse=np.full(len(orbit),-1);reverse[inds]=np.arange(len(sector));H=np.zeros((len(sector),len(sector)),complex)
    for x,row in rows.items():
        col=reverse[perms[x][inds]];keep=col>=0
        assert np.max(abs(row[inds][~keep]),initial=0)<1e-10
        H[np.flatnonzero(keep),col[keep]]+=row[inds[keep]]
    np.testing.assert_allclose(H,H.conj().T,atol=1e-12)
    ev,v=linalg.eigh(H);sdet=list(sector).index(det)
    return SimpleNamespace(name=name,q=q,D=D,case=case,terms=terms,det=det,orbit=orbit,sector=sector,inds=inds,
         ev=ev,v=v,sdet=sdet,psi=(orbit==det).astype(complex),E0=ev[0]+case['shift'],shift=case['shift'],
         blocks=blocks,rows=rows,perms=perms,parity=parity,mask=mask,
         p0=abs(v[sdet,0])**2,radius=np.max(np.sum(np.abs(list(rows.values())),axis=0)))

def trajectory(s,dt,length,exact=False):
    if exact:
        out=np.zeros((length,len(s.orbit)),complex)
        out[:,s.inds]=(np.exp(-1j*np.arange(length)[:,None]*dt*s.ev)*s.v[s.sdet].conj())@s.v.T
        return out
    a=[];b=[];perms=[]
    for x,members in s.blocks:
        perm=s.perms[x];left=np.ones(len(s.orbit),complex);right=np.zeros_like(left)
        for c,phase in members:
            cc=np.cos(c*dt/R);ss=-1j*np.sin(c*dt/R)*phase
            left,right=cc*left+ss*right[perm],cc*right+ss*left[perm]
        a.append(left);b.append(right);perms.append(perm)
    out=paths(np.asarray(perms),np.asarray(a),np.asarray(b),s.psi,R,length)
    assert np.max(abs(np.sum(abs(out)**2,axis=1)-1))<1e-8
    # Independent chronological product on the full Pauli-connected block.
    reference=s.psi.copy()
    for _ in range(R):
        for x,members in s.blocks:
            for c,phase in members:
                reference=np.cos(c*dt/R)*reference-1j*np.sin(c*dt/R)*phase*reference[s.perms[x]]
    if length>1:np.testing.assert_allclose(reference,out[1],atol=1e-11)
    return out

def matrices(s,traj):
    phi=traj.T
    applied=sum(v[:,None]*phi[s.perms[x]] for x,v in s.rows.items())
    return phi.conj().T@applied,phi.conj().T@phi

def ritz(h,S,tau,shift,vector=False):
    vals,u=np.linalg.eigh(S);ranks=(vals>max(tau,1.e-11)).sum(axis=-1)
    if vector:
        k=np.flatnonzero(vals>max(tau,1.e-11))
        if len(k)==0:return np.nan,np.zeros(len(vals))
        w=u[:,k]/np.sqrt(vals[k]);e,a=np.linalg.eigh(w.conj().T@h@w)
        return e[0]+shift,w@a[:,0]
    out=np.full(len(S),np.nan)
    for rank in np.unique(ranks[ranks>0]):
        use=ranks==rank;w=u[use,:,-rank:]/np.sqrt(vals[use,None,-rank:])
        out[use]=np.linalg.eigvalsh(w.conj().swapaxes(-1,-2)@h[use]@w)[:,0]+shift
    return out

def resource(s,method,size,n,dt):
    if method=='QPE':
        settings=1; cu=2**size-1;un=0;depth=cu
    elif method=='ODMD':
        settings=2*size;cu=size*(size+1);un=0;depth=size
    else:
        settings=s.G*size**2;depth=size-1
        cu=s.G*sum(abs(i-j) for i in range(size) for j in range(size))
        un=s.G*sum(min(i,j) for i in range(size) for j in range(size))
    C=int(cu*n*R*len(s.terms));U=int(un*n*R*len(s.terms))
    return dict(size=size,n=n,dt=float(dt),r=R,settings=settings,shots=settings*n,
                max_slices=depth*R,max_time=float(dt*depth),C=C,U=U,W=2*C+U)

def odmd_batch(g,dt,size,n,shift,runs,rng):
    idx=np.arange(size//2)[:,None]+np.arange(size-size//2+1)
    tau=np.sqrt(2*np.count_nonzero(idx)/n)
    if rng is None:values=np.expand_dims(g,0)
    else:
        means=np.stack([g[1:].real,g[1:].imag])
        sample=2*rng.binomial(n,np.clip((means+1)/2,0,1),size=(runs,2,size))/n-1
        values=np.column_stack((np.ones(runs),sample[:,0]+1j*sample[:,1]))
    result=[]
    for block in np.array_split(values,(len(values)+63)//64):
        u,sv,vh=np.linalg.svd(block[:,idx],full_matrices=False)
        ranks=(sv>tau).sum(axis=1);out=np.full(len(block),np.nan)
        for rank in np.unique(ranks[ranks>0]):
            keep=ranks==rank
            small=(u[keep,:,:rank].conj().swapaxes(-1,-2)@block[keep][:,idx+1]@
                   vh[keep,:rank,:].conj().swapaxes(-1,-2))/sv[keep,None,:rank]
            poles=np.linalg.eigvals(small)
            out[keep]=np.where(np.min(abs(poles),axis=1)>1e-12,shift+np.min(-np.angle(poles)/dt,axis=1),np.nan)
        result.extend(out)
    return np.array(result)

def rng_at(s,method,stage,candidate=0):
    return np.random.default_rng([CONFIG['seed'],s.q,method,stage,candidate])

def qpe_run(s):
    dt=.25 if s.name=='6He' else min(.125,np.pi/(1.05*s.radius))
    bits=int(np.ceil(np.log2(2*np.pi/(dt*.05))));size=2**bits
    traj=trajectory(s,dt,size)
    bins=s.shift-2*np.pi*np.fft.fftfreq(size)/dt
    p=np.sum(abs(np.fft.fft(traj,axis=0)/size)**2,axis=1);p/=p.sum()
    phases=(s.ev[:,None]*(-dt)/(2*np.pi)-np.arange(size)/size+.5)%1-.5
    ideal=abs(s.v[s.sdet])**2@((np.sinc(size*phases)/np.sinc(phases))**2)
    low=p[bins<s.E0-.1].sum();high=p[bins<=s.E0+.1].sum()
    for n in range(1,401):
        k=int(np.ceil(n/4))
        rate=stats.binom.sf(k-1,n,high)-stats.binom.sf(k-1,n,low)
        if rate>=CONFIG['calibration_goal']:break
    else:raise ValueError('QPE calibration failed')
    samples=rng_at(s,0,99).choice(bins,(CONFIG['validation'],n),p=p)
    energies=np.partition(samples,k-1,axis=1)[:,k-1]
    def bias(prob):
        idx=np.argsort(bins)
        return float(bins[idx][np.searchsorted(np.cumsum(prob[idx]),.25)]-s.E0)
    return (dict(method='QPE',**resource(s,'QPE',bits,n,dt),k=k,bias=bias(p),exact_bias=bias(ideal),
            p_single=float(high-low),p_theory=float(rate)),
            energies,dict(bins=bins,p=p,ideal=ideal))

def qk_sample(s,model,m,n,runs,rng):
    h=np.zeros((runs,m,m),complex);S=np.zeros_like(h)
    S[:,np.arange(m),np.arange(m)]=1.
    for i,j,component,p,values in model:
        obs=(rng.multinomial(n,p,size=runs)@values)/n
        h[:,i,j]+=component*obs[:,0]
        if i!=j:S[:,i,j]+=component*obs[:,1]/s.G
    for matr in (h,S):matr+=np.triu(matr,1).conj().swapaxes(1,2)
    return ritz(h,S,np.sqrt(2*m*(m-1)/(s.G*n)),s.shift)

def group_data(s):
    terms = sorted(s.terms, key=lambda term: (-abs(term[1]), term[0]))
    labels, coeffs = zip(*terms)
    paulis = PauliList(labels)
    packed = np.column_stack([np.packbits(arr, axis=1, bitorder='little') for arr in (paulis.x, paulis.z)])
    supports, degree = core.paulis_to_support(packed, s.q)
    ids, mates = core.pairing_insertion(supports, degree, np.arange(len(terms)), s.q)
    core.verify_matching(supports, degree, ids, mates)
    pairs, _, _ = core.matching_readout(supports, degree, ids, mates)
    return np.asarray(coeffs), paulis, ids, {s.name+'__nqubits':s.q, s.name+'__pairing_greedy__pairs':pairs}


def diagonal_measurement(s,  paulis, coeffs, members, data, g, full):
    qc = core.setting_circuit(data, s.name, "pairing_greedy", g)
    diag = paulis[members].evolve(Clifford(qc), frame='s')
    assert not diag.x.any() and np.all(diag.phase % 2 == 0)
    masks = diag.z.astype(int) @ (1 << np.arange(s.q))
    signs = 1 - diag.phase.astype(int)  # Qiskit phases 0 or 2
    positions = np.arange(s.D)
    energy = sum(w*sign*s.parity[positions & mask] for w,sign,mask in zip(coeffs[members], signs, masks))
    rotated = np.column_stack([Statevector(v).evolve(qc).data for v in full.T])
    # Check both moments per group; the second moment contains cross terms of all
    # jointly measured Paulis. The matrix checks include real and imaginary coherences.
    phi = full[s.orbit]
    applied = np.zeros_like(phi)
    for member, weight in zip(paulis[members].to_labels(), coeffs[members]):
        x = s.mask(member,'XY')
        z = s.mask(member,'YZ')
        phase = (1j**member.count('Y')) * s.parity[(s.orbit ^ x) & z]
        applied += weight * phase[:,None] * phi[s.perms[x]]
    weighted = energy[:,None]*rotated
    np.testing.assert_allclose(rotated.conj().T @ weighted, phi.conj().T @ applied,
                               rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(weighted.conj().T @ weighted, applied.conj().T @ applied,
                               rtol=1e-10, atol=1e-10)
    compiled = transpile(qc, basis_gates=['cx','u'], optimization_level=1,
                         seed_transpiler=31)
    return rotated, energy, compiled.count_ops().get('cx',0)

def measurement_model(s,  traj):
    coeffs, paulis, ids, data = group_data(s)
    ng = int(ids.max()) + 1
    s.G = ng
    m = len(traj)
    full = np.zeros((s.D, m), complex)
    full[s.orbit] = traj.T
    model, cxs = [], []
    h_mean = np.zeros((m,m),complex)
    s_mean = np.eye(m,dtype=complex)
    for g in range(ng):
        members = np.flatnonzero(ids == g)
        assert len(members)
        rotated, energy, cx = diagonal_measurement(s,  paulis, coeffs, members, data, g, full)
        cxs.append(int(cx))
        levels, categories = np.unique(energy.round(12), return_inverse=True)
        def probabilities(v):
            return np.bincount(categories, weights=abs(v)**2, minlength=len(levels))
        for i in range(m):
            a = rotated[:,i]
            p = probabilities(a)
            mask = p > 0
            values = np.column_stack((levels[mask],np.zeros(mask.sum())))
            model.append((i,i,1.,p[mask]/p[mask].sum(),values))
            h_mean[i,i] += p @ levels
            for j in range(i+1,m):
                for phase, component in [(1.,1.),(-1j,1j)]:
                    b = phase*rotated[:,j]
                    plus = .25*probabilities(a+b)
                    minus = .25*probabilities(a-b)
                    probability = np.r_[plus,minus]
                    np.testing.assert_allclose(probability.sum(),1.,atol=1e-11)
                    mask = probability > 0
                    values = np.column_stack((np.r_[levels,-levels],np.repeat([1.,-1.],len(levels))))
                    model.append((i,j,component,probability[mask]/probability[mask].sum(),values[mask]))
                    h_mean[i,j] += component*((plus-minus) @ levels)
                    s_mean[i,j] += component*(plus-minus).sum()/ng
    h, gram = matrices(s,traj)
    h_residual = np.max(abs(h_mean+np.triu(h_mean,1).conj().T-h))
    s_residual = np.max(abs(s_mean+np.triu(s_mean,1).conj().T-gram))
    assert max(h_residual,s_residual) < 1e-9
    assert len(model) == ng*m**2
    return model, dict(groups=ng, max_abs_H_residual=float(h_residual),
                       max_abs_S_residual=float(s_residual), basis_cx_by_group=cxs,
                       assignment_in_coefficient_order=ids.tolist())

def replay(name, method):
    s = setup(name)
    row = next(r for r in PROTOCOLS[name]['selected'] if r['method'] == method)
    assert np.isclose(s.E0, PROTOCOLS[name]['E0'])
    size,dt,n = row['size'], row['dt'], row['n']
    if method == 'QPE':
        actual, energies, _ = qpe_run(s)
        assert (actual['size'],actual['n'],actual['dt']) == (size,n,dt)
        return energies
    if method == 'ODMD':
        signal = trajectory(s,dt,size+1)[:,s.psi.argmax()]
        return odmd_batch(signal,dt,size,n,s.shift,row['trials'],rng_at(s,1,99))
    if method == 'QKrylov':
        model, measured = measurement_model(s,trajectory(s,dt,size))
        assert measured['assignment_in_coefficient_order'] == row['assignment_in_coefficient_order']
        rng = np.random.default_rng([194221,s.q,2,3,n])
        return qk_sample(s,model,size,n,row['trials'],rng)
    raise ValueError(method)

if __name__ == '__main__':
    with np.load(DATA/'finite_shot_energies.npz') as archived:
        for name in CASES:
            for method in ('QPE','ODMD','QKrylov'):
                energies = replay(name,method)
                np.testing.assert_allclose(energies,archived[name+'_'+method],rtol=1e-10,atol=1e-9)
                print(name, method, len(energies), 'estimates reproduced',flush=True)
