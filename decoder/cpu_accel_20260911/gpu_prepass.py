"""GPU implementation of the CPU relay's deterministic first leg.

This is a screening stage, not a replacement decoder: unresolved shots must
be sent to the original full CPU cascade in original chunk order. Successful
first legs consume no relay RNG. Keep floating-point prefix/suffix sums in the
CPU order, including signed zeros. No torch scatter-add or atomic float sums.
"""
import numpy as np
import scipy.sparse as sp
import torch
import triton as tr
import triton.language as tl
from variable_kernels import VARIABLE_KERNELS
from variable_streamed import variable_streamed
from screen_contract import (binary_array,model_fingerprint,detectors_fingerprint,
                             make_screen_protocol,observables_fingerprint)


@tr.jit
def _check(Q, CV, PTR, ROWS, SYN, DONE, NB,
           NR:tl.constexpr, K:tl.constexpr, BS:tl.constexpr):
    ri = tl.program_id(0)
    row = tl.load(ROWS+ri)
    b = tl.program_id(1)*BS+tl.arange(0,BS)
    live = (b<NB) & (tl.load(DONE+b,b<NB,other=1)==0)
    start=tl.load(PTR+row); end=tl.load(PTR+row+1)
    edge=start+tl.arange(0,K)
    valid=(edge[:,None]<end)&live[None,:]
    q=tl.load(Q+edge[:,None]*NB+b[None,:],valid,other=0.)
    a=tl.where(valid&(tl.abs(q)<3.4028234663852886e38),tl.abs(q),3.4028234663852886e38)
    m1=tl.min(a,0)
    count=tl.sum(((a==m1[None,:])&valid).to(tl.int32),0)
    m2=tl.min(tl.where(a==m1[None,:],3.4028234663852886e38,a),0)
    m2=tl.where(count>1,m1,m2)
    negative=(q.to(tl.int32,bitcast=True)<0)&valid
    parity=tl.sum(negative.to(tl.int32),0)&1
    syn=tl.load(SYN+row*NB+b,live,other=0)
    sign=(parity[None,:]^negative.to(tl.int32)^syn[None,:])!=0
    mag=tl.where(a==m1[None,:],m2[None,:],m1[None,:])
    tl.store(CV+edge[:,None]*NB+b[None,:],tl.where(sign,-mag,mag),valid)


@tr.jit
def _variable(Q,CV,POST,PRIOR,PTR,MAP,VARS,DONE,HARD,
              NV:tl.constexpr,NB,D:tl.constexpr,
              BS:tl.constexpr,VS:tl.constexpr,G:tl.constexpr,IG:tl.constexpr):
    vi=tl.program_id(0)*VS+tl.arange(0,VS)
    v=tl.load(VARS+vi,vi<NV,other=0)
    b=tl.program_id(1)*BS+tl.arange(0,BS)
    live=(vi[:,None]<NV)&(b[None,:]<NB)&(tl.load(DONE+b,b<NB,other=1)[None,:]==0)
    start=tl.load(PTR+v);end=tl.load(PTR+v+1)
    prior=tl.load(PRIOR+v)[:,None]
    old=tl.load(POST+v[:,None]*NB+b[None,:],live,other=0.)
    acc=prior*IG+old*G
    acc=tl.where(prior==3.4028234663852886e38,prior,acc)
    prefixes=()
    values=()
    for j in tl.static_range(D):
        idx=start+j
        m=live&(idx[:,None]<end[:,None])
        edge=tl.load(MAP+idx,(vi<NV)&(idx<end),other=0)
        cv=tl.load(CV+edge[:,None]*NB+b[None,:],m,other=0.)
        prefixes+=(acc,)
        values+=(cv,)
        acc=tl.where(m,acc+cv,acc)
    tl.store(POST+v[:,None]*NB+b[None,:],acc,live)
    tl.store(HARD+v[:,None]*NB+b[None,:],acc<=0.,live)
    suffix=tl.full((VS,BS),0.,tl.float32)
    for jj in tl.static_range(D):
        idx=start+D-1-jj
        m=live&(idx[:,None]<end[:,None])
        edge=tl.load(MAP+idx,(vi<NV)&(idx<end),other=0)
        cv=values[D-1-jj]
        prefix=prefixes[D-1-jj]
        tl.store(Q+edge[:,None]*NB+b[None,:],prefix+suffix,m)
        suffix=tl.where(m,suffix+cv,suffix)


@tr.jit
def _syndrome(HARD,PTR,IND,ROWS,SYN,BAD,DONE,
              NB,K:tl.constexpr,BS:tl.constexpr):
    row=tl.load(ROWS+tl.program_id(0))
    b=tl.program_id(1)*BS+tl.arange(0,BS)
    live=(b<NB)&(tl.load(DONE+b,b<NB,other=1)==0)
    start=tl.load(PTR+row);end=tl.load(PTR+row+1)
    edge=start+tl.arange(0,K)
    var=tl.load(IND+edge,edge<end,other=0)
    h=tl.load(HARD+var[:,None]*NB+b[None,:],(edge[:,None]<end)&live[None,:],other=0)
    parity=tl.sum(h.to(tl.int32),0)&1
    syn=tl.load(SYN+row*NB+b,live,other=0)
    tl.store(BAD+row*NB+b,(parity!=syn)&live,b<NB)


class ExactFirstLeg:
    def __init__(self,H,pri,gamma0=.1,device='cuda',L=None,
                 check_warps=4,variable_warps=8,syndrome_warps=4):
        from relay_bp import RelayDecoderF32
        self.dev=device
        if any(w not in (4,8,16) for w in (check_warps,variable_warps,syndrome_warps)):
            raise ValueError('GPU launch warps must be 4, 8, or 16')
        self.check_warps=check_warps
        self.variable_warps=variable_warps
        self.syndrome_warps=syndrome_warps
        self.model_id=model_fingerprint(H,pri)
        self.gamma0=gamma0
        make_screen_protocol(self.model_id,pre_iter=0,gamma0=gamma0)
        H=sp.csr_matrix(H,dtype=np.uint8)
        if not H.has_canonical_format or not np.all(H.data==1):
            raise ValueError('Require a canonical binary CSR matrix')
        self.H=H;self.C,self.N=H.shape
        self.gamma=np.float32(gamma0)
        self.igamma=np.float32(1)-self.gamma
        # Obtain the exact Rust/libm -> F32 prior values (no numpy log drift).
        d=RelayDecoderF32(check_matrix=H,error_priors=np.asarray(pri,np.float64),
                          gamma0=gamma0,pre_iter=0,num_sets=0)
        p=d.decode_detailed(np.zeros(self.C,np.uint8)).posterior_ratios
        self.prior=torch.tensor(np.asarray(p,np.float32),device=device)
        def gpu(x):return torch.as_tensor(np.asarray(x,dtype=np.int32),device=device)
        self.ptr=gpu(H.indptr);self.ind=gpu(H.indices)
        numbered=sp.csr_matrix((np.arange(H.nnz,dtype=np.int32)+1,H.indices,H.indptr),shape=H.shape).tocsc()
        self.vptr=gpu(numbered.indptr);self.mapping=gpu(numbered.data-1)
        def groups(degrees):
            powers=np.array([tr.next_power_of_2(max(1,int(n))) for n in degrees])
            return [(int(d),gpu(np.flatnonzero(powers==d))) for d in np.unique(powers)]
        self.rows=groups(np.diff(H.indptr));self.variables=groups(np.diff(numbered.indptr))
        self.logical=None
        if L is not None:
            self.logical_id=observables_fingerprint(L)
            L=sp.csr_matrix(L,dtype=np.uint8)
            if L.shape[1]!=self.N:raise ValueError('Logical matrix column mismatch')
            segments=[(row,start,min(start+256,L.indptr[row+1]))
                      for row in range(L.shape[0])
                      for start in range(L.indptr[row],L.indptr[row+1],256)]
            parts=np.asarray(segments,dtype=np.int32).reshape(-1,3)
            self.logical=(L.shape[0],gpu(L.indices),gpu(parts[:,0]),gpu(parts[:,1]),gpu(parts[:,2]))

    def decode(self,det,pre_iter=200,check_every=8,return_details=True):
        protocol=make_screen_protocol(self.model_id,pre_iter=pre_iter,gamma0=self.gamma0)
        if not isinstance(check_every,int) or check_every<1:
            raise ValueError('check_every must be a positive integer')
        det=binary_array(det,'detectors')
        if det.ndim!=2 or det.shape[1]!=self.C or not np.isin(det,[0,1]).all():
            raise ValueError('Invalid detectors')
        B=len(det)
        original_B=B
        if not return_details and self.logical is None:
            raise ValueError('Logical matrix is required for compact output')
        if not B:raise ValueError('Empty batch')
        syn=torch.tensor(det.T.copy(),device=self.dev)
        q=self.prior[self.ind.long(),None].expand(-1,B).contiguous()
        cv=torch.empty_like(q)
        # With B=1, contiguous() alone can alias and overwrite the prior.
        post=self.prior[:,None].expand(-1,B).clone(memory_format=torch.contiguous_format)
        hard=torch.zeros((self.N,B),device=self.dev,dtype=torch.uint8)
        done=torch.zeros(B,device=self.dev,dtype=torch.bool)
        bad=torch.empty((self.C,B),device=self.dev,dtype=torch.bool)
        iterations=torch.zeros(B,device=self.dev,dtype=torch.int32)
        unsafe=torch.zeros(B,device=self.dev,dtype=torch.int32)
        ids=torch.arange(B,device=self.dev)
        out_hard=torch.zeros_like(hard)
        out_post=post.clone() if return_details else None
        out_success=torch.zeros(B,device=self.dev,dtype=torch.bool)
        out_unsafe=torch.zeros(B,device=self.dev,dtype=torch.bool)
        out_iterations=torch.full((B,),pre_iter,device=self.dev,dtype=torch.int32)
        bs=16
        for it in range(pre_iter):
            for k,rows in self.rows:
                _check[(len(rows),tr.cdiv(B,bs))](q,cv,self.ptr,rows,syn,done,B,len(rows),k,bs,
                                               num_warps=self.check_warps,enable_fp_fusion=False)
            for degree,variables in self.variables:
                vs=32 if degree<=16 else 16 if degree<=32 else 8
                variable_streamed[(tr.cdiv(len(variables),vs),tr.cdiv(B,bs))](q,cv,post,self.prior,
                    self.vptr,self.mapping,variables,done,hard,unsafe,len(variables),B,degree,bs,vs,
                    float(self.gamma),float(self.igamma),num_warps=self.variable_warps,enable_fp_fusion=False)
            for k,rows in self.rows:
                _syndrome[(len(rows),tr.cdiv(B,bs))](hard,self.ptr,self.ind,rows,syn,bad,done,B,k,bs,
                                                  num_warps=self.syndrome_warps,enable_fp_fusion=False)
            ok=~bad.any(dim=0)
            iterations.masked_fill_(ok&~done,it+1)
            done|=ok
            if (it+1)%check_every==0 or it+1==pre_iter:
                good=done.nonzero().flatten()
                if good.numel():
                    dst=ids.index_select(0,good)
                    out_hard.index_copy_(1,dst,hard.index_select(1,good))
                    if return_details:out_post.index_copy_(1,dst,post.index_select(1,good))
                    out_iterations.index_copy_(0,dst,iterations.index_select(0,good))
                    out_success[dst]=unsafe.index_select(0,good)==0
                    out_unsafe[dst]=unsafe.index_select(0,good)!=0
                    remain=(~done).nonzero().flatten()
                    ids=ids.index_select(0,remain)
                    if not ids.numel():break
                    # Only unresolved columns continue. Results were frozen
                    # on their original first convergence iteration above.
                    q=q.index_select(1,remain).contiguous()
                    cv=cv.index_select(1,remain).contiguous()
                    post=post.index_select(1,remain).contiguous()
                    hard=hard.index_select(1,remain).contiguous()
                    syn=syn.index_select(1,remain).contiguous()
                    B=int(ids.numel())
                    done=torch.zeros(B,device=self.dev,dtype=torch.bool)
                    bad=torch.empty((self.C,B),device=self.dev,dtype=torch.bool)
                    iterations=torch.zeros(B,device=self.dev,dtype=torch.int32)
                    unsafe=unsafe.index_select(0,remain).contiguous()
        if ids.numel():
            out_hard.index_copy_(1,ids,hard)
            if return_details:out_post.index_copy_(1,ids,post)
            out_unsafe[ids]=unsafe!=0
        result=dict(success=out_success.cpu().numpy(),iterations=out_iterations.cpu().numpy(),
                    unsafe_mask=out_unsafe.cpu().numpy(),protocol=protocol,
                    detectors_fingerprint=detectors_fingerprint(det))
        if return_details:
            result.update(decoding=out_hard.T.cpu().numpy(),posterior=out_post.T.cpu().numpy())
        if self.logical is not None:
            count,indices,rows,starts,ends=self.logical
            predictions=torch.zeros((count,original_B),device=self.dev,dtype=torch.int32)
            if len(rows):
                _logical[(len(rows),tr.cdiv(original_B,16))](out_hard,indices,rows,starts,ends,
                    predictions,original_B,256,16,num_warps=4)
            result.update(observables=predictions.T.to(torch.uint8).cpu().numpy(),
                          observables_fingerprint=self.logical_id)
        return result


@tr.jit
def _logical(HARD,IND,ROWS,STARTS,ENDS,OUT,NB,K:tl.constexpr,BS:tl.constexpr):
    part=tl.program_id(0)
    row=tl.load(ROWS+part);start=tl.load(STARTS+part);end=tl.load(ENDS+part)
    edge=start+tl.arange(0,K)
    var=tl.load(IND+edge,edge<end,other=0)
    b=tl.program_id(1)*BS+tl.arange(0,BS)
    hard=tl.load(HARD+var[:,None]*NB+b[None,:],(edge[:,None]<end)&(b[None,:]<NB),other=0)
    parity=tl.sum(hard.to(tl.int32),0)&1
    tl.atomic_xor(OUT+row*NB+b,parity,mask=b<NB,sem='relaxed')
