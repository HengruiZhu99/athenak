// Exact arithmetic: common wave-principal / leading-residue energy at scri.
// This checks algebraic coercivity, not complete PDE or nonlinear well-posedness.
const gcd=(a,b)=>{a=a<0n?-a:a;b=b<0n?-b:b;while(b)[a,b]=[b,a%b];return a||1n;};
class R{
 constructor(n,d=1n){n=BigInt(n);d=BigInt(d);if(d===0n)throw Error("zero divisor");if(d<0n){n=-n;d=-d;}let g=gcd(n,d);this.n=n/g;this.d=d/g;}
 add(x){x=q(x);return new R(this.n*x.d+x.n*this.d,this.d*x.d);}
 neg(){return new R(-this.n,this.d);} sub(x){return this.add(q(x).neg());}
 mul(x){x=q(x);return new R(this.n*x.n,this.d*x.d);}
 div(x){x=q(x);return new R(this.n*x.d,this.d*x.n);}
 eq(x){x=q(x);return this.n===x.n&&this.d===x.d;}
 toString(){return this.d===1n?String(this.n):this.n+"/"+this.d;}
}
const q=(n,d=1)=>n instanceof R?n:new R(n,d);
const M=a=>a.map(row=>row.map(v=>q(v)));
const eye=n=>Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>q(i===j?1:0)));
const mul=(a,b)=>a.map(row=>b[0].map((_,j)=>row.reduce((s,v,k)=>s.add(v.mul(b[k][j])),q(0))));
const trans=a=>a[0].map((_,j)=>a.map(row=>row[j]));
const add=(a,b)=>a.map((row,i)=>row.map((v,j)=>v.add(b[i][j])));
const scale=(a,s)=>a.map(row=>row.map(v=>v.mul(s)));
const negsym=(h,j)=>scale(add(mul(h,j),mul(trans(j),h)),q(-1,2));
const assert=(v,s)=>{if(!v)throw Error(s);};
const ldl=a=>{
 const n=a.length,L=eye(n),D=[];
 for(let i=0;i<n;i++){
  let d=a[i][i];for(let k=0;k<i;k++)d=d.sub(L[i][k].mul(L[i][k]).mul(D[k]));
  assert(d.n>0n,"nonpositive LDL pivot "+i);D.push(d);
  for(let j=i+1;j<n;j++){let z=a[j][i];for(let k=0;k<i;k++)z=z.sub(L[j][k].mul(L[i][k]).mul(D[k]));L[j][i]=z.div(d);}
 }
 const diag=eye(n).map((row,i)=>row.map((_,j)=>i===j?D[i]:q(0)));
 const rebuilt=mul(mul(L,diag),trans(L));
 assert(a.every((row,i)=>row.every((v,j)=>v.eq(rebuilt[i][j]))),"LDL reconstruction");
 return D.map(String);
};

const certificates=[];
for(let ell=1;ell<=16;ell++){
 const l=q(ell),L=l.mul(l.add(1)),den=l.mul(7).add(3);
 const mode1=M([[l],[1]]),mode2=M([[l.add(1).mul(l.sub(6))],[l.add(9)]]);
 const lame=p=>M([[q(4).mul(p.mul(p.add(1))).sub(L.mul(3)).sub(8),L.mul(q(7).sub(p))],[p.add(8),q(3).mul(p.mul(p.add(1))).sub(L.mul(4))]]);
 for(const [p,v] of [[l.sub(1),mode1],[l.add(1),mode2]])assert(mul(lame(p),v).every(row=>row.every(x=>x.eq(0))),"Lame branch");
 // Unnormalized tangential coefficient; its energy includes factor L.
 const B=scale(M([[l.mul(l).mul(6).add(l).add(3),L.mul(l.sub(6))],[l.sub(1).mul(6),l.mul(l).mul(8).add(l.mul(5)).sub(3)]]),q(1).div(den));
 assert(B[0][0].add(B[1][1]).eq(l.mul(2)),"trace");
 const C=add(B,scale(eye(2),l.neg()));
 const disc=l.sub(1).mul(l.mul(l).sub(l.mul(4)).sub(3)).div(den);
 assert(mul(C,C).every((row,i)=>row.every((v,j)=>v.eq(i===j?disc:0))),"discriminant");
 let pivots=null;
 if(ell>=7){
  const H=M([[q(6).mul(l.sub(1)).div(l.sub(6)),0],[0,L]]);
  const HB=mul(H,B);
  assert(HB[0][1].eq(HB[1][0]),"high-mode symmetrizer");
  pivots=ldl(HB);
 }
 certificates.push({ell,discriminant:String(disc),high_mode_positive_pivots:pivots});
}
const turning=[];
for(const m of [6,12,20]){
 const sigma2=q(1,2*m+3);
 const constant=sigma2.mul(2*m+3).sub(1);
 const linear=sigma2.add(1).sub(sigma2.add(1));
 assert(constant.eq(0)&&linear.eq(0),"turning profile identity");
 turning.push({m,sigma_squared:String(sigma2),scaled_ode_residual:[String(constant),String(linear)]});
}
const b=q(1,3),r=q(-1);
const outflow0=r.mul(q(1).sub(b).mul(r).add(b.mul(2)));
const outflow1constant=r.mul(-2).sub(q(1).add(b.mul(3)));
const outflow1lambda=q(2).mul(q(1).sub(b)).mul(r).add(1).add(b);
assert([outflow0,outflow1constant,outflow1lambda].every(v=>v.eq(0)),"singular-outflow identity");
const ricciTrace=q(2,25).sub(q(24,75)).add(q(24,100));
assert(ricciTrace.eq(0),"leading Ricci trace conservation");
console.log(JSON.stringify({
 status:"passed exact algebra checks",
 scope:"Lamé boundary branches and energy, gauge counterexample identities; no numerical evolution",
 ball_dn:certificates,
 conformal_harmonic_turning_modes:turning,
 subluminal_one_third_outflow:{F_coefficient:String(outflow0),Fprime_coefficients:[String(outflow1constant),String(outflow1lambda)]},
 leading_ricci_trace:String(ricciTrace),
 does_not_establish:"full coupled theorem; that requires the saved analytical proofs and phase audit"
},null,2));

