// Exact, dependency-free arithmetic audit for the paper-like constraint branch.
// Run with node indicial-audit.mjs. This verifies finite algebra, not PDE estimates.
const gcd=(a,b)=>{a=a<0n?-a:a;b=b<0n?-b:b;while(b){[a,b]=[b,a%b];}return a||1n;};
class R {
  constructor(n,d=1n){n=BigInt(n);d=BigInt(d);if(d===0n)throw Error("zero divisor");if(d<0n){n=-n;d=-d;}const g=gcd(n,d);this.n=n/g;this.d=d/g;}
  add(b){b=rat(b);return new R(this.n*b.d+b.n*this.d,this.d*b.d);}
  neg(){return new R(-this.n,this.d);}
  sub(b){return this.add(rat(b).neg());}
  mul(b){b=rat(b);return new R(this.n*b.n,this.d*b.d);}
  div(b){b=rat(b);return new R(this.n*b.d,this.d*b.n);}
  eq(b){b=rat(b);return this.n===b.n&&this.d===b.d;}
  toString(){return this.d===1n?String(this.n):this.n+"/"+this.d;}
}
const rat=x=>x instanceof R?x:new R(x), q=(n,d=1)=>new R(n,d);
const M=a=>a.map(row=>row.map(rat));
const tr=a=>a.reduce((s,row,i)=>s.add(row[i]),q(0));
const mul=(a,b)=>a.map(row=>b[0].map((_,j)=>row.reduce((s,v,k)=>s.add(v.mul(b[k][j])),q(0))));
const trans=a=>a[0].map((_,j)=>a.map(row=>row[j]));
const eye=n=>Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>q(i===j?1:0)));
const sub=(a,b)=>a.map((row,i)=>row.map((v,j)=>v.sub(b[i][j])));
const scale=(a,s)=>a.map(row=>row.map(v=>v.mul(s)));
const det=a=>{if(a.length===1)return a[0][0];return a[0].reduce((sum,v,j)=>sum.add(v.mul(det(a.slice(1).map(row=>row.filter((_,k)=>k!==j)))).mul(j%2?-1:1)),q(0));};
const zero=a=>a.every(row=>row.every(v=>v.eq(0)));
const inv2=a=>scale([[a[1][1],a[0][1].neg()],[a[1][0].neg(),a[0][0]]],q(1).div(det(a)));
const chars=a=>{const powers=[];let p=a;for(let i=1;i<=a.length;i++){powers.push(tr(p));p=mul(p,a);}const c=[q(1)];for(let k=1;k<=a.length;k++){let z=q(0);for(let i=1;i<=k;i++)z=z.add(c[k-i].mul(powers[i-1]));c.push(z.neg().div(k));}return c;};
const assert=(v,label)=>{if(!v)throw Error(label);};
const pencil=s=>{
 s=rat(s);
 return [
 [q(1,2),q(0),s.sub(9),s.add(1).neg()],
 [q(0),q(1),s.add(2).neg(),s.sub(7)],
 [s.sub(4),s.mul(2).sub(4),q(12),s.add(1).mul(8)],
 [s.div(2),s.sub(2),s.add(1).mul(-2),s.mul(4).add(28).div(-3)]
 ];
};
const samples=[q(0),q(1),q(2),q(3),q(8),q(133,16)];
const factor=s=>s.sub(2).mul(s.mul(16).sub(133));
for(const s of samples)assert(det(pencil(s)).eq(factor(s).mul(q(8,3))),"four-field indicial determinant");
assert(zero(mul(pencil(q(2)),M([[22],[q(32,3)],[1],[q(4,3)]]))),"root-two kernel");
const L=M([[26,20],[7,q(26,3)]]),K=M([[-68,-28],[-6,q(-70,3)]]);
for(const s of samples){const P=L.map((row,i)=>row.map((v,j)=>v.mul(s).add(K[i][j])));assert(det(P).eq(factor(s).mul(q(16,3))),"reduced-pencil determinant sign");}
const F=scale(mul(inv2(L),K),-1);
assert(zero(sub(F,M([[q(11,2),q(-21,8)],[q(-15,4),q(77,16)]]))),"Fuchsian matrix");
const F0=sub(F,scale(eye(2),2)),S=M([[10,0],[0,7]]),SF=mul(S,F0);
assert(zero(sub(SF,trans(SF)))&&det(SF).eq(0),"radial symmetrizer");
const modal=M([[q(45,101),q(42,101)],[q(56,101),q(-42,101)]]);
assert(zero(sub(mul(modal,F0),mul(M([[0,0],[0,q(101,16)]]),modal))),"radial modal decomposition");
const J=M([[-4,-4,12,8],[0,-2,-2,q(-28,3)],[q(1,2),0,-9,-1],[0,1,-2,-7]]);
const c=chars(J), expected=[q(1),q(22),q(505,3),q(1690,3),q(2128,3)];
assert(c.every((v,i)=>v.eq(expected[i])),"value-residue characteristic polynomial");
const r1=c[1].mul(c[2]).sub(c[3]);
const r2=c[1].mul(c[2]).mul(c[3]).sub(c[3].mul(c[3])).sub(c[1].mul(c[1]).mul(c[4]));
assert(r1.eq(3140)&&r2.eq(q(4276648,3)),"Routh-Hurwitz minors");
const G=M([[q(-13,2),-5],[q(-7,2),q(-13,3)]]),HG=mul(M([[7,0],[0,10]]),G);
assert(zero(sub(HG,trans(HG)))&&det(HG).eq(q(2240,3))&&HG[0][0].n<0,"glancing dissipation");
const report={
 status:"passed exact finite-dimensional checks",
 parameters:{a_kappa1:"10",kappa2:"-9/10",theta_source:"Appendix-C correction"},
 scalar_indicial_det:"(8/3)(s-2)(16s-133)",
 reduced_indicial_det:"(16/3)(s-2)(16s-133)",
 roots:["2","133/16"],
 natural_remainder_exponent:"101/16",
 integer_recursion_nonresonance_orders:[1,2,3,4,5,6],
 scalar_kernel_at_2:["22","32/3","1","4/3"],
 value_residue_charpoly:c.map(String),
 routh_minors:[String(r1),String(r2)],
 glancing_weighted_determinant:String(det(HG)),
 not_proved:["invariant PDE asymptotics","time/angular energy estimate","elliptic trace estimate in variable coefficients","full coupled well-posedness","nonlinear well-posedness"]
};
console.log(JSON.stringify(report,null,2));

