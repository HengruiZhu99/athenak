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
const results=[];
for(const y of [q(-1),q(-99,100)]){
 const x=q(4), c=q(2).sub(x),d=x.mul(q(2).add(y)),dk=x.mul(3).mul(q(1).add(y));
 // scalar order: pi,Wn,Qn,Znn,Z11+Z22,tau,zeta_n
 const J=M([
 [-4,-2,2,c,c,dk.mul(2),c],
 [-1,-2,q(3).add(x.mul(y)),q(4,3),-2,x.mul(y),-8],
 [-1,0,q(1).sub(d),0,0,d.neg(),0],
 [0,0,0,c,0,0,c],
 [0,0,0,0,c,0,0],
 [1,0,0,0,0,q(2).sub(d),0],
 [0,1,0,0,0,0,q(3).sub(x)]
 ]);
 const H=eye(7);H[4][4]=q(1,2);H[5][5]=q(4);H[5][6]=H[6][5]=q(2);H[6][6]=q(8);
 const K=negsym(H,J),scalar=ldl(K);
 // each transverse order: WA,QA,Z_nA,Z_An,zeta_A
 const JV=M([
 [-2,q(3).add(x.mul(y)),0,q(10,3),-6],
 [0,q(1).sub(d),0,0,0],
 [0,0,c,0,c],
 [0,0,0,c,0],
 [1,0,0,0,q(3).sub(x)]
 ]);
 const HV=eye(5);HV[4][4]=q(8);
 const vector=ldl(negsym(HV,JV));
 results.push({x:String(x),y:String(y),scalar_ldl_pivots:scalar,each_transverse_ldl_pivots:vector,remaining_tensor_pivots:["2","2","2"],field_count:"7+2*5+3=20"});
}
// Independent checks of the compatible scalar radial pencil at x=4,y=-99/100.
const L=M([[q(302,25),8],[q(49,25),q(8,3)]]);
const K=M([[q(-596,25),-4],[q(-102,25),q(-34,3)]]);
const F=M([[q(58,31),q(-150,31)],[q(24,155),q(242,31)]]);
assert(add(mul(L,F),K).every(row=>row.every(v=>v.eq(0))),"L F + K");
const F0=add(F,scale(eye(2),q(-2)));
const modal=M([[6,5],[2,75]]), S=mul(trans(modal),modal), mu=q(176,31);
const modes=mul(modal,F0);
assert(modes[0].every(v=>v.eq(0)),"outgoing integer root");
assert(modes[1].every((v,j)=>v.eq(mu.mul(modal[1][j]))),"fractional root");
ldl(S);
const SF0=mul(S,F0), desired=M([[4,150],[150,5625]]);
assert(SF0.every((row,i)=>row.every((v,j)=>v.eq(desired[i][j].mul(mu)))),"radial weighted symmetrizer");
const det2=a=>a[0][0].mul(a[1][1]).sub(a[0][1].mul(a[1][0]));
for(const s of [q(0),q(1),q(2),q(238,31),q(3),q(15,2)]){
 const det=det2(add(scale(L,s),K));
 assert(det.eq(q(8,15).mul(s.sub(2)).mul(q(31).mul(s).sub(238))),"pencil determinant at "+s);
}
console.log(JSON.stringify({
 status:"passed exact common-energy block-coercivity checks",
 wave_target_metric:"I_4 repeated in wave velocity and all spatial-gradient blocks",
 value_metric:"4 tau^2+4 tau zeta_n+8 |zeta|^2",
 parameters_and_certificates:results,
 compatible_indicial_audit:{roots:["2","238/31"],natural_fractional_exponent:"176/31",radial_symmetrizer:S.map(row=>row.map(String)),modal_rows:modal.map(row=>row.map(String)),pencil_determinant:"(8/15)(s-2)(31s-238)",all_exact_checks:"passed"},
 outside_this_algebra_audit:["exact bounded-remainder firstorder PDE reduction","propagation of natural asymptotic jets","full geometric/lapse/elliptic boundary estimate","nonlinear well-posedness"]
},null,2));

