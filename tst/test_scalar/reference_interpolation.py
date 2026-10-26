"""High-order local value interpolation for smooth independent scalar profiles."""
import numpy as np

def lagrange6(x,values,query):
 x=np.asarray(x);values=np.asarray(values);query=np.asarray(query);flat=query.ravel();result=np.empty((len(flat),*values.shape[1:]),dtype=np.result_type(values,float))
 if len(x)<6 or len(values)!=len(x) or np.any(np.diff(x)<=0):raise ValueError('Six distinct increasing profile nodes required')
 if flat.size and (flat.min()<x[0]-1e-10 or flat.max()>x[-1]+1e-10):raise ValueError('Independent profile does not cover the query domain')
 for start in range(0,len(flat),65536):
  sl=slice(start,start+65536);q=flat[sl];index=np.clip(np.searchsorted(x,q,side='right')-3,0,len(x)-6);anchor=values[index+2];v=anchor.copy()
  for a in range(6):
   weight=np.ones(len(q))
   for b in range(6):
    if a!=b:weight*=(q-x[index+b])/(x[index+a]-x[index+b])
   v+=weight.reshape((-1,)+(1,)*(values.ndim-1))*(values[index+a]-anchor)
  result[sl]=v
 return result.reshape((*query.shape,*values.shape[1:]))
