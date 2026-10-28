"""Exact elimination of normal shift, with no bulk-constraint projection."""
import numpy as np

def normal_shift(alpha,chi,g):
    if np.any(np.asarray(alpha)<=0) or np.any(np.asarray(chi)<=0) or np.any(np.asarray(g)<=0):
        raise FloatingPointError('Null boundary requires positive finite geometry')
    value=-np.asarray(alpha)*np.sqrt(np.asarray(chi)/np.asarray(g))
    if not np.all(np.isfinite(value)):raise FloatingPointError('Nonfinite null shift')
    return value

def normal_rate(alpha,chi,g,alpha_t,chi_t,g_t):
    beta=normal_shift(alpha,chi,g)
    return beta*(alpha_t/alpha+.5*chi_t/chi-.5*g_t/g)

def vector_shift(alpha,metric,covector,tangential_trial):
    """Fixed OUTWARD covector (-Omega_i for Omega>0 inside), Penrose metric.

    Tangential coordinates remain free. The caller supplies orientation; using
    +Omega_i instead would select the opposite branch.
    """
    inverse=np.linalg.inv(metric);raised=inverse@covector
    norm=np.sqrt(covector@raised);s=covector/norm;sup=raised/norm
    tangent=tangential_trial-sup*(s@tangential_trial)
    return tangent-alpha*sup

def vector_rate_identity(alpha,metric,covector,beta,alpha_t,metric_t):
    """Required s_i beta_t^i, including evolving unit normal at fixed Omega_i."""
    inverse=np.linalg.inv(metric);raised=inverse@covector
    norm=np.sqrt(covector@raised);s=covector/norm;sup=raised/norm
    st=.5*s*(sup@metric_t@sup)
    return -alpha_t-beta@st
