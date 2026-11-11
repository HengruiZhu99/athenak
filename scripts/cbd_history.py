"""Explicit schema adapter for circumbinary histories (no gap filling or screening).

Use native athena_read.hst_segments to retain schema, metadata and restart boundaries.
Pass one segment's named columns to inward_currents. Historical frame interpretation
must be supplied explicitly. Cached manuscript numerical results are never rewritten.
"""
import re
import numpy as np


def frame_from_metadata(metadata, historical_frame=None):
    match = re.search(r'(?:^|\s)frame=(inertial|corotating)(?:\s|$)', metadata)
    if match:
        frame = match.group(1)
        if historical_frame is not None and historical_frame != frame:
            raise ValueError('Explicit historical interpretation conflicts with frame metadata')
        return frame
    if historical_frame not in ('inertial', 'corotating'):
        raise ValueError('Historical data requires an explicit inertial/corotating interpretation')
    return historical_frame


def inward_currents(columns, surface, *, mode, metadata='', historical_frame=None,
                    reynolds=None):
    """Return inward mass, Eulerian/coordinate energy, momentum and barycentric J.

    mode='moving' consumes already corrected schema-2 channels. mode='legacy_fixed'
    preserves legacy fixed-surface terms. mode='legacy_plus_reynolds' requires an
    explicitly supplied correction dictionary with matching times/frame/worldtube.
    No choice of mode changes the magnetic normalization or selects a different surface.
    """
    frame = frame_from_metadata(metadata, historical_frame)
    get = lambda name: np.asarray(columns[f'{name}_{surface}'])
    if mode == 'moving':
        if reynolds is not None:
            raise ValueError('Moving channels already contain the Reynolds correction')
        if f'M0_net_in_moving_{surface}' not in columns:
            raise ValueError('This segment has no corrected moving-surface channels')
        vector = lambda prefix, part: np.stack([get(f'{prefix}{a}_{part}_in_moving')
                                               for a in 'xyz'], axis=-1)
        out = dict(mdot=get('M0_net_in_moving'), mdot_in=get('M0_in_moving'),
                   mdot_out=get('M0_out_moving'), edot_f=get('Eeu_f_in_moving'),
                   edot_em=get('Eeu_em_in_moving'),
                   et_f=get('Et_f_in_moving'), et_em=get('Et_em_in_moving'),
                   pin=vector('P','f')+vector('P','em'),
                   jin=vector('J','f')+vector('J','em'))
        valid = np.isfinite(get('valid')) & (get('valid') == 1)
        for name in out:
            mask = valid[...,None] if out[name].ndim > valid.ndim else valid
            out[name] = np.where(mask, out[name], np.nan)
    elif mode in ('legacy_fixed', 'legacy_plus_reynolds'):
        vector = lambda prefix: -np.stack([get(f'{prefix}{a}dot_f')+get(f'{prefix}{a}dot_em')
                                          for a in 'xyz'],axis=-1)
        out = dict(mdot=get('mdot'),edot_f=get('edot_f'),edot_em=get('edot_em'),
                   pin=vector('p'),jin=vector('l'))
        if mode == 'legacy_plus_reynolds':
            if reynolds is None:
                raise ValueError('Legacy moving interpretation needs an explicit Reynolds correction')
            for name in ('mdot','edot_f','edot_em','pin','jin'):
                out[name] = out[name] + reynolds[name]
        elif reynolds is not None:
            raise ValueError('Choose legacy_plus_reynolds to apply a correction explicitly')
    else:
        raise ValueError(f'Unknown transport interpretation {mode}')
    out['edot'] = out['edot_f']+out['edot_em']
    out['phiB_HL'] = get('phiB')
    out['area'] = get('area')
    out['frame'] = frame
    out['surface'] = surface
    out['interpretation'] = mode
    return out


def intrinsic_flux(j_global, position, p_inward):
    """Finite-radius coordinate flux about position; not measured BH spin growth."""
    return np.asarray(j_global)-np.cross(position,p_inward)


def magnetic_normalization(phi_HL, reference_rate, hole_mass, *, convention,
                           rate_provenance):
    """Keep explicit per-hole or fixed-total moving-5M transfer-rate denominators."""
    if convention not in ('per_hole', 'fixed_total') or not rate_provenance:
        raise ValueError('Specify normalization convention and transfer-rate provenance')
    if 'Kerr' in rate_provenance or 'horizon' in rate_provenance.lower():
        raise ValueError('Modeled-horizon rates cannot replace manuscript transfer-rate denominators')
    rate=np.asarray(reference_rate)
    return np.sqrt(4*np.pi)*np.asarray(phi_HL)/(hole_mass*np.sqrt(np.where(rate>0,rate,np.nan)))
