#!/usr/bin/env python3
"""Freeze/audit historical spherical snapshots and export physical Cartesian jets.

Historical coefficient factors are used ONLY to decode their recorded storage
format. Exported physical fields have no imposed radial exponent. A spherical
snapshot is initialization/comparison data, never spherical enforcement of a
subsequent 3D evolution. Positive-node interpolation excludes the coefficient
endpoint and refuses extrapolation; no floor or parity condition is applied.
"""
import argparse
import ast
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np

RADIAL_FIELDS = ['chi', 'alpha', 'A_rr', 'Khat', 'Theta', 'Lambda_r', 'beta_r', 'B_r']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights(nodes, target):
    """Local polynomial weights in scaled coordinates, orders0/1/2."""
    scale = float(max(abs(nodes-target)))
    if not scale > 0:
        raise ValueError('Degenerate interpolation stencil')
    x = (nodes-target)/scale
    matrix = np.array([x**p for p in range(len(x))])
    rhs = np.zeros((len(x), 3));rhs[0, 0] = 1;rhs[1, 1] = 1/scale;rhs[2, 2] = 2/scale**2
    return np.linalg.solve(matrix, rhs).T


class RadialSnapshot:
    def __init__(self, path):
        self.path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            self.r = z['r'].copy();self.q = z['q'].copy();self.q_physical = z['q_physical'].copy()
            self.B = z['B'].copy();self.beta = z['beta'].copy();self.diagnostic = z['diagnostic'].copy()
            self.time = float(z['t']);self.step = int(z['step'])
        if self.q.shape != (len(self.r), 8) or not np.all(np.diff(self.r)>0) or self.r[0]!=0 or self.r[-1]!=1:
            raise ValueError('Unexpected full-domain historical spherical layout')
        if not np.all(np.isfinite(self.q)):
            raise ValueError('Nonfinite stored historical coefficients')
        if np.max(abs(self.q_physical[:, 1])) != 0:
            raise ValueError('Exporter currently requires the recorded g=1 spherical family; metric deviations are nonzero')
        r = self.r;omega = (1-r*r)/2;Aref = (1+r*r)/2
        weight = r*r/(r*r+omega**4)
        # Decode ONLY positive nodes. The r=0 slot stores asymptotic coefficients,
        # not an ordinary physical value; never exponentiate unused endpoint data.
        chi = np.zeros_like(r);alpha = np.zeros_like(r)
        chi[1:] = r[1:]**4/(r[1:]**4+omega[1:]**4)*np.exp(self.q[1:, 0])
        alpha[1:] = Aref[1:]*weight[1:]*np.exp(self.q[1:, 6])
        self.values = np.column_stack([chi, alpha, weight*self.q[:, 2],
            weight*(self.q[:, 3]-3), weight*self.q[:, 4], self.q[:, 5], self.beta, self.B])
        if not np.all(np.isfinite(self.values)) or not np.all(self.values[1:, :2]>0):
            raise ValueError('Nonpositive/nonfinite decoded physical snapshot; no floor applied')
        self._cache = {}

    def sample(self, radius, width=7):
        radius = float(radius)
        if width not in (5, 7, 9) or len(self.r)-1 < width:
            raise ValueError('Interpolation width must be5,7,9 with enough positive nodes')
        if not self.r[1] <= radius <= self.r[-1]:
            raise ValueError(f'radius={radius} outside recorded positive interval [{self.r[1]},1]; endpoint limits/extrapolation not inferred')
        key = (radius, width)
        if key in self._cache:
            return tuple(a.copy() for a in self._cache[key])
        first = int(np.clip(np.searchsorted(self.r, radius)-width//2, 1, len(self.r)-width))
        indices = np.arange(first, first+width);r = self.r[indices]
        W = weights(r, radius);raw = W@self.values[indices]
        value, derivative, second = raw[0].copy(), raw[1].copy(), raw[2].copy()
        # Interpolate positive fields in log-radius/log-field coordinates. This
        # preserves positivity without choosing an exponent; a pure noninteger
        # power is a linear function here. Derivatives include the full chain.
        L = weights(np.log(r), math.log(radius))@np.log(self.values[indices, :2])
        value[:2] = np.exp(L[0]);derivative[:2] = value[:2]*L[1]/radius
        second[:2] = value[:2]*(L[2]+L[1]**2-L[1])/radius**2
        if not np.all(np.isfinite([value, derivative, second])) or not np.all(value[:2]>0):
            raise ValueError('Unrepresentable physical interpolation; no floor applied')
        self._cache[key] = (value.copy(), derivative.copy(), second.copy())
        return value, derivative, second

    def cartesian(self, points, width=7, center=(0., 0., 0.), samples_only=False):
        points = np.asarray(points, dtype=float);center = np.asarray(center, dtype=float)
        if points.ndim != 2 or points.shape[1]!=3 or center.shape!=(3,) or not np.all(np.isfinite(points)) or not np.all(np.isfinite(center)):
            raise ValueError('Need finite Cartesian points and one center')
        count = len(points);data = dict(points=points.copy(), radius=np.linalg.norm(points-center, axis=1),
            radial_value=np.zeros((count, 8)), radial_d=np.zeros((count, 8)), radial_dd=np.zeros((count, 8)))
        if samples_only:
            for p, r in enumerate(data['radius']):data['radial_value'][p], data['radial_d'][p], data['radial_dd'][p] = self.sample(r, width)
            return data
        for name in ['chi', 'alpha', 'trace', 'theta']:
            data[name] = np.zeros(count);data[name+'_d'] = np.zeros((count, 3));data[name+'_dd'] = np.zeros((count, 3, 3))
        for name in ['metric', 'a']:
            data[name] = np.zeros((count, 3, 3));data[name+'_d'] = np.zeros((count, 3, 3, 3));data[name+'_dd'] = np.zeros((count, 3, 3, 3, 3))
        for name in ['lambda', 'beta', 'B']:
            data[name] = np.zeros((count, 3));data[name+'_d'] = np.zeros((count, 3, 3));data[name+'_dd'] = np.zeros((count, 3, 3, 3))
        for p, point in enumerate(points):
            y = point-center;r = data['radius'][p]
            value, derivative, second = self.sample(r, width)
            n = y/r
            data['radial_value'][p] = value;data['radial_d'][p] = derivative;data['radial_dd'][p] = second
            fields = [Jet(value[i], derivative[i]*n,
                second[i]*np.outer(n, n)+derivative[i]/r*(np.eye(3)-np.outer(n, n))) for i in range(8)]
            radius = Jet(r, n, (np.eye(3)-np.outer(n, n))/r)
            normal = [Jet.variable(y[i], i)/radius for i in range(3)]
            for name, field in [('chi', fields[0]), ('alpha', fields[1]), ('trace', fields[3]), ('theta', fields[4])]:
                data[name][p] = field.value;data[name+'_d'][p] = field.d;data[name+'_dd'][p] = field.dd
            for i in range(3):
                data['metric'][p, i, i] = 1
                for name, radial in [('lambda', fields[5]), ('beta', fields[6]), ('B', fields[7])]:
                    field = radial*normal[i];data[name][p, i] = field.value;data[name+'_d'][p, :, i] = field.d;data[name+'_dd'][p, :, :, i] = field.dd
                for j in range(3):
                    field = 1.5*fields[2]*(normal[i]*normal[j]-(1./3 if i==j else 0.))
                    data['a'][p, i, j] = field.value;data['a_d'][p, :, i, j] = field.d;data['a_dd'][p, :, :, i, j] = field.dd
        if not all(np.all(np.isfinite(v)) for v in data.values()):
            raise ValueError('Nonfinite Cartesian jet export')
        return data


class Jet:
    """Independent second-order Cartesian algebra for component conversion."""
    def __init__(self, value, d=None, dd=None):
        self.value = float(value);self.d = np.zeros(3) if d is None else np.array(d);self.dd = np.zeros((3, 3)) if dd is None else np.array(dd)
    @staticmethod
    def variable(value, axis):
        result = Jet(value);result.d[axis] = 1;return result
    @staticmethod
    def coerce(value):
        return value if isinstance(value, Jet) else Jet(value)
    def __add__(self, other):
        other = self.coerce(other);return Jet(self.value+other.value, self.d+other.d, self.dd+other.dd)
    __radd__ = __add__
    def __neg__(self):
        return Jet(-self.value, -self.d, -self.dd)
    def __sub__(self, other):
        return self+-self.coerce(other)
    def __rsub__(self, other):
        return self.coerce(other)+-self
    def __mul__(self, other):
        other = self.coerce(other)
        return Jet(self.value*other.value, self.d*other.value+self.value*other.d,
            self.dd*other.value+self.value*other.dd+np.outer(self.d, other.d)+np.outer(other.d, self.d))
    __rmul__ = __mul__
    def power(self, exponent):
        return Jet(self.value**exponent, exponent*self.value**(exponent-1)*self.d,
            exponent*self.value**(exponent-1)*self.dd+exponent*(exponent-1)*self.value**(exponent-2)*np.outer(self.d, self.d))
    def __truediv__(self, other):
        return self*self.coerce(other).power(-1)


def diagnostics(snapshot):
    r = snapshot.r[1:];positive = snapshot.values[1:];bands = []
    for low, high in [(r[0], 5e-9), (1e-8, 1e-7), (1e-6, 1e-5), (1e-4, 1e-3), (.001, .01), (.01, .05), (.05, .2)]:
        mask = (r>=low)&(r<=high);row = dict(radius_interval=[float(low), high], nodes=int(sum(mask)), diagnostic_only=True)
        if sum(mask)>=6:
            log_r = np.log(r[mask]);centered = log_r-log_r.mean()
            for j, name in [(0, 'chi'), (1, 'alpha')]:
                log_f = np.log(positive[mask, j]);slope = float(np.dot(centered, log_f-log_f.mean())/np.dot(centered, centered))
                residual = log_f-log_f.mean()-slope*centered
                adjacent = np.diff(log_f)/np.diff(log_r)
                row[name] = dict(fitted_power=slope, log_fit_RMS=float(np.sqrt(np.mean(residual**2))),
                    adjacent_slope_min=float(min(adjacent)), adjacent_slope_max=float(max(adjacent)))
        else:row['status'] = 'insufficient nodes for power diagnostic'
        bands.append(row)
    profiles = {}
    for column, name in [(6, 'beta_r'), (7, 'B_r')]:
        values = snapshot.values[:, column];i = int(np.argmax(abs(values)))
        profiles[name] = dict(min=float(min(values)), max=float(max(values)), maximum_absolute=float(abs(values[i])), radius_of_maximum_absolute=float(snapshot.r[i]))
    return dict(time=snapshot.time, step=snapshot.step, positive_nodes=len(r),
        minimum_positive_radius=float(r[0]), minimum_chi=float(min(positive[:, 0])), minimum_alpha=float(min(positive[:, 1])),
        positive_on_every_positive_node=bool(np.all(positive[:, :2]>0)),
        endpoint_slot='Independent AF coefficients at r=0; not a physical positive-radius point',
        endpoint_stored_log_chi=float(snapshot.q[0, 0]), endpoint_stored_log_alpha=float(snapshot.q[0, 6]),
        metric_g1_deviation_max=float(np.max(abs(snapshot.q_physical[:, 1]))),
        reconstructed_vs_saved_chi_max=float(np.max(abs(snapshot.values[:, 0]-snapshot.diagnostic[:, 8]))),
        reconstructed_vs_saved_alpha_max=float(np.max(abs(snapshot.values[:, 1]-snapshot.diagnostic[:, 7]))),
        physical_q_cancellation_warning='Do not recover tiny chi/alpha by1+q_physical_chi or Aref+q_physical_alpha',
        measured_power_bands=bands, vector_profiles=profiles)


def freeze(args):
    source = args.source.resolve();output = args.output.resolve()
    if output==source or source in output.parents:
        raise ValueError('Freeze output must be outside the original read-only source tree')
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = source/'manifest.json';manifest = json.loads(manifest_path.read_text())
    if manifest.get('status')!='finished' or manifest.get('failure') is not None or manifest.get('time')!=10.:
        raise ValueError('Requested completed t10 source has not finished successfully')
    selected = {};targets = [1., 5., 10.]
    for path in sorted(source.glob('profile-*.npz')):
        with np.load(path, allow_pickle=False) as z:time = float(z['t'])
        if time in targets:
            if time in selected:raise ValueError('Duplicate exact snapshot time')
            selected[time] = path
    if set(selected)!=set(targets):raise ValueError('Exact requested snapshot times are missing')
    provenance = dict(source=str(source), manifest_sha256=sha(manifest_path), source_status=manifest['status'],
        source_time=manifest['time'], source_steps=manifest['steps'], failure=manifest['failure'],
        arguments=manifest['arguments'], initial_data=manifest['initial_data'], binding=manifest['binding'],
        method=manifest['method'], all_RK_stages=manifest['all_RK_stages'], snapshots=[],
        exporter_sha256=sha(__file__), frozen_source_verification={})
    for name, expected in manifest['binding']['source_sha256'].items():
        path = source/'frozen-source'/name
        provenance['frozen_source_verification'][name] = dict(present=path.exists(), matches=path.exists() and sha(path)==expected)
    for key, argument in [('library_sha256', 'library'), ('fast_sha256', 'fast_library')]:
        path = source/'frozen-source'/Path(manifest['arguments'][argument]).name
        provenance['frozen_source_verification'][path.name] = dict(present=path.exists(), matches=path.exists() and sha(path)==manifest['binding'][key])
    if not all(v['matches'] for v in provenance['frozen_source_verification'].values()):
        raise ValueError('Source/library binding verification failed')
    options_tree = ast.parse((source/'frozen-source'/'telegraph.py').read_text());fixed_options = {}
    for function in options_tree.body:
        if isinstance(function, ast.FunctionDef) and function.name=='options':
            for node in ast.walk(function):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id=='SimpleNamespace':
                    for keyword in node.keywords:
                        try:fixed_options[keyword.arg] = ast.literal_eval(keyword.value)
                        except (ValueError, TypeError):pass
    provenance['frozen_literal_options'] = fixed_options
    if args.acceptance:
        path = args.acceptance.resolve();acceptance = json.loads(path.read_text())
        review_path = Path(acceptance['review']);protocol_path = Path(acceptance['protocol'])
        if sha(review_path)!=acceptance['review_sha256'] or sha(protocol_path)!=acceptance['protocol_sha256']:
            raise ValueError('Historical acceptance review/protocol binding failed')
        provenance['source_success_review'] = dict(path=str(path), sha256=sha(path), checks=acceptance['checks'],
            all_listed_checks_passed=all(v.get('passed',False) for v in acceptance['checks'].values()),
            spatial_status=acceptance['spatial_status'], goal_status=acceptance['goal_status'],
            missing_evidence=acceptance['missing_evidence'], scope=acceptance['scope'],
            review_sha256=acceptance['review_sha256'], protocol_sha256=acceptance['protocol_sha256'])
    audit = []
    for time, path in sorted(selected.items()):
        destination = output/path.name
        if destination.exists() and sha(destination)!=sha(path):raise ValueError('Refusing to overwrite different frozen snapshot')
        if not destination.exists():shutil.copy2(path, destination)
        if sha(destination)!=sha(path):raise ValueError('Copied snapshot checksum mismatch')
        snapshot = RadialSnapshot(destination);audit.append(diagnostics(snapshot))
        provenance['snapshots'].append(dict(time=time, source_file=str(path), frozen_file=destination.name, sha256=sha(path), bytes=path.stat().st_size))
    provenance['exporter_manufactured_self_test'] = self_test()
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    (output/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(dict(snapshots=len(audit), exact_times=targets, source_binding_verified=True,
        all_positive_nodes=all(x['positive_on_every_positive_node'] for x in audit),
        source_spatial_status=provenance.get('source_success_review',{}).get('spatial_status'),
        source_goal_status=provenance.get('source_success_review',{}).get('goal_status'))))


def export(args):
    if args.output.exists() or args.output.with_suffix(args.output.suffix+'.metadata.json').exists():
        raise ValueError('Refusing to overwrite an existing export or source/evidence file')
    snapshot = RadialSnapshot(args.snapshot)
    if args.points:
        if args.points.suffix=='.npz':
            with np.load(args.points, allow_pickle=False) as z:points = z['points'].copy()
        else:
            table = np.genfromtxt(args.points, delimiter=',', names=True)
            points = np.column_stack([np.atleast_1d(table[name]) for name in ['x', 'y', 'z']])
    else:
        radii = np.array([float(x) for x in args.radii.split(',')]);points = np.column_stack([radii, np.zeros_like(radii), np.zeros_like(radii)])+args.center
    if len(points)>args.max_points:raise ValueError('Explicit export point budget exceeded')
    data = snapshot.cartesian(points, args.width, args.center, args.samples_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix=='.csv':
        arrays = [];names = []
        for key, value in data.items():
            if value.ndim==1:arrays.append(value);names.append(key)
            else:
                for index in np.ndindex(value.shape[1:]):arrays.append(value[(slice(None),)+index]);names.append(key+'_'+ '_'.join(map(str,index)))
        np.savetxt(args.output, np.column_stack(arrays), delimiter=',', header=','.join(names), comments='', fmt='%.17g')
    else:np.savez_compressed(args.output, **data)
    metadata = dict(snapshot=str(args.snapshot), snapshot_sha256=sha(args.snapshot), time=snapshot.time, step=snapshot.step,
        exporter_sha256=sha(__file__), points=len(points), center=args.center, width=args.width, radial_fields=RADIAL_FIELDS,
        array_shapes={k:list(v.shape) for k,v in data.items()}, derivative_convention='Derivative indices precede Cartesian component indices',
        interpolation='Positive fields: local log-field polynomial in log radius; signed fields: local physical polynomial in radius',
        endpoint='No zero-radius inference, parity condition or positive-radius extrapolation',
        scope='Instantaneous initialization/comparison only; no symmetry enforcement in any evolution',
        translated_center_warning='Nonzero center is a coordinate-placement test; current global-radius lapse source is not automatically compatible with translating punctures')
    args.output.with_suffix(args.output.suffix+'.metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), points=len(points), time=snapshot.time, width=args.width)))


def self_test():
    snapshot = RadialSnapshot.__new__(RadialSnapshot)
    snapshot.r = np.r_[0., np.geomspace(1e-8, 1., 257)];r = snapshot.r
    snapshot.values = np.column_stack([.9*r**2.3, .7*r**1.09, .2*r*r, .1*r*r,
        .01*r**3, .03*r, .1*r+.02*r**3, .2*r]);snapshot._cache = {}
    center = np.array([.03, -.02, .04]);directions = np.array([[1.,2.,3.],[-2.,1.,2.]])
    directions /= np.linalg.norm(directions,axis=1)[:,None]
    points = np.array([center+rho*n for rho in [1e-7,1e-4,.02,.3,.9] for n in directions])
    actual = snapshot.cartesian(points, center=center);worst = 0.
    for p, point in enumerate(points):
        y = [Jet.variable(point[i]-center[i],i) for i in range(3)]
        radius2 = sum(a*a for a in y);rho = actual['radius'][p]
        def compare(name,jet,index=()):
            nonlocal worst
            scale = max(abs(jet.value),1e-300)
            worst=max(worst,abs(actual[name][(p,)+index]-jet.value)/scale,
                np.max(abs(actual[name+'_d'][(p,slice(None))+index]-jet.d))*rho/scale,
                np.max(abs(actual[name+'_dd'][(p,slice(None),slice(None))+index]-jet.dd))*rho*rho/scale)
        for name, power, factor in [('chi',2.3,.9),('alpha',1.09,.7),('trace',2.,.1),('theta',3.,.01)]:
            compare(name,factor*radius2.power(power/2))
        for i in range(3):
            compare('lambda',.03*y[i],(i,));compare('beta',.1*y[i]+.02*radius2*y[i],(i,));compare('B',.2*y[i],(i,))
            for j in range(3):
                target=.3*y[i]*y[j]-(.1*radius2 if i==j else 0.)
                # Offdiagonal target can vanish on special directions; use the
                # radial tensor magnitude as a derivative conditioning scale.
                scale=.2*rho*rho
                worst=max(worst,abs(actual['a'][p,i,j]-target.value)/scale,
                    np.max(abs(actual['a_d'][p,:,i,j]-target.d))*rho/scale,
                    np.max(abs(actual['a_dd'][p,:,:,i,j]-target.dd))*rho*rho/scale)
    if worst>3e-10:raise ValueError(f'Cartesian manufactured export check failed: {worst}')
    return dict(points=len(points), maximum_scaled_value_gradient_Hessian_error=float(worst),
        physical_powers=[2.3,1.09], coordinate_center=center.tolist(), endpoint_extrapolation_used=False)


def main():
    parser = argparse.ArgumentParser();sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('freeze');p.add_argument('--source', type=Path, required=True);p.add_argument('--output', type=Path, required=True);p.add_argument('--acceptance', type=Path)
    p = sub.add_parser('export');p.add_argument('--snapshot', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True);group.add_argument('--points', type=Path);group.add_argument('--radii')
    p.add_argument('--center', type=float, nargs=3, default=[0.,0.,0.]);p.add_argument('--width',type=int,default=7,choices=[5,7,9]);p.add_argument('--max-points',type=int,default=50000);p.add_argument('--samples-only',action='store_true')
    sub.add_parser('self-test')
    args = parser.parse_args()
    if args.command=='freeze':freeze(args)
    elif args.command=='export':export(args)
    else:print(json.dumps(self_test()))


if __name__=='__main__':main()
