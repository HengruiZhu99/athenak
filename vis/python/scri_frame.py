"""Inertial-coordinate transport on a recorded scri, starting from a round cut.

The native evolution is untouched. Angular interpolation of frame variables uses
real scalar harmonics; radiation is evaluated from the full native nodal jets.
"""
import numpy as np
from scri_angular import scalar_basis, weighted_projection
from scri_geometry import conformal_ricci_projection


def interpolation_weights(nodes, at):
    nodes = np.asarray(nodes)
    out = np.ones(len(nodes))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                out[i] *= (at-nodes[j])/(nodes[i]-nodes[j])
    return out


def generator_geometry(x, inverse, connection, omega_gradient=None, omega_hessian=None):
    """Native-time generator velocity and preferred-frame conformal source."""
    if omega_gradient is None:
        omega_gradient = np.column_stack((np.zeros(len(x)), -x))
        omega_hessian = np.diag([0., -1, -1, -1])
    generator = np.einsum('qab,qb->qa', inverse, omega_gradient)
    if np.min(generator[:, 0]) <= 0:
        raise ValueError('scri generator does not advance native time')
    hessian = omega_hessian-np.einsum('qcab,qc->qab', connection, omega_gradient)
    box = np.einsum('qab,qab->q', inverse, hessian)
    return generator[:, 1:]/generator[:, :1], box/(4*generator[:, 0]), generator[:, 0], hessian


class FrameTransport:
    def __init__(self, x, weights, lmax=8):
        if np.max(abs(np.linalg.norm(x, axis=-1)-1)) > 2e-13:
            raise ValueError('transport grid must be the native unit sphere')
        self.x = x; self.weights = weights; self.lmax = lmax
        self.y, self.dy, self.ddy = scalar_basis(x, lmax)
        self.project = weighted_projection(self.y, weights)
        self.initial = self.project@np.column_stack((np.zeros(len(x)), x, np.zeros(len(x))))

    def rhs(self, coefficients, flow, source, generator_time):
        values = self.y@coefficients
        gradients = np.einsum('qki,kf->qfi', self.dy, coefficients)
        rates = -np.einsum('qi,qfi->qf', flow, gradients)
        rates[:, 0] -= source
        rates[:, 4] += np.exp(values[:, 0])/generator_time
        return self.project@rates

    def evolve(self, times, flows, sources, generator_times):
        """RK4 transport of log(omega), three inertial labels, and Bondi time.

        A cubic interpolant of the sampled geometry supplies RK midpoint data.
        The initial round unit-sphere frame has omega=1, y=x, u=0.
        """
        times = np.asarray(times)
        if len(times) < 4 or np.any(np.diff(times) <= 0):
            raise ValueError('frame transport requires four increasing sample times')
        states = np.empty((len(times),)+self.initial.shape); states[0] = self.initial
        for q in range(len(times)-1):
            start = min(max(q-1, 0), len(times)-4)
            indices = slice(start, start+4)
            weights = interpolation_weights(times[indices], .5*(times[q]+times[q+1]))
            middle = [np.einsum('t,t...->...', weights, field[indices]) for field in (flows, sources, generator_times)]
            if np.min(middle[2]) <= 0:
                raise ValueError('interpolated generator time is nonpositive')
            step = times[q+1]-times[q]; state = states[q]
            k1 = self.rhs(state, flows[q], sources[q], generator_times[q])
            k2 = self.rhs(state+.5*step*k1, *middle)
            k3 = self.rhs(state+.5*step*k2, *middle)
            k4 = self.rhs(state+step*k3, flows[q+1], sources[q+1], generator_times[q+1])
            states[q+1] = state+step*(k1+2*k2+2*k3+k4)/6
        if not np.isfinite(states).all():
            raise ValueError('nonfinite transported frame')
        return states

    def observe(self, coefficients, metric, curvature, connection, flow, source,
                generator_time, coefficient_time2, coefficient_time1=None,
                omega_gradient=None):
        """Project on the explicit constant-Bondi-time cuts of the transported frame.

        First/second time derivatives of the transported coefficients must be
        supplied consistently with their sampled history. This includes the
        temporal conformal derivatives when Bondi cuts tilt relative to native t.
        """
        if coefficient_time1 is None:
            coefficient_time1 = self.rhs(coefficients, flow, source, generator_time)
        values = self.y@coefficients
        time_rates = self.y@coefficient_time1
        gradients = np.einsum('qki,kf->qfi', self.dy, coefficients)
        factor = np.exp(values[:, 0])
        gradient = np.column_stack((time_rates[:, 0], gradients[:, 0]))
        hessian = np.zeros((len(self.x), 4, 4))
        hessian[:, 1:, 1:] = np.einsum('qkij,k->qij', self.ddy, coefficients[:, 0])
        hessian[:, 0, 0] = self.y@coefficient_time2[:, 0]
        mixed = np.einsum('qki,k->qi', self.dy, coefficient_time1[:, 0])
        hessian[:, 0, 1:] = hessian[:, 1:, 0] = mixed
        # Normalize the interpolated inertial embedding and differentiate that
        # normalization in time and all spatial directions.
        labels = values[:, 1:4]; norm = np.linalg.norm(labels, axis=-1)
        inertial = labels/norm[:, None]
        label_gradient = np.concatenate((time_rates[:, 1:4, None], gradients[:, 1:4]), axis=-1)
        dl = (label_gradient-inertial[:, :, None]*np.einsum('qa,qai->qi', inertial, label_gradient)[:, None, :])/norm[:, None, None]
        sin_theta = np.sqrt(1-inertial[:, 0]**2)
        if np.min(sin_theta) < 1e-8:
            raise ValueError('inertial label reached dyad-chart pole')
        dtheta = -dl[:, 0]/sin_theta[:, None]
        dphi = (inertial[:, 1, None]*dl[:, 2]-inertial[:, 2, None]*dl[:, 1])/sin_theta[:, None]**2
        du = np.column_stack((time_rates[:, 4], gradients[:, 4]))
        if omega_gradient is None:
            omega_gradient = np.column_stack((np.zeros(len(self.x)), -self.x))
        coordinate_jacobian = np.stack((du, factor[:, None]*omega_gradient, dtheta, dphi), axis=1)
        inverse_jacobian = np.linalg.inv(coordinate_jacobian)
        polar = inverse_jacobian[:, :, 2]
        azimuth = inverse_jacobian[:, :, 3]/sin_theta[:, None]
        conformal_metric = metric*factor[:, None, None]**2
        e = np.einsum('qa,qab,qb->q', polar, conformal_metric, polar)
        f = np.einsum('qa,qab,qb->q', polar, conformal_metric, azimuth)
        g = np.einsum('qa,qab,qb->q', azimuth, conformal_metric, azimuth)
        if np.min(e) <= 0 or np.min(g-f*f/e) <= 0:
            raise ValueError('nonpositive transformed cut metric')
        # Anchor polarization to inertial north. Orthonormalization removes the
        # measured finite-discretization defect, which is reported below.
        first = polar/np.sqrt(e)[:, None]
        second = (azimuth-(f/e)[:, None]*polar)/np.sqrt(g-f*f/e)[:, None]
        bondi_dyad = (first-1j*second)/np.sqrt(2)
        news = -conformal_ricci_projection(curvature, connection, bondi_dyad*factor[:, None], gradient, hessian, factor)
        # Angular measure on this native-time sample. Subsequent interpolation
        # follows fixed inertial labels to construct common Bondi-time samples.
        n = self.x; sin_native = np.sqrt(1-n[:, 0]**2)
        etheta = (n*n[:, :1]-np.array([1., 0, 0]))/sin_native[:, None]
        ephi = np.column_stack((np.zeros(len(n)), -n[:, 2], n[:, 1]))/sin_native[:, None]
        jacobian = np.stack((np.column_stack((np.sum(dtheta[:, 1:]*etheta, axis=-1), np.sum(dtheta[:, 1:]*ephi, axis=-1))),
                             np.column_stack((np.sum(dphi[:, 1:]*etheta, axis=-1), np.sum(dphi[:, 1:]*ephi, axis=-1)))), axis=1)
        determinant = np.linalg.det(jacobian)
        if np.min(determinant) <= 0:
            raise ValueError('inertial angular map folded')
        angular_weights = self.weights*sin_theta*determinant
        requested_time_rate = -source-np.einsum('qi,qi->q', flow, gradients[:, 0])
        trace_defect = .5*(e+g)-1
        mean_trace_defect = float(np.sum(self.weights*trace_defect)/np.sum(self.weights))
        diagnostics = dict(label_norm_max=float(np.max(abs(norm-1))),
            round_scale_mean=mean_trace_defect,
            round_nonuniform_scale_max=float(np.max(abs(trace_defect-mean_trace_defect))),
            round_shear_max=float(np.max(np.sqrt(.5*(e-g)**2+2*f*f))),
            round_cut_max=float(np.max(np.sqrt((e-1)**2+(g-1)**2+2*f*f))),
            round_cut_rms=float(np.sqrt(np.sum(self.weights*((e-1)**2+(g-1)**2+2*f*f))/np.sum(self.weights))),
            frame_source_fit_max=float(np.max(abs(time_rates[:, 0]-requested_time_rate))),
            clock_transport_max=float(np.max(abs(generator_time/factor*(time_rates[:, 4]+np.einsum('qi,qi->q', flow, gradients[:, 4]))-1))),
            solid_angle_sum=float(np.sum(angular_weights)))
        return dict(news=news, direction=inertial, bondi_time=values[:, 4], factor=factor,
                    angular_weights=angular_weights, diagnostics=diagnostics)
