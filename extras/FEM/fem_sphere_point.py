#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2019-2020 Jakub M. Dzik (Laboratory of Neuroinformatics;   #
#    Nencki Institute of Experimental Biology of Polish Academy of Sciences)  #
#                                                                             #
#    This software is free software: you can redistribute it and/or modify    #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This software is distributed in the hope that it will be useful,         #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this software.  If not, see http://www.gnu.org/licenses/.     #
#                                                                             #
###############################################################################

import os
import logging
import warnings

import numpy as np
from scipy.interpolate import RegularGridInterpolator

try:
    from . import _fem_common as fc
    # When run as script raises:
    #  - `ModuleNotFoundError(ImportError)` (Python 3.6-7), or
    #  - `SystemError` (Python 3.3-5), or
    #  - `ValueError` (Python 2.7).

except (ImportError, SystemError, ValueError):
    import _fem_common as fc


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# TODO: Deduplicate with `fem_sphere_gaussian`
class _SomeSpherePointLoaderBase(object):
    ATTRIBUTES = ['R',
                  'COMPLETED',
                  'brain_conductivity',
                  'brain_radius',
                  'scalp_radius',
                  'degree',
                  ]

    def _load(self):
        self._provide_attributes()

    def _load_attributes(self):
        with np.load(self.path) as fh:
            self._load_attributes_from_numpy_file(fh)

    def _load_attributes_from_numpy_file(self, fh):
        for attr in self.ATTRIBUTES:
            setattr(self, attr, fh[attr])


class _SomeSpherePointPotentialLoaderBase(_SomeSpherePointLoaderBase):
    def _load_attributes_from_numpy_file(self, fh):
        super(_SomeSpherePointPotentialLoaderBase,
              self)._load_attributes_from_numpy_file(fh)

        self.POTENTIAL = fh['POTENTIAL']
        self.STATS = list(fh['STATS'])


class _PointLoaderBase(_SomeSpherePointPotentialLoaderBase):
    ATTRIBUTES = _SomeSpherePointPotentialLoaderBase.ATTRIBUTES + \
                 ['sampling_frequency',
                  ]


class _PointLoaderBase3D(_PointLoaderBase):
    @property
    def _xz(self):
        sf = self.sampling_frequency
        for x_idx, z_idx in self._xz_idx:
            yield self.X_SAMPLE[sf + x_idx], self.Z_SAMPLE[sf + z_idx]

    @property
    def _xz_idx(self):
        _idx = 0
        for z_idx in range(self.sampling_frequency + 1):
            for x_idx in range(z_idx + 1):
                assert _idx == z_idx * (z_idx + 1) // 2 + x_idx
                yield x_idx, z_idx
                _idx += 1

    def _load(self):
        super(_PointLoaderBase3D, self)._load()

        self.X_SAMPLE = np.linspace(-self.scalp_radius,
                                    self.scalp_radius,
                                    2 * self.sampling_frequency + 1)
        self.Y_SAMPLE = self.X_SAMPLE
        self.Z_SAMPLE = self.X_SAMPLE


class _PointLoaderBase2D(_PointLoaderBase):
    def _load(self):
        super(_PointLoaderBase2D, self)._load()
        self.X_SAMPLE = np.linspace(0,
                                    self.scalp_radius,
                                    self.sampling_frequency + 1)
        self.Y_SAMPLE = np.linspace(-self.scalp_radius,
                                    self.scalp_radius,
                                    2 * self.sampling_frequency + 1)


# class SomeSpherePointSourceFactoryOnlyCSD(_SomeSpherePointLoaderBase):
#     def __init__(self, filename):
#         self.path = filename
#         self._load()
#         self._r_index = {r: i for i, r in enumerate(self.R)}
#
#     def __call__(self, r, altitude, azimuth):
#         return _SourceBase(r, altitude, azimuth,
#                            self)
#
#     def _provide_attributes(self):
#         self._load_attributes()


class _SourceBase(object):
    def __init__(self, r, altitude, azimuth, parent):
        self._r = r
        self._altitude = altitude
        self._azimuth = azimuth
        self.parent = parent

        sin_alt = np.sin(self._altitude)  # np.cos(np.pi / 2 - altitude)
        cos_alt = np.cos(self._altitude)  # np.sin(np.pi / 2 - altitude)
        sin_az = np.sin(self._azimuth)  # -np.sin(-azimuth)
        cos_az = np.cos(self._azimuth)  # np.cos(-azimuth)

        self._apply_trigonometric_functions(cos_alt, sin_alt, cos_az, sin_az)

    def _apply_trigonometric_functions(self, cos_alt, sin_alt, cos_az, sin_az):
        r2 = self._r * cos_alt
        self._x = r2 * cos_az
        self._y = self._r * sin_alt
        self._z = -r2 * sin_az

    # def csd(self, X, Y, Z):
    #     return np.where((X**2 + Y**2 + Z**2 > self.parent.brain_radius ** 2),
    #                     0,
    #                     self._a
    #                     * np.exp(-0.5
    #                              * (np.square(X - self._x)
    #                                 + np.square(Y - self._y)
    #                                 + np.square(Z - self._z))
    #                              / self.parent.standard_deviation ** 2))

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def r(self):
        return self._r

    @property
    def altitude(self):
        return self._altitude

    @property
    def azimuth(self):
        return self._azimuth


class _RotatingSourceBase(_SourceBase):
    def __init__(self, r, altitude, azimuth, parent, interpolator):
        super(_RotatingSourceBase,
              self).__init__(r, altitude, azimuth, parent)

        self._interpolator = interpolator

    def _apply_trigonometric_functions(self, cos_alt, sin_alt, cos_az, sin_az):
        super(_RotatingSourceBase,
              self)._apply_trigonometric_functions(cos_alt, sin_alt, cos_az, sin_az)
        self._ROT = np.matmul([[sin_alt, cos_alt, 0],
                               [-cos_alt, sin_alt, 0],
                               [0, 0, 1]],
                              [[cos_az, 0, sin_az],
                               [0, 1, 0],
                               [-sin_az, 0, cos_az]]
                              )

    def _rotated(self, X, Y, Z):
        _X = self._ROT[0, 0] * X + self._ROT[1, 0] * Y + self._ROT[2, 0] * Z
        _Y = self._ROT[0, 1] * X + self._ROT[1, 1] * Y + self._ROT[2, 1] * Z
        _Z = self._ROT[0, 2] * X + self._ROT[1, 2] * Y + self._ROT[2, 2] * Z
        return _X, _Y, _Z


class _SomeSpherePointSourceFactoryBase(object):
    def __init__(self, filename):
        self.path = filename
        self._load()
        self._r_index = {r: i for i, r in enumerate(self.R)}
        self._interpolator = [None] * len(self.R)

    def _provide_attributes(self):
        self._load_attributes()

    def __call__(self, r, altitude, azimuth):
        interpolator = self._source_prefabricates(r)
        return self._Source(r, altitude, azimuth,
                            self,
                            interpolator)

    def _source_prefabricates(self, r):
        r_idx = self._r_index[r]
        return self._get_interpolator(r_idx)

    def _get_interpolator(self, r_idx):
        interpolator = self._interpolator[r_idx]
        if interpolator is None:
            interpolator = self._make_interpolator(self.POTENTIAL[r_idx, :, :])
            self._interpolator[r_idx] = interpolator
        return interpolator


class SomeSpherePointSourceFactory3D(_SomeSpherePointSourceFactoryBase,
                                     _PointLoaderBase3D):
    def _make_interpolator(self, COMPRESSED):
        sf = self.sampling_frequency
        POTENTIAL = fc.empty_array((sf + 1,
                                    len(self.Y_SAMPLE),
                                    sf + 1))
        for xz_idx, (x_idx, z_idx) in enumerate(self._xz_idx):
            P = COMPRESSED[xz_idx, :]
            POTENTIAL[x_idx, :, z_idx] = P
            if x_idx > z_idx:
                POTENTIAL[z_idx, :, x_idx] = P
        interpolator = RegularGridInterpolator((self.X_SAMPLE[sf:],
                                                self.Y_SAMPLE,
                                                self.Z_SAMPLE[sf:]),
                                               POTENTIAL,
                                               bounds_error=False,
                                               fill_value=np.nan)
        return interpolator

    class _Source(_RotatingSourceBase):
        def potential(self, X, Y, Z):
            _X, _Y, _Z = self._rotated(X, Y, Z)
            return self._interpolator(np.stack((abs(_X), _Y, abs(_Z)),
                                               axis=-1))


class SomeSpherePointSourceFactoryLinear2D(_SomeSpherePointSourceFactoryBase,
                                           _PointLoaderBase2D):
    def _make_interpolator(self, POTENTIAL):
        return RegularGridInterpolator((self.X_SAMPLE,
                                        self.Y_SAMPLE),
                                       POTENTIAL,
                                       bounds_error=False,
                                       fill_value=np.nan)

    class _Source(_RotatingSourceBase):
        def potential(self, X, Y, Z):
            _X, _Y, _Z = self._rotated(X, Y, Z)
            return self._interpolator(np.stack((np.sqrt(np.square(_X)
                                                        + np.square(_Z)),
                                                _Y),
                                               axis=-1))


class SomeSpherePointSourceFactory2D(SomeSpherePointSourceFactoryLinear2D):
    def __init__(self, *args, **kwargs):
        warnings.warn('SomeSpherePointSourceFactory2D is deprecated; use SomeSpherePointSourceFactoryLinear2D instead',
                      DeprecationWarning,
                      stacklevel=2)
        super(SomeSpherePointSourceFactory2D,
              self).__init__(*args, **kwargs)


class _SomeSphereControllerBase(object):
    FEM_ATTRIBUTES = ['brain_conductivity',
                      'brain_radius',
                      'scalp_radius',
                      ]
    TOLERANCE = np.finfo(float).eps

    cortex_radius_external = 0.079
    cortex_radius_internal = 0.067
    source_resolution = 4

    def __init__(self, fem):
        self._fem = fem

    def __enter__(self):
        self._load()
        self._anything_new = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._anything_new:
            self.save(self.path)

    def _results(self):
        return {attr: getattr(self, attr)
                for attr in self.ATTRIBUTES
                }

    def save(self, path):
        results = self._results()
        results['STATS'] = self.STATS
        results['POTENTIAL'] = self.POTENTIAL
        np.savez_compressed(path, **results)


    def fem(self, y):
        return self._fem(int(self.degree), y)

    def _validate_attributes(self):
        for attr, value in self._fem_attributes:
            loaded = getattr(self, attr)
            err = (loaded - value) / value
            msg = "self.{0} / self._fem.{0} - 1 = {1:.2e}".format(attr, err)

            assert abs(err) < self.TOLERANCE, msg

    def _provide_attributes(self):
        try:
            self._load_attributes()

        except Exception as e:
            logger.warning(str(e))
            for attr, value in self._fem_attributes:
                setattr(self, attr, value)

            self._empty_solutions()

        else:
            self._validate_attributes()

    def _empty_solutions(self):
        n = len(self.R)
        self.COMPLETED = np.zeros(n, dtype=bool)
        self.STATS = []
        self.POTENTIAL = fc.empty_array(self._potential_size(n))

    @property
    def _fem_attributes(self):
        for attr in self.FEM_ATTRIBUTES:
            logger.debug(attr)
            yield attr, getattr(self._fem, attr)


class _SomeSpherePointController3D(_PointLoaderBase3D,
                                   _SomeSphereControllerBase):
    sampling_frequency = 256

    def _potential_size(self, n):
        xz_size = (self.sampling_frequency + 1) * (self.sampling_frequency + 2) // 2
        return (n,
                xz_size,
                2 * self.sampling_frequency + 1)

    @property
    def path(self):
        fn = '{0._fem.mesh_name}_point_deg_{0.degree}.npz'.format(self)
        return fc._SourceFactory_Base.solution_path(fn, False)

    def sample(self, r, potential, fem=None):
        if fem is None:
            fem = self._fem

        self._anything_new = True
        idx_r = np.where(np.isclose(self.R, r))[0][0]

        sampling_time = np.nan
        if potential is not None:
            logging.info('Sampling 3D')
            hits = 0
            exceptions = 0
            misses = 0
            r2 = self.scalp_radius ** 2
            XZ = np.array(list(self._xz))
            sample_stopwatch = fc.Stopwatch()
            with sample_stopwatch:
                for idx_xz, (x, z) in enumerate(XZ):
                    r_xz_2 = x ** 2 + z ** 2
                    if r_xz_2 > r2:
                        misses += len(self.Y_SAMPLE)
                        continue

                    for idx_y, y in enumerate(self.Y_SAMPLE):
                        if r_xz_2 + y ** 2 > r2:
                            misses += 1
                            continue
                        try:
                            v = potential(x, y, z)
                        except Exception as e:
                            if x < 0 or z < 0 or abs(
                                    y) > self.scalp_radius or x > self.scalp_radius or z > self.scalp_radius:
                                logger.warning('coords out of bounding box')
                            exceptions += 1
                        else:
                            hits += 1
                            self.POTENTIAL[idx_r,
                                           idx_xz,
                                           idx_y] = v

                self.POTENTIAL[idx_r, :, :] += fem._base_potential(r,
                                                                   XZ[:, :1],
                                                                   self.Y_SAMPLE.reshape(1, -1),
                                                                   XZ[:, 1:])

            sampling_time = float(sample_stopwatch)
            logger.info('H={} ({:.2f}),\tM={} ({:.2f}),\tE={} ({:.2f})'.format(hits,
                                                                               hits / float(hits + misses + exceptions),
                                                                               misses,
                                                                               misses / float(
                                                                                   hits + misses + exceptions),
                                                                               exceptions,
                                                                               exceptions / float(hits + exceptions)))

        self.STATS.append((r,
                           sampling_time,
                           fem.iterations,
                           float(fem.solving_time),
                           float(fem.local_preprocessing_time),
                           float(fem.global_preprocessing_time)))
        self.COMPLETED[idx_r] = True
        return 0 if np.isnan(sampling_time) else sampling_time


class _SomeSpherePointController2D(_PointLoaderBase2D,
                                   _SomeSphereControllerBase):
    sampling_frequency = 1024

    def _potential_size(self, n):
        return (n,
                self.sampling_frequency + 1,
                2 * self.sampling_frequency + 1)

    @property
    def path(self):
        fn = '{0._fem.mesh_name}_point_deg_{0.degree}_2D.npz'.format(self)

        return fc._SourceFactory_Base.solution_path(fn, False)

    def sample(self, r, potential, fem=None):
        if fem is None:
            fem = self._fem

        self._anything_new = True
        idx_r = np.where(np.isclose(self.R, r))[0][0]

        sampling_time = np.nan
        if potential is not None:
            logging.info('Sampling 2D')
            angle = fem.FRACTION_OF_SPACE * np.pi
            SIN_COS = np.array([np.sin(angle),
                                np.cos(angle)])
            r2 = self.scalp_radius ** 2
            sample_stopwatch = fc.Stopwatch()
            with sample_stopwatch:
                for idx_xz, xz in enumerate(self.X_SAMPLE):
                    x, z = SIN_COS * xz
                    xz2 = xz ** 2
                    for idx_y, y in enumerate(self.Y_SAMPLE):
                        if xz2 + y ** 2 > r2:
                            continue
                        try:
                            v = potential(x, y, z)
                        except Exception as e:
                            pass
                        else:
                            self.POTENTIAL[idx_r,
                                           idx_xz,
                                           idx_y] = v
                # It is known that `(sin(a) * r) ** 2 + (cos(a) * r) **2 == r ** 2`
                # and the source is at x, z == 0, 0, thus the commented calculations
                # can be safely omitted.
                # XZ_SAMPLE = self.X_SAMPLE.reshape(-1, 1) * SIN_COS.reshape(1, -1)
                self.POTENTIAL[idx_r, :, :] += fem._base_potential(r,
                                                                   self.X_SAMPLE.reshape(-1, 1),
                                                                   # XZ_SAMPLE[:, :1],
                                                                   self.Y_SAMPLE.reshape(1, -1),
                                                                   0)
                                                                   # XZ_SAMPLE[:, 1:])

            sampling_time = float(sample_stopwatch)

        self.STATS.append((r,
                           sampling_time,
                           fem.iterations,
                           float(fem.solving_time),
                           float(fem.local_preprocessing_time),
                           float(fem.global_preprocessing_time)))
        self.COMPLETED[idx_r] = True
        return 0 if np.isnan(sampling_time) else sampling_time


if __name__ == '__main__':
    import sys

    try:
        from dolfin import (Expression, Constant, DirichletBC, Measure,
                            FacetNormal,
                            inner, grad, assemble,
                            HDF5File)

    except (ModuleNotFoundError, ImportError):
        print("""Run docker first:
        $ docker run -ti --env HOST_UID=$(id -u) --env HOST_GID=$(id -g) \\
                     -v $(pwd):/home/fenics/shared:Z \\
                     -w /home/fenics/shared \\
                     quay.io/fenicsproject/stable
        """)
    else:
        class _SphericalPointPotential(fc._FEM_Base):
            FRACTION_OF_SPACE = 1.0
            scalp_radius = 0.090

            def __init__(self, mesh_name):
                super(_SphericalPointPotential, self).__init__(
                      mesh_path=os.path.join(fc.DIRNAME,
                                             'meshes',
                                             mesh_name))
                self._base_potential_constant = 0.25 / (np.pi * self.BASE_CONDUCTIVITY)
                self.mesh_name = mesh_name

            def create_integration_subdomains(self):
                super(_SphericalPointPotential,
                      self).create_integration_subdomains()
                self._ds = Measure("ds")(subdomain_data=self._boundaries)

            def _lhs(self):
                return sum(inner(Constant(c) * grad(self._potential_trial),
                                 grad(self._v)) * self._dx(x)
                           for x, c in self.CONDUCTIVITY.items())

            def _boundary_condition(self, y, *args, **kwargs):
                gdim = self._mesh.geometry().dim()
                dofs_x = self._V.tabulate_dof_coordinates().reshape((-1, gdim))
                R2 = np.square(dofs_x).sum(axis=1)
                # logger.debug('R2.min() == {}'.format(R2.min()))
                central_idx = np.argmin(R2)
                x0, y0, z0 = dofs_x[central_idx]
                # logger.debug('R2[{}] == {}'.format(central_idx, R2[central_idx]))
                logger.debug('DBC at: {}, {}, {}'.format(x0, y0, z0))

                base_potential_at_center = self._base_potential(y, x0, y0, z0)
                return DirichletBC(self._V,
                                   Constant(-base_potential_at_center),
                                   "near(x[0], {}) && near(x[1], {}) && near(x[2], {})".format(x0, y0, z0),
                                   "pointwise")

            def _base_potential(self, src_y, x, y, z):
                return (self._base_potential_constant
                        / np.sqrt(np.square(x)
                                  + np.square(y - src_y)
                                  + np.square(z)))

            def _rhs(self, degree, y):
                base_potential = Expression(
                    f'''
                    0.25 / ({np.pi * self.BASE_CONDUCTIVITY})
                    / sqrt((x[0])*(x[0])
                           + (x[1] - {y})*(x[1] - {y})
                           + (x[2])*(x[2]))
                    ''',
                    degree=degree,
                    domain=self._mesh)
                n = FacetNormal(self._mesh)
                return -sum((inner((Constant(c - self.BASE_CONDUCTIVITY)
                                    * grad(base_potential)),
                                   grad(self._v))
                             * self._dx(x)
                             for x, c in self.CONDUCTIVITY.items()
                             if c != self.BASE_CONDUCTIVITY),
                            # # Eq. 20 at Piastra et al 2018
                            # sum((Constant(self.BASE_CONDUCTIVITY)
                            #      * inner(n, grad(self._base_potential))
                            #      * self._v
                            #      * self._ds(s))
                            #      for s in self.SURFACE_CONDUCTIVITY)

                            # Eq. 19 at Piastra et al 2018
                            sum((Constant(c)
                                 * inner(n, grad(base_potential))
                                 * self._v
                                 * self._ds(s))
                                for s, c in self.SURFACE_CONDUCTIVITY.items())
                            )

            @property
            def degree(self):
                return self._degree


        class OneSpherePointPotentialFEM(_SphericalPointPotential):
            startswith = 'one_sphere'

            brain_conductivity = 0.33  # S / m

            brain_radius = 0.079
            # roi_radius_min = 0.067
            # roi_radius_tangent = 0.006

            _BRAIN_VOLUME = 1
            _SCALP_SURFACE = 2

            CONDUCTIVITY = {
                            _BRAIN_VOLUME: brain_conductivity,
                            }

            BASE_CONDUCTIVITY = brain_conductivity

            SURFACE_CONDUCTIVITY = {
                                    _SCALP_SURFACE: brain_conductivity,
                                    }


        class EighthWedgeOfOneSpherePointPotentialFEM(
                  OneSpherePointPotentialFEM):
            startswith = 'eighth_wedge_of_one_sphere'
            FRACTION_OF_SPACE = 0.125


        class SixthWedgeOfOneSpherePointPotentialFEM(
                  OneSpherePointPotentialFEM):
            startswith = 'sixth_wedge_of_one_sphere'
            FRACTION_OF_SPACE = 1.0 / 6


        class TwoSpheresPointPotentialFEM(_SphericalPointPotential):
            startswith = 'two_spheres'

            brain_conductivity = 0.33  # S / m
            skull_conductivity = brain_conductivity / 20

            brain_radius = 0.079
            # roi_radius_min = 0.067
            # roi_radius_tangent = 0.006

            _BRAIN_VOLUME = 1
            _SKULL_VOLUME = 2
            _SCALP_SURFACE = 4

            CONDUCTIVITY = {
                            _BRAIN_VOLUME: brain_conductivity,
                            _SKULL_VOLUME: skull_conductivity,
                            }

            BASE_CONDUCTIVITY = brain_conductivity

            SURFACE_CONDUCTIVITY = {
                                    _SCALP_SURFACE: skull_conductivity,
                                    }

        class EighthWedgeOfTwoSpheresPointPotentialFEM(
                  TwoSpheresPointPotentialFEM):
            startswith = 'eighth_wedge_of_two_spheres'
            FRACTION_OF_SPACE = 0.125


        class SixthWedgeOfTwoSpheresPointPotentialFEM(
                  TwoSpheresPointPotentialFEM):
            startswith = 'sixth_wedge_of_two_spheres'
            FRACTION_OF_SPACE = 1.0 / 6


        class FourSpheresPointPotentialFEM(_SphericalPointPotential):
            startswith = 'four_spheres'

            brain_conductivity = 0.33  # S / m
            csf_conductivity = brain_conductivity * 5
            skull_conductivity = brain_conductivity / 20
            scalp_conductivity = brain_conductivity

            brain_radius = 0.079
            # roi_radius_min = 0.067
            # roi_radius_tangent = 0.006


            _BRAIN_VOLUME = 1
            _CSF_VOLUME = 2
            _SKULL_VOLUME = 3
            _SCALP_VOLUME = 4
            _SCALP_SURFACE = 6

            CONDUCTIVITY = {
                            _BRAIN_VOLUME: brain_conductivity,
                            _CSF_VOLUME: csf_conductivity,
                            _SKULL_VOLUME: skull_conductivity,
                            _SCALP_VOLUME: scalp_conductivity,
                            }

            BASE_CONDUCTIVITY = brain_conductivity

            SURFACE_CONDUCTIVITY = {
                                    _SCALP_SURFACE: scalp_conductivity,
                                    }

        class EighthWedgeOfFourSpheresPointPotentialFEM(
                  FourSpheresPointPotentialFEM):
            startswith = 'eighth_wedge_of_four_spheres'
            FRACTION_OF_SPACE = 0.125


        class SixthWedgeOfFourSpheresPointPotentialFEM(
                  FourSpheresPointPotentialFEM):
            startswith = 'sixth_wedge_of_four_spheres'
            FRACTION_OF_SPACE = 1.0 / 6


        logging.basicConfig(level=logging.INFO)

        if not os.path.exists(fc.SOLUTION_DIRECTORY):
            os.makedirs(fc.SOLUTION_DIRECTORY)

        for mesh_name in sys.argv[1:]:
            for SpherePointFEM in [OneSpherePointPotentialFEM,
                                   TwoSpheresPointPotentialFEM,
                                   FourSpheresPointPotentialFEM,
                                   EighthWedgeOfOneSpherePointPotentialFEM,
                                   EighthWedgeOfTwoSpheresPointPotentialFEM,
                                   EighthWedgeOfFourSpheresPointPotentialFEM,
                                   SixthWedgeOfOneSpherePointPotentialFEM,
                                   SixthWedgeOfTwoSpheresPointPotentialFEM,
                                   SixthWedgeOfFourSpheresPointPotentialFEM,
                                   ]:
                if mesh_name.startswith(SpherePointFEM.startswith):
                    fem = SpherePointFEM(mesh_name=mesh_name)
                    break
            else:
                logger.warning('Missing appropriate FEM class for {}'.format(mesh_name))
                continue

            controller = _SomeSpherePointController3D(fem)
            controller_2D = _SomeSpherePointController2D(fem)

            for controller.degree in [1, 2, 3]:
                controller_2D.degree = controller.degree
                RES = 4
                span = controller.cortex_radius_external - controller.cortex_radius_internal
                controller.R = np.linspace(controller.cortex_radius_internal + 0.5 * span / RES,
                                           controller.cortex_radius_external - 0.5 * span / RES,
                                           RES)
                controller_2D.R = controller.R

                logger.info('Point ({}; deg={})'.format(
                    mesh_name,
                    controller.degree))

                tmp_mark = 0
                with controller:
                    with controller_2D:
                        save_stopwatch = fc.Stopwatch()
                        sample_stopwatch = fc.Stopwatch()

                        with fc.Stopwatch() as unsaved_time:
                            for idx_r, (src_r, completed_3D, completed_2D) in enumerate(zip(controller.R,
                                                                                            controller.COMPLETED,
                                                                                            controller_2D.COMPLETED)):
                                logger.info(
                                    'Point r={} ({}, deg={})'.format(
                                        src_r,
                                        mesh_name,
                                        controller.degree))
                                if completed_2D and completed_3D:
                                    logger.info('Already found, skipping')
                                    continue

                                potential = controller.fem(src_r)

                                if fem.FRACTION_OF_SPACE >= 0.125 and not completed_3D:
                                    sampling_time_3D = controller.sample(src_r, potential)

                                if not completed_2D:
                                    sampling_time_2D = controller_2D.sample(src_r, potential, fem)

                                logger.info(
                                    'Point r={}, (deg={}): {}\t({fem.iterations}, {time}, {sampling})'.format(
                                        src_r,
                                        controller.degree,
                                        'SUCCEED' if potential is not None else 'FAILED',
                                        fem=fem,
                                        time=fem.local_preprocessing_time.duration + fem.solving_time.duration,
                                        sampling=sampling_time_2D + sampling_time_3D))

                                if float(unsaved_time) > 10 * float(save_stopwatch):
                                    with save_stopwatch:
                                        if fem.FRACTION_OF_SPACE >= 0.125:
                                            logging.info('Saving 3D')
                                            controller.save(controller.path + str(tmp_mark))

                                        logging.info('Saving 2D')
                                        controller_2D.save(controller_2D.path + str(tmp_mark))

                                    unsaved_time.reset()
                                    tmp_mark = 1 - tmp_mark
