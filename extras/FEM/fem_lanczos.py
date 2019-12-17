# $ docker run -ti --env HOST_UID=$(id -u) --env HOST_GID=$(id -g) -v $(pwd):/home/fenics/shared:Z quay.io/fenicsproject/stable
# fenics@...$ cd /home/fenics/shared/

import os
import logging

import numpy as np

import _fem_common


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


SAMPLING_FREQUENCY = 5


class LanczosSourceFactory(_fem_common._SymmetricSourceFactory_Base):
    def load_specific_attributes(self, fh):
        self.n = fh['folds']

    def solution_array_name(self, degree):
        return 'Lanczos_{}'.format(degree)

    def _lanczos(self, X):
        return np.where(abs(X) >= self.n,
                        0,
                        np.sinc(X) * np.sinc(X / self.n))

    def csd(self, X, Y, Z):
        return self._lanczos(X) * self._lanczos(Y) * self._lanczos(Z) * self.a

    def __call__(self, x=0, y=0, z=0, scale=1, conductivity=1):
        return self._Source(scale,
                            conductivity,
                            x, y, z,
                            self)


if __name__ == '__main__':
    import sys
    try:
        from dolfin import Expression

    except (ModuleNotFoundError, ImportError):
        print("""Run docker first:
        $ docker run -ti --env HOST_UID=$(id -u) --env HOST_GID=$(id -g) -v $(pwd):/home/fenics/shared:Z quay.io/fenicsproject/stable
        $ cd /home/fenics/shared/
        """)
    else:
        class LanczosPotentialFEM(_fem_common._SymmetricFEM_Base):
            def __init__(self, degree=1, mesh_name='eighth_of_sphere'):
                         super(LanczosPotentialFEM, self).__init__(
                               degree=degree,
                               mesh_path=os.path.join(_fem_common.DIRNAME,
                                                      'meshes',
                                                      mesh_name))

            def _make_csd(self, degree, n):
                return Expression(f'''
                    x[0] >= n || x[1] >= n || x[2] >= n ?
                     0 :
                     a * (x[0] < {np.finfo(np.float32).eps} ? 1 : sin({np.pi} * x[0]) * sin({np.pi} * x[0] / n) / (x[0] * x[0] * {np.pi ** 2} / n))
                     * (x[1] < {np.finfo(np.float32).eps} ? 1 : sin({np.pi} * x[1]) * sin({np.pi} * x[1] / n) / (x[1] * x[1] * {np.pi ** 2} / n))
                     * (x[2] < {np.finfo(np.float32).eps} ? 1 : sin({np.pi} * x[2]) * sin({np.pi} * x[2] / n) / (x[2] * x[2] * {np.pi ** 2} / n))
                    ''',
                                 n=n,
                                 degree=degree,
                                 a=1.0)

            def potential_behind_dome(self, radius, n):
                return 0.25 / radius / np.pi

        logging.basicConfig(level=logging.INFO)

        if not os.path.exists(_fem_common.SOLUTION_DIRECTORY):
            os.makedirs(_fem_common.SOLUTION_DIRECTORY)

        for mesh_name in sys.argv[1:]:
            fem = LanczosPotentialFEM(mesh_name=mesh_name)
            N = 1 + int(np.ceil(fem.RADIUS))

            for n in [1, 2, 3]:
                solution_filename = '{}_lanczos_{}.npz'.format(mesh_name,
                                                               n)
                stats = []
                results = {'N': N,
                           'folds': n,
                           'STATS': stats,
                           'radius': fem.RADIUS,
                           'sampling_frequency': SAMPLING_FREQUENCY,
                           }
                for degree in [1, 2, 3]:
                    logger.info('Lanczos{} (deg={})'.format(n, degree))
                    potential = fem(degree, n)

                    stats.append((degree,
                                  potential is not None,
                                  fem.iterations,
                                  fem.time.total_seconds()))
                    logger.info('Lanczos{} (deg={}): {}'.format(n, degree,
                                                                'SUCCEED' if potential is not None else 'FAILED'))
                    if potential is not None:
                        N_LIMIT = (N - 1) * SAMPLING_FREQUENCY + 1 # TODO: prove correctness
                        POTENTIAL = np.empty(N_LIMIT * (N_LIMIT + 1) * (N_LIMIT + 2) // 6)
                        POTENTIAL.fill(np.nan)
                        for x in range(N_LIMIT):
                            for y in range(x + 1):
                                for z in range(y + 1):
                                    idx = x * (x + 1) * (x + 2) // 6 + y * (
                                                y + 1) // 2 + z
                                    xx = x / float(SAMPLING_FREQUENCY)
                                    yy = y / float(SAMPLING_FREQUENCY)
                                    zz = z / float(SAMPLING_FREQUENCY)
                                    r = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
                                    if r >= fem.RADIUS:
                                        v = fem.potential_behind_dome(r, n)
                                    else:
                                        try:
                                            v = potential(xx, yy, zz)
                                        except RuntimeError as e:
                                            logger.warning("""
                                    potential({}, {}, {})
                                    (r = {})
                                    raised:
                                    {}""".format(xx, yy, zz, r, e))
                                            v = fem.potential_behind_dome(r, n)
                                    POTENTIAL[idx] = v
                        results['Lanczos_{}'.format(degree)] = POTENTIAL
                        results['A_{}'.format(degree)] = fem.a
                        np.savez_compressed(LanczosSourceFactory.solution_path(solution_filename),
                                            **results)
