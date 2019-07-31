# $ docker run -ti --env HOST_UID=$(id -u) --env HOST_GID=$(id -g) -v $(pwd):/home/fenics/shared:Z quay.io/fenicsproject/stable
# fenics@...$ cd /home/fenics/shared/

import numpy as np
import gc
from dolfin import Constant, Mesh, MeshFunction, FunctionSpace, TestFunction, TrialFunction, Function, Measure, inner, grad, assemble, Point, PointSource, KrylovSolver, info
from dolfin import Expression, DirichletBC, File, Vector, interpolate, XDMFFile, UserExpression
import collections

from datetime import datetime


sigma_B = 1. / 300.  # S / cm
sigma_brain = Constant(sigma_B)
sigma_scalp = Constant(sigma_B)
sigma_csf = Constant(sigma_B) #Constant(5 * sigma_B)
sigma_skull = Constant(sigma_B) #Constant(sigma_B / 20.)

whitemattervol1 = 32
whitemattervol2 = 64
graymattervol = 96
csfvol = 128
skullvol = 160
scalp = 192

BRAIN_R = 7.9
SCALP_R = 9.0
WHITE_R = 7.5
RAD_TOL = 0.01
NECK_ANGLE = -np.pi / 3
NECK_AT = BRAIN_R * np.sin(NECK_ANGLE)


def extract_pots(phi, positions):
    compt_values = np.zeros(positions.shape[0])
    for ii in range(positions.shape[0]):
        compt_values[ii] = phi(positions[ii, :])
    return compt_values


def extract_csd(csd, positions):
    compt_values = np.zeros(positions.shape[0])
    for ii in range(positions.shape[0]):
        compt_values[ii] = csd(positions[:, ii])
    return compt_values


src_pos = np.load('src_pos.npy')
X, Y, Z = src_pos



PATH = '_meshes/sphere_6_higherres'
mesh = Mesh(PATH + '.xml')
# lowres: 5.1s
subdomains = MeshFunction("size_t", mesh, PATH + '_physical_region.xml')
# lowres: 1.4s
boundaries = MeshFunction("size_t", mesh, PATH + '_facet_region.xml')
# lowres: 12s

DEGREE = 1
V = FunctionSpace(mesh, "CG", DEGREE)
# lowres: 42s (first time: 58s)
v = TestFunction(V)
# lowres << 1s
potential_trial = TrialFunction(V)
# lowres << 1s
potential = Function(V)
# lowres < 1s

dx = Measure("dx")(subdomain_data=subdomains)
# lowres << 1s
ds = Measure("ds")(subdomain_data=boundaries)
# lowres << 1s
a = inner(sigma_brain * grad(potential_trial), grad(v)) * dx(whitemattervol1) + \
    inner(sigma_brain * grad(potential_trial), grad(v)) * dx(whitemattervol2) + \
    inner(sigma_brain * grad(potential_trial), grad(v)) * dx(graymattervol) + \
    inner(sigma_scalp * grad(potential_trial), grad(v)) * dx(scalp) + \
    inner(sigma_csf * grad(potential_trial), grad(v)) * dx(csfvol) + \
    inner(sigma_skull * grad(potential_trial), grad(v)) * dx(skullvol)
# lowres < 1s
TERMS_WITH_UNKNOWN = assemble(a)
# lowres: 120s
solver = KrylovSolver("cg", "ilu")
solver.parameters["maximum_iterations"] = 2000
solver.parameters["absolute_tolerance"] = 1E-8

class CartesianBase(object):
    def __init__(self, ROW):
        self.init(ROW.X, ROW.Y, ROW.Z, ROW)


class GaussianSourceBase(object):
    def init(self, x, y, z, ROW):
        self.x = x
        self.y = y
        self.z = z
        self._sigma2 = ROW.SIGMA ** 2
        self._a = (2 * np.pi * self._sigma2) ** -1.5
        self._ROW = ROW

    def __getattr__(self, name):
        return getattr(self._ROW, name)


class GaussianSourceFEM(GaussianSourceBase):
    _BRAIN_R = 7.9
    NECK_ANGLE = -np.pi / 3
    NECK_AT = _BRAIN_R * np.sin(NECK_ANGLE)

    def csd(self, X, Y, Z):
        DIST2 = (X*X + Y*Y + Z*Z)
        return np.where((DIST2 <= self._BRAIN_R ** 2) & (Y > self.NECK_AT),
                        self._a * np.exp(-0.5 * ((X - self.x) ** 2 + (Y - self.y) ** 2 + (Z - self.z) ** 2)/self._sigma2),
                        0)

    def potential(self, electrodes):
        return self._ROW.loc[electrodes]

class CartesianGaussianSourceFEM(CartesianBase, GaussianSourceFEM):
    pass


DummyRow = collections.namedtuple('DummyRow',
                                  ['X', 'Y', 'Z', 'SIGMA'])

class MyFunctionExpression(UserExpression):
    def __init__(self, f, norm=1, *args, **kwargs):
        super(MyFunctionExpression,
                      self).__init__(*args, **kwargs)
        self.__f = f
        self.__norm = norm
        

    def eval(self, values, x):
        values[0] = self.__norm * self.__f(*x)

est_pos = np.load('estm_pos_6_higher.npy').T
est_x, est_y, est_z = est_pos


#TMP_FILENAME = f'proof_of_concept_fem_dirchlet_newman_CTX_deg_1_6_higherres_kESI_ele.npz'
#try:
#    fh = np.load(TMP_FILENAME)
#
#except FileNotFoundError:
#    print('no previous results found')
#    previously_solved = set()
#
#else:
#    SIGMA = fh['SIGMA']
#    R = fh['R']
#    X = fh['X']
#    Y = fh['Y']
#    Z = fh['Z']
#    POTENTIAL = fh['POTENTIAL']
#
#    for s, r, pot in zip(SIGMA, R, POTENTIAL):
#        row = [s, r]
#        row.extend(pot)
#        SOURCES.append(row)
#
#    previously_solved = set(zip(SIGMA, R, X, Y, Z))




def boundary(x, on_boundary):
    return x[1] <= NECK_AT

#L = csd * v * dx
#known_terms = assemble(L)
bc = DirichletBC(V, Constant(0.), boundary)
#terms_with_unknown = TERMS_WITH_UNKNOWN.copy()
#bc.apply(terms_with_unknown, known_terms)
#
#


ele_x, ele_y, ele_z = np.mgrid[0.:1.:5j,
                               -0.2:0.8:5j,
                               6.7:7.7:5j]
ele_pos = np.array([ele_x.flatten(), ele_y.flatten(), ele_z.flatten()])
ele_coords = ele_pos
print(ele_coords.shape)

#vals_pots = extract_pots(potential, ele_coords.T)
#np.save('pots_6_higher_kESI.npy', vals_pots)
##est_pos = np.load('estm_pos.npy')
#vals_csd = extract_csd(csd, est_pos)
#np.save('csd_6_higher_kESI.npy', vals_csd)


#TMP_FILENAME = f'proof_of_concept_fem_dirchlet_newman_CTX_deg_{DEGREE}_6_higherres_cube_source3_ele.npz'


def f():
    global SOURCES, DBG, previously_solved, solver



    for x, y, z in zip(X, Y, Z):
        sigma = 0.1

        
#        csd = MyFunctionExpression(CartesianGaussianSourceFEM(DummyRow(x,y,z,sigma)).csd)
        
        csd = Expression(f'''
                  x[0]*x[0] + x[1]*x[1] + x[2]*x[2] <= {BRAIN_R**2} && x[1] > {NECK_AT}?
                  A * exp(-(
                             (x[0] - source_x)*(x[0] - source_x) 
                           + (x[1] - source_y)*(x[1] - source_y)
                           + (x[2] - source_z)*(x[2] - source_z))
                          /(2*sigma_2)):
                  0
                  ''',
                  source_z=z,
                  source_y=y,
                  source_x=x,
                  sigma_2=sigma**2,
                  A=(2 * np.pi * sigma**2) ** -1.5,
                  degree=DEGREE)

        L = csd * v * dx
        # lowres: 1.1s (first time: 3.3s)
        known_terms = assemble(L)
        # lowres: 8.3s (first time: 9.5s)
        terms_with_unknown = TERMS_WITH_UNKNOWN.copy()
        bc.apply(terms_with_unknown, known_terms)


#        solver.parameters["monitor_convergence"] = True
#        solver.solve(terms_with_unknown, potential.vector(), known_terms)

        gc.collect()
#        print(f'{sigma:.2f}\t{x:.3f}\t{y:.3f}\t{z:.3f}')
        try:
            start = datetime.now()
            iterations = solver.solve(terms_with_unknown, potential.vector(), known_terms)
            time = datetime.now() - start
            # lowres: 1300 s
            # : 4900 s
        except RuntimeError as e:
            print(f'{sigma:.2f}\t{x:.3f}\t{y:.3f}\t{z:.3f}\tFAILED')
            continue



        ELECTRODES = ele_coords
        row = [iterations, sigma, x, y, z, time.total_seconds()]
        row.extend(potential(*loc) for loc in ELECTRODES.T)
        SOURCES.append(row)

        SRC = np.array([row[1:] for row in SOURCES])
        ITERATIONS = np.array([row[0] for row in SOURCES])
        np.savez_compressed(f'proof_of_concept_fem_dirchlet_newman_CTX_deg_1_6_higherres_kESI_ele_quicker.npz',
                            SIGMA=SRC[:, 0],
                            X=SRC[:, 1],
                            Y=SRC[:, 2],
                            Z=SRC[:, 3],
                            TIME=SRC[:, 4],
                            POTENTIAL=SRC[:, 5:],
                            MAX_ITER=ITERATIONS,
                            ELECTRODES=ele_coords)
        gc.collect()

SOURCES = []
DBG = {}


start = datetime.now()          
f()
print(datetime.now() - start)
