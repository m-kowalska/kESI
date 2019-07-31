# $ docker run -ti --env HOST_UID=$(id -u) --env HOST_GID=$(id -g) -v $(pwd):/home/fenics/shared:Z quay.io/fenicsproject/stable
# fenics@...$ cd /home/fenics/shared/

import numpy as np
import gc
from dolfin import Constant, Mesh, MeshFunction, FunctionSpace, TestFunction, TrialFunction, Function, Measure, inner, grad, assemble, Point, PointSource, KrylovSolver, info
from dolfin import Expression, DirichletBC, File, Vector, interpolate, XDMFFile, UserExpression

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
        compt_values[ii] = csd(positions[ii, :])
    return compt_values

#ele_coords = []

for altitude in np.linspace(NECK_ANGLE, np.pi / 2, 16):
    for azimuth in np.linspace(0, 2 * np.pi, int(round(np.cos(altitude) * 36 / np.pi))+1,
                               endpoint=False):
        r = SCALP_R - RAD_TOL
#        ele_coords.append(r * np.array([np.cos(altitude) * np.sin(azimuth),
#                                        np.sin(altitude),
#                                        np.cos(altitude) * np.cos(azimuth)]))

# ele_coords.append(np.array([0, 0, BRAIN_R]))
# ele_coords.append(np.array([np.nan] * 3))
#ele_coords = np.transpose(ele_coords)
ele_x, ele_y, ele_z = np.mgrid[0.:1.:5j,
                               -0.2:0.8:5j,
                               6.7:7.7:5j]
ele_pos = np.array([ele_x.flatten(), ele_y.flatten(), ele_z.flatten()])
ele_coords = ele_pos
print(ele_coords.shape)

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


def get_states_3D(seed):
    """
    Used in the random seed generation for 3D sources
    """
    rstate = np.random.RandomState(seed)  # seed here!
    states = rstate.random_sample(24)
    return states


def gauss_3d_small(csd_at):
    '''A random quadpole small souce in 3D'''
    x, y, z = csd_at
    if (x**2 + y**2 + z**2 <= BRAIN_R**2 and y > NECK_AT):
#        states = get_states_3D(seed)
        x0, y0, z0 = [0.1, 0., 7.35]
        x1, y1, z1 = [0.1, 0., 7.25]
        sig_2 = 0.01
#        p1, p2, p3 = (ii*0.5 for ii in states[8:11])
        A = (2*np.pi*sig_2)**-1
        f1 = A*np.exp((-(x-x0)**2 - (y-y0)**2 - (z-z0)**2) / (2*sig_2))
        f2 = -1*A*np.exp((-(x-x1)**2 - (y-y1)**2 - (z-z1)**2) / (2*sig_2))
        x2, y2, z2 = [0.9, 0., 7.35]
        x3, y3, z3 = [0.9, 0., 7.25]
        f3 = -A*np.exp((-(x-x2)**2 - (y-y2)**2 - (z-z2)**2) / (2*sig_2))
        f4 = 1*A*np.exp((-(x-x3)**2 - (y-y3)**2 - (z-z3)**2) / (2*sig_2))
        f = f1+f2+f3+f4
        return f
    else:
        return 0


class MyFunctionExpression(UserExpression):
    def __init__(self, f, norm=1, *args, **kwargs):
        super(MyFunctionExpression,
                      self).__init__(*args, **kwargs)
        self.__f = f
        self.__norm = norm
        

    def eval(self, values, x):
        values[0] = self.__norm * self.__f(x)


csd = MyFunctionExpression(gauss_3d_small, degree=DEGREE)
est_pos = np.load('estm_pos_6_higher.npy')
csd = MyFunctionExpression(gauss_3d_small,
                           norm=1./np.abs(extract_csd(csd, est_pos)).max(),
                           degree=DEGREE)

#csd = Expression(f'''
#                  x[0]*x[0] + x[1]*x[1] + x[2]*x[2] <= {BRAIN_R**2} && x[1] > {NECK_AT}?
#                  (A * exp((-(x[0]-source_x)*(x[0]-source_x) - (x[1] - source_y)*(x[1] - source_y) - (x[2] - source_z)*(x[2] - source_z))/(2*sig_2)) +
#                  -A * exp((-(x[0]-source_x1)*(x[0]-source_x1) - (x[1] - source_y1)*(x[1] - source_y1) - (x[2] - source_z1)*(x[2] - source_z1))/(2*sig_2)) +
#                  -A * exp((-(x[0]-source_x2)*(x[0]-source_x2) - (x[1] - source_y2)*(x[1] - source_y2) - (x[2] - source_z2)*(x[2] - source_z2))/(2*sig_2)) +
#                  A * exp((-(x[0]-source_x3)*(x[0]-source_x3) - (x[1] - source_y3)*(x[1] - source_y3) - (x[2] - source_z3)*(x[2] - source_z3))/(2*sig_2)))/
#                  (max):
#                  0
#                  ''',
#                  source_z=7.35,
#                  source_y=0,
#                  source_x=0,
#                  source_z1=7.25,
#                  source_y1=0,
#                  source_x1=0,
#                  source_z2=7.35,
#                  source_y2=0,
#                  source_x2=1.5,
#                  source_z3=7.25,
#                  source_y3=0,
#                  source_x3=1.5,
#                  sig_2=0.1,
#                  A=(2 * np.pi * 0.1) ** -1,
#                  degree=DEGREE)


def boundary(x, on_boundary):
    return x[1] <= NECK_AT

L = csd * v * dx
known_terms = assemble(L)
bc = DirichletBC(V, Constant(0.), boundary)
terms_with_unknown = TERMS_WITH_UNKNOWN.copy()
bc.apply(terms_with_unknown, known_terms)


solver = KrylovSolver("cg", "ilu")
solver.parameters["maximum_iterations"] = 1100
solver.parameters["absolute_tolerance"] = 1E-8
#solver.parameters["monitor_convergence"] = True
solver.solve(terms_with_unknown, potential.vector(), known_terms)

vals_pots = extract_pots(potential, ele_coords.T)
np.save('pots_6_higher.npy', vals_pots)
#est_pos = np.load('estm_pos.npy')
vals_csd = extract_csd(csd, est_pos)
np.save('csd_6_higher.npy', vals_csd)

SOURCES = []
DBG = {}

TMP_FILENAME = f'proof_of_concept_fem_dirchlet_newman_CTX_deg_{DEGREE}_6_higherres_cube_source3_ele.npz'
#TMP_FILENAME = f'proof_of_concept_fem_dirchlet_newman_CTX_deg_1_6_lowres_cube_ele.npz'
try:
    fh = np.load(TMP_FILENAME)

except FileNotFoundError:
    print('no previous results found')
    previously_solved = set()

else:
    SIGMA = fh['SIGMA']
    R = fh['R']
    ALTITUDE = fh['ALTITUDE']
#    AZIMUTH = fh['AZIMUTH']
    POTENTIAL = fh['POTENTIAL']

    for s, r, pot in zip(SIGMA, R, POTENTIAL):
        row = [s, r]
        row.extend(pot)
        SOURCES.append(row)

    previously_solved = set(zip(SIGMA, R, ALTITUDE))

def f():
    global SOURCES, DBG, previously_solved

    for sigma in np.logspace(1, -0.5, 4):
        csd.sigma_2 = sigma ** 2
        csd.A = (2 * np.pi * sigma ** 2) ** -1.5
        for z in np.linspace(WHITE_R, BRAIN_R, int(round((BRAIN_R - WHITE_R) / sigma)) + 1):
            for altitude in np.linspace(0, np.pi/2, int(round(np.pi/2 * z / sigma)) + 2):
                if (sigma, z, altitude) in previously_solved:
                    print(f'{sigma:.2f}\t{z:.3f}\t{int(round(altitude * 180/np.pi)):d}\tALREADY SOLVED: SKIPPING')
                    continue

                csd.source_z = z * np.cos(altitude)
                csd.source_y = z * np.sin(altitude)

#                A_inv = assemble(csd * Measure('dx', mesh))
#                if A_inv <= 0:
#                    print(f'{sigma:.2f}\t{z:.3f}\t{int(round(altitude * 180/np.pi)):d}\tFAILED MISERABLY ({A_inv:g} <= 0)')
#                    DBG[sigma, z, altitude] = ('FAILED MISERABLY', A_inv)
#                    continue

                L = csd * v * dx
                # lowres: 1.1s (first time: 3.3s)
                known_terms = assemble(L)
                # lowres: 8.3s (first time: 9.5s)
                terms_with_unknown = TERMS_WITH_UNKNOWN.copy()
                bc.apply(terms_with_unknown, known_terms)

                gc.collect()
                print(f'{sigma:.2f}\t{z:.3f}\t{int(round(altitude * 180/np.pi)):d}')
                try:
                    solver.solve(terms_with_unknown, potential.vector(), known_terms)
                    # lowres: 1300 s
                    # : 4900 s
                except RuntimeError as e:
                    print(f'{sigma:.2f}\t{z:.3f}\t{int(round(altitude * 180/np.pi)):d}\tFAILED')
#                    DBG[sigma, z, altitude] = ('FAILED', A_inv)
                    continue

                # ELE_ALT = np.dot([[1, 0, 0],
                #                   [0, np.cos(-altitude), -np.sin(-altitude)],
                #                   [0, np.sin(-altitude), np.cos(-altitude)]],
                #                  ele_coords)
                ELE_ALT = np.array(ele_coords)
#                DBG[sigma, z, altitude] = ('SUCCEEDED', A_inv)
                for azimuth in np.linspace(0, 2*np.pi, int(round(2 * np.pi * np.cos(altitude) * z / sigma)) + 2, endpoint=False):
                    #print(f'{sigma}\t{z}\t{altitude}\t{azimuth}')
                    ELECTRODES = np.dot([[np.cos(-azimuth), 0, np.sin(-azimuth)],
                                         [0, 1, 0],
                                         [-np.sin(-azimuth), 0, np.cos(-azimuth)]],
                                        ELE_ALT)
                    # ELECTRODES[:, -1] = [0, 0, z]
                    row = [sigma, z, altitude, azimuth]
                    row.extend(potential(*loc) for loc in ELECTRODES.T)
                    SOURCES.append(row)

                SRC = np.array(SOURCES)
                np.savez_compressed(f'proof_of_concept_fem_dirchlet_newman_CTX_deg_{DEGREE}_6_higherres_cube_source3_ele.npz',
                                    SIGMA=SRC[:, 0],
                                    R=SRC[:, 1],
                                    ALTITUDE=SRC[:, 2],
#                                    AZIMUTH=SRC[:, 3],
                                    POTENTIAL=SRC[:, 4:],
                                    ELECTRODES=ele_coords)
                gc.collect()

    SRC = np.array(SOURCES)
    np.savez_compressed(f'proof_of_concept_fem_dirchlet_newman_CTX_deg_{DEGREE}_6_higherres_cube_source3_ele.npz',
                        SIGMA=SRC[:, 0],
                        R=SRC[:, 1],
                        ALTITUDE=SRC[:, 2],
#                        AZIMUTH=SRC[:, 3],
                        POTENTIAL=SRC[:,4:],
                        ELECTRODES=ele_coords)

#f()