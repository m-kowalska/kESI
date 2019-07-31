#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:20:10 2019

@author: mkowalska
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

import time

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy.special import lpmv, erf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import griddata
from matplotlib import colors, gridspec
from kcsd import csd_profile as CSD
from kcsd import KCSD3D, ValidateKCSD3D, oKCSD3D
import common


def evd_kCSD(ele_pos, pots, n_src, h=50, sigma=1, R_init=0.23):
    obj = KCSD3D(ele_pos, pots, h=h, sigma=sigma,
                 xmin=0, xmax=1, ymax=1, ymin=0, zmin=0, zmax=1, R_init=R_init,
                 n_src_init=n_src, src_type='gauss',
                 ext_x=0, ext_y=0, ext_z=0,
                 gdx=0.01, gdy=0.01, gdz=0.01)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(obj.k_pot +
                                                   obj.lambd *
                                                   np.identity
                                                   (obj.k_pot.shape[0]))
    except LinAlgError:
        raise LinAlgError('EVD is failing - try moving the electrodes'
                          'slightly')
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    return obj, eigenvalues, eigenvectors


def evd_okCSD(ele_pos, pots, own_src, own_est, n_src, h=50, sigma=1,
              R_init=0.23):
    print('inside')
    obj = oKCSD3D(ele_pos, pots, own_src=own_src, own_est=own_est, h=h,
                  sigma=sigma, src_type='gauss')
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(obj.k_pot +
                                                   obj.lambd *
                                                   np.identity
                                                   (obj.k_pot.shape[0]))
    except LinAlgError:
        raise LinAlgError('EVD is failing - try moving the electrodes'
                          'slightly')
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    return obj, eigenvalues, eigenvectors


def eigensources(ele_pos, pots, n_src, h=50, sigma=1, R_init=0.23):
    csd_at = np.mgrid[0.:1.:100j,
                      0.:1.:100j,
                      0.:1.:100j]
    x, y, z = csd_at
    obj, eigenvalues, eigenvectors = evd_kCSD(ele_pos, pots, n_src, h, sigma,
                                              R_init)
    v = np.dot(obj.k_interp_cross, eigenvectors)
    a = obj.process_estimate(v[:, 0])
    plt.figure()
    plt.contourf(x[:, :, 49], y[:, :, 49], a[:, :, 49, 0], cmap=cm.bwr)
    plt.show()
    return v, a


def eigensources_okCSD(ele_pos, pots, own_src, own_est, n_src, h=50, sigma=1,
                       R_init=0.23):
    obj, eigenvalues, eigenvectors = evd_okCSD(ele_pos, pots, own_src=own_src,
                                               own_est=own_est,
                                               n_src=n_src, h=h, sigma=sigma,
                                               R_init=R_init)
    print(eigenvalues)
    v = np.dot(obj.k_interp_cross, eigenvectors)
    a = obj.process_estimate(v[:, 0])
    print('a', a.shape)
    plt.figure()
    plt.contourf(x.get_values()[:], y.get_values()[:],
                 a[:], cmap=cm.bwr)
    plt.show()
    return


class GaussianSourceBase(object):
    def __init__(self, ROW):
        self._sigma2 = ROW.SIGMA ** 2
        self._a = (2 * np.pi * self._sigma2) ** -1.5
        self.y = ROW.R * np.sin(ROW.ALTITUDE)
        r = ROW.R * np.cos(ROW.ALTITUDE)
        self.x = r * np.sin(ROW.AZIMUTH)
        self.z = r * np.cos(ROW.AZIMUTH)
        self._ROW = ROW

    def __getattr__(self, name):
        return getattr(self._ROW, name)


class GaussianSurceFEM(GaussianSourceBase):
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


class GaussianSourceKCSD3D(GaussianSourceBase):
    CONDUCTIVITY = 1
    _b = 0.25 / (np.pi * CONDUCTIVITY)

    def __init__(self, ROW):
        super(GaussianSourceKCSD3D, self).__init__(ROW)
        self._c = np.sqrt(0.5) / ROW.SIGMA

    def csd(self, X, Y, Z):
        return self._a * np.exp(-0.5 * ((X - self.x) ** 2 + (Y - self.y) ** 2 + (Z - self.z) ** 2)/self._sigma2)

    def potential(self, electrodes):
        R = np.sqrt((electrodes.X - self.x) ** 2 + (electrodes.Y - self.y) ** 2 + (electrodes.Z - self.z) ** 2)
        return self._b * (erf(R * self._c) / R)



filename = '/home/mkowalska/Marta/ForMarta/proof_of_concept_fem_dirchlet_newman_CTX_deg_3.npz'
print(f'loading {filename}...')
fh = np.load(filename)
ELECTRODES = fh['ELECTRODES']
ELECTRODE_NAMES = [f'E{i + 1:03d}' for i in range(ELECTRODES.shape[1])]
ELECTRODES = pd.DataFrame(ELECTRODES.T, columns=['X', 'Y', 'Z'], index=ELECTRODE_NAMES)
POTENTIAL = pd.DataFrame(fh['POTENTIAL'], columns=ELECTRODES.index)
for k in ['SIGMA', 'R', 'ALTITUDE', 'AZIMUTH',]:
    POTENTIAL[k] = fh[k]
GND_ELECTRODES = ELECTRODES.index[(POTENTIAL[ELECTRODES.index] == 0).all()]
RECORDING_ELECTRODES = ELECTRODES.index[(POTENTIAL[ELECTRODES.index] != 0).any()]
CLS = [(GaussianSourceKCSD3D, ELECTRODES.loc[RECORDING_ELECTRODES]),
       (GaussianSurceFEM, RECORDING_ELECTRODES),
       ]
DF = []
for cls, electrodes in CLS:
    print(cls.__name__)
POTENTIAL['CONDUCTIVITY'] = 1. / 300

sources = [cls(ROW) for _, ROW in POTENTIAL[POTENTIAL.SIGMA <= 1].iterrows()]
y = POTENTIAL.R * np.sin(POTENTIAL.ALTITUDE)
r = POTENTIAL.R * np.cos(POTENTIAL.ALTITUDE)
x = r * np.sin(POTENTIAL.AZIMUTH)
z = r * np.cos(POTENTIAL.AZIMUTH)
own_src = np.array([x.get_values(), y.get_values(), z.get_values()])

n_src_init = 1000
ELE_LIMS = [0., 1.]  # range of electrodes space

est_at = np.mgrid[-9:9:100j,
                  -9:9:100j,
                  -9:9:100j]
X, Y, Z = est_at
R = np.ones(len(X.flatten()))*9
R2 = np.ones(len(X.flatten()))*7
mask = np.where(((est_at[0].flatten()**2 + est_at[1].flatten()**2 + est_at[2].flatten()**2) <= R**2) & ((est_at[0].flatten()**2 + est_at[1].flatten()**2 + est_at[2].flatten()**2) >= R2**2))
own_est = np.array([est_at[0].flatten()[mask], est_at[2].flatten()[mask], est_at[2].flatten()[mask]])
ele_pos = np.array([ELECTRODES.X, ELECTRODES.Y, ELECTRODES.Z])
ele_pos = ele_pos.T

sources = [common.ElectrodeAwarePolarGaussianSourceKCSD3D(ROW, ELECTRODES) for _, ROW in POTENTIAL.iterrows()]
W = np.random.normal(size=len(sources))
GT = sum(s.csd(*own_est) for w, s in zip(W, sources))


print('Checking 3D')
CSD_PROFILE = CSD.gauss_3d_small
CSD_SEED = 20  # 0-49 are small sources, 50-99 are large sources
h = 50
sigma = 1
R_init = 0.23
TIC = time.time()
KK = ValidateKCSD3D(CSD_SEED, h=50, sigma=1)
#pots = KK.calculate_potential(GT, own_est, ele_pos, h, sigma)

pots = np.zeros(ele_pos.shape[0])
xlin = own_est[0, :]
ylin = own_est[1, :]
zlin = own_est[2, :]
for ii in range(ele_pos.shape[0]):
    pots[ii] = KK.integrate(own_est, GT,
                            [ele_pos[ii][0], ele_pos[ii][1],
                             ele_pos[ii][2]], h, [xlin, ylin, zlin])
pots /= 4*np.pi*sigma


#print('1')
#ele_pos, pots = KK.electrode_config(CSD_PROFILE, CSD_SEED, total_ele=125,
#                                    ele_lims=ELE_LIMS, h=50., sigma=1.)
#print('2')
#v, a = eigensources(ele_pos, pots, n_src_init, h, sigma, R_init)
#
#v, a = eigensources_okCSD(ele_pos, pots, own_src, own_src, n_src_init, h, sigma, R_init)

#data = np.load('/home/mkowalska/Marta/ForMarta/proof_of_concept_fem_dirchlet_newman_CTX_deg_3.npz')
#SIGMA = data['SIGMA']
#R = data['R']
#altitude = data['ALTITUDE']
#azimuth = data['AZIMUTH']
#potential = data['POTENTIAL']
#electrodes = data['ELECTRODES']
