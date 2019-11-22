#!/usr/bin/env python
# encoding: utf-8
###############################################################################
#                                                                             #
#    kESI                                                                     #
#                                                                             #
#    Copyright (C) 2019 Jakub M. Dzik (Laboratory of Neuroinformatics;        #
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

import numpy as np

from ._engine import (FunctionalFieldReconstructor, LinearMixture,
                      _MissingAttributeError)

class VerboseFFR(FunctionalFieldReconstructor):
    """
    Attributes
    ----------
    number_of_basis
    number_of_electrodes
    probed_basis
    kernel

    Methods
    -------
    get_probed_basis(measurement_manager)
    get_kernel_matrix(measurement_manager)

    Class attributes
    ----------------
    MeasurementManagerBase
    MeasurementManagerHasNoProbeAtSinglePointMethodError

    Notes
    -----
    This class extends its parent, providing access to theoretical concepts
    presented in our publication [1]_, thus some names follow convention used
    therein.

    .. [1] C. Chintaluri et al. (2019) "kCSD-python, a tool for
       reliable current source density estimation" (preprint available at
       `bioRxiv <https://www.biorxiv.org/content/10.1101/708511v1>`)
       doi: 10.1101/708511
    """
    class MeasurementManagerBase(FunctionalFieldReconstructor.MeasurementManagerBase):
        def probe_at_single_point(self, field, *args, **kwargs):
            """
            Probe the field at single point.

            An abstract method implementing  probing of the appropriate basis
            function at appropriate point.

            Parameters
            ----------
            field : object
                An object which implements corresponding basis functions.
            args, kwargs
                Description of the point.

            Returns
            -------
            float
                Value of the field at the point.

            Notes
            -----
            It is possible, that vector returntypes will be allowed
            """
            raise NotImplementedError

    class MeasurementManagerHasNoProbeAtSinglePointMethodError(_MissingAttributeError):
        _missing = 'probe_at_single_point'

    _mm_validators = (FunctionalFieldReconstructor._mm_validators
                      + [MeasurementManagerHasNoProbeAtSinglePointMethodError])

    @property
    def probed_basis(self):
        r"""
        A matrix of basis functions (rows) probed at measurement points
        (columns).

        The matrix is not normalized.

        Returns
        -------
        probed_basis : numpy.ndarray
            The matrix as number_of_basis x number_of_electrodes numpy array.

        See also
        --------
        number_of_basis, number_of_electrodes

        Notes
        -----
        The measurement manager may affect the returned value.

        The matrix is denormalized as parental class uses normalized one.
        Thus it may be affected by numerical errors.

        `probed_basis[:, i]` is :math:`\Phi(x_i)` (see eq. 16 and above
        in [1]_), where :math:`x_i` is the i-th measurement point(electrode).

        .. [1] C. Chintaluri et al. (2019) "kCSD-python, a tool for
           reliable current source density estimation" (preprint available at
           `bioRxiv <https://www.biorxiv.org/content/10.1101/708511v1>`)
           doi: 10.1101/708511
        """
        return self._pre_kernel * self.number_of_basis

    @property
    def kernel(self):
        r"""
        The kernel matrix.

        Returns
        -------
        kernel : numpy.ndarray
            The kernel matrix as an number_of_electrodes x number_of_electrodes numpy array.

        See also
        --------
        number_of_electrodes, number_of_basis, probed_basis

        Notes
        -----
        The measurement manager may affect the returned value.

        The kernel matrix is a normalized matrix :math:`K` defined in [1]_
        (see eq. 16 and 25).

        `kernel == K / number_of_basis`
        `K == probed_basis.T @ probed_basis`

        .. [1] C. Chintaluri et al. (2019) "kCSD-python, a tool for
           reliable current source density estimation" (preprint available at
           `bioRxiv <https://www.biorxiv.org/content/10.1101/708511v1>`)
           doi: 10.1101/708511
        """
        return self._kernel

    @property
    def number_of_basis(self):
        r"""
        The number of basis functions.

        Returns
        -------
        int
        """
        return self._pre_kernel.shape[0]

    @property
    def number_of_electrodes(self):
        r"""
        The number of measurement points (electrodes).

        Returns
        -------
        int
        """
        return self._pre_kernel.shape[1]

    def get_probed_basis(self, measurement_manager):
        r"""
        A matrix of basis functions (rows) probed at estimation points
        (columns).

        The matrix is not normalized.

        Parameters
        ----------
        measurement_manager : instance of kesi.MeasurementManagerBase subclass
            The measurement manager is an object implementing `.probe(basis)`
            method, which probes appropriate function related to `basis`
            at appropriate estimation points and returns sequence of values.
            The number of the estimation points is given by its
            `.number_of_measurements` attribute.

        Returns
        -------
        probed_basis : numpy.ndarray
            The matrix as a `measurement_manager.number_of_measurements`
             x `number_of_basis` numpy array.

        See also
        --------
        number_of_basis

        Notes
        -----
        The measurement manager may affect the returned value.

        `probed_basis[:, i]` may be either :math:`\Phi(x_i)`
        or :math:`\tilde{Phi}(x_i)` (see eq. 16  and above in [1]_),
        depending on the measurement manager.
        :math:`x_i` is the i-th estimation point.

        .. [1] C. Chintaluri et al. (2019) "kCSD-python, a tool for
           reliable current source density estimation" (preprint available at
           `bioRxiv <https://www.biorxiv.org/content/10.1101/708511v1>`)
           doi: 10.1101/708511
        """
        probed_basis = np.empty((self.number_of_basis,
                                 measurement_manager.number_of_measurements))
        self._fill_probed_components(probed_basis,
                                     measurement_manager.probe)
        return probed_basis

    def get_kernel_matrix(self, measurement_manager):
        r"""
        The (cross-)kernel matrix.

        Parameters
        ----------
        measurement_manager : instance of kesi.MeasurementManagerBase subclass
            The measurement manager is an object implementing `.probe(basis)`
            method, which probes appropriate function related to `basis`
            at appropriate estimation points and returns sequence of values.
            The number of the estimation points is given by its
            `.number_of_measurements` attribute.

        Returns
        -------
        kernel_matrix : numpy.ndarray
            The (cross-)kernel matrix as a
            `measurement_manager.number_of_measurements`
            x `number_of_electrodes` numpy array.

        See also
        --------
        number_of_electrodes, number_of_basis, probed_basis

        Notes
        -----
        Measurement managers may affect the returned value.

        The cross-kernel matrix (:math:`\tilde{K}`) is defined in [1]_
        (see eq. above 27) analogously to the kernel matrix :math:`K` (eq 25).

        `kernel_matrix == PHI.T @ probed_basis / number_of_basis`,
        where `PHI` is `.get_probed_basis(measurement_manager)`.

        .. [1] C. Chintaluri et al. (2019) "kCSD-python, a tool for
           reliable current source density estimation" (preprint available at
           `bioRxiv <https://www.biorxiv.org/content/10.1101/708511v1>`)
           doi: 10.1101/708511
        """
        # Matrix multiplication not used as matrix returned by
        # .get_probed_basis() may be too big to be stored in available memory.
        # The sum() builtin not used as it uses + operator instead of augmented
        # assignment, thus it may be less memory-efficient than the loop below.
        cross_kernel = 0
        for component, ROW in zip(self._field_components,
                                  self._pre_kernel):
            cross_kernel += np.outer(measurement_manager.probe(component),
                                     ROW)
        return cross_kernel

    def get_kernel_functions(self, *args, **kwargs):
        """
        The (cross-)kernel unary functions.

        Parameters
        ----------
            Anything that passed to by the `.probe_at_single_point()`
            method of the measurement manager (given to the constructor) will
            be interpreted as a single measurement point (:math:`x`).

        Returns
        -------
        object
            The object implements the unary (cross-)kernel functions as its
            methods.

        Notes
        -----
        The unary (cross-)kernel function is a binary (cross-)kernel function
        (:math:`K(y, x)`, see eq. 16 in [1]_) fixed with one leg at :math:`x`.

        .. [1] C. Chintaluri et al. (2019) "kCSD-python, a tool for
           reliable current source density estimation" (preprint available at
           `bioRxiv <https://www.biorxiv.org/content/10.1101/708511v1>`)
           doi: 10.1101/708511
        """
        probe = self._measurement_manager.probe_at_single_point
        return (LinearMixture([(component, probe(component, *args, **kwargs))
                               for component in self._field_components])
                / self.number_of_basis)