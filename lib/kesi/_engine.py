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

import sys

import numpy as np
import warnings


class _MissingAttributeError(TypeError):
    """
    An abstract base class for TypeError object validators.

    Attributes
    ----------
    _missing : str
        The name of the attribute which presence is to be validated.
        A required class attribute of a concrete subclass.
    """
    @classmethod
    def _validate(cls, o):
        """
        Validate the object.

        Parameters
        ----------
            o : object
                The object to be validated.

        Raises
        ------
            cls
                If the object is missing `cls._missing` attribute.
        """
        if not hasattr(o, cls._missing):
            raise cls


class _LinearKernelSolver(object):
    def __init__(self, kernel):
        self._kernel = kernel

    def __call__(self, rhs, regularization_parameter=0):
        return np.linalg.solve(self._kernel + regularization_parameter * np.identity(self._kernel.shape[0]),
                               rhs)


class _EigenvectorKernelSolver(object):
    def __init__(self, kernel):
        self.EIGENVALUES, self.EIGENVECTORS = np.linalg.eigh(kernel)
        self._SCALED_EIGENVECTORS = np.matmul(self.EIGENVECTORS,
                                              np.diag(1. / self.EIGENVALUES))

    def __call__(self, rhs, mask=None):
        EIGENVECTORS = (self.EIGENVECTORS
                        if mask is None
                        else self.EIGENVECTORS[:, mask])
        SCALED = (self._SCALED_EIGENVECTORS
                  if mask is None
                  else self._SCALED_EIGENVECTORS[:, mask])
        return np.matmul(SCALED,
                         np.matmul(EIGENVECTORS.T,
                                   rhs))


class _FunctionalFieldReconstructorBase(object):
    class MeasurementManagerBase(object):
        """
        Base class for measurement managers.

        An abstract base class for classes implementing measurement handling, i.e.
        probing a field at some ordered measurement points (`probe()` method) and
        loading such values from some other object (e.g. converting the object).

        Objects of this class implement the measurement points as well as probing
        and loading.

        Attributes
        ----------
            number_of_measurements: int
                A number of measurement points. May be implemented as a property.
        """
        number_of_measurements = None

        def load(self, measurements):
            """
            Load the measurements.

            Returns
            -------
            Sequence
                Values measured at the measurement points.

            Note
            ----
                Unless overriden in a subclass, requires `measurements` to be an
                sequence appropriate to be returned.
            """
            return measurements

    class MeasurementManagerHasNoLoadMethodError(_MissingAttributeError):
        _missing = 'load'

    class MeasurementManagerHasNoNumberOfMeasurementsAttributeError(_MissingAttributeError):
        _missing = 'number_of_measurements'

    _mm_validators = [MeasurementManagerHasNoLoadMethodError,
                      MeasurementManagerHasNoNumberOfMeasurementsAttributeError,
                      ]

    def _basic_setup(self, field_components, measurement_manager):
        self._field_components = field_components
        self._measurement_manager = measurement_manager
        self._validate_measurement_manager()

    def _validate_measurement_manager(self):
        for validator in self._mm_validators:
            validator._validate(self._measurement_manager)

    def _process_kernels(self, KernelSolverClass):
        self._solve_kernel = KernelSolverClass(self._kernel)

    def _wrap_kernel_solution(self, solution):
        return LinearMixture(zip(self._field_components,
                                 np.matmul(self._pre_kernel, solution).flatten()))

    def __call__(self, measurements, regularization_parameter=0):
        return self._wrap_kernel_solution(
                        self._solve_kernel(
                                 self._measurement_vector(measurements),
                                 regularization_parameter))

    def _measurement_vector(self, values):
        measurements = self._ensure_is_array(
                                self._measurement_manager.load(values))

        # required by *.*.testLeaveOneOut* and
        # TestFunctionalFieldReconstructorMayUseArbitraryKernelSolverClass
        if len(measurements.shape) == 1:
            return measurements.reshape(-1, 1)

        return measurements

    def _ensure_is_array(self, values):
        if isinstance(values, np.ndarray):
            return values

        return np.array(values)

    def leave_one_out_errors(self, measured, regularization_parameter):
        """
        Note
        ----

            In the future result for a static measurement may be a sequence
            of scalars instead of arrays.
        """
        n = self._kernel.shape[0]
        KERNEL = self._kernel + regularization_parameter * np.identity(n)
        IDX_N = np.arange(n)
        X = self._measurement_vector(measured)
        return [self._leave_one_out_estimate(KERNEL, X, i, IDX_N != i) - ROW
                for i, ROW in enumerate(X)]

    def _leave_one_out_estimate(self, KERNEL, X, i, IDX):
        return np.matmul(KERNEL[np.ix_([i], IDX)],
                         np.linalg.solve(KERNEL[np.ix_(IDX, IDX)],
                                         X[IDX, :]))[0, :]

    def save(self, file):
        np.savez_compressed(file,
                            KERNEL=self._kernel,
                            PRE_KERNEL=self._pre_kernel)


class FunctionalFieldReconstructor(_FunctionalFieldReconstructorBase):
    class MeasurementManagerHasNoProbeMethodError(_MissingAttributeError):
        _missing = 'probe'

    class MeasurementManagerBase(_FunctionalFieldReconstructorBase.MeasurementManagerBase):
        def probe(self, field):
            """
            Probe the field.

            An abstract method implementing probing the field in the measurement
            points.

            Parameters
            ----------
            field : object
                An object implementing the field. It is up to the measurement
                manager to interpret the object and use its API.

            Returns
            -------
            Sequence
                A sequence of the field quantities measured at the measurement
                points.

            Raises
            ------
            NotImplementedError
                Always (unless overriden in a subclass).
            """
            raise NotImplementedError

    _mm_validators = _FunctionalFieldReconstructorBase._mm_validators + \
                     [MeasurementManagerHasNoProbeMethodError,
                      ]

    def __init__(self, field_components, measurement_manager,
                 KernelSolverClass=_LinearKernelSolver):
        self._basic_setup(field_components, measurement_manager)
        self._generate_kernels()
        self._process_kernels(KernelSolverClass)

    def _generate_kernels(self):
        self._generate_pre_kernel()
        self._generate_kernel()

    def _generate_kernel(self):
        self._kernel = np.matmul(self._pre_kernel.T,
                                 self._pre_kernel) * self._pre_kernel.shape[0]

    def _generate_pre_kernel(self):
        m = len(self._field_components)
        n = self._measurement_manager.number_of_measurements
        self._pre_kernel = np.empty((m, n))
        self._fill_probed_components(self._pre_kernel,
                                     self._measurement_manager.probe)
        self._pre_kernel /= m

    def _fill_probed_components(self, values, probe):
        for i, component in enumerate(self._field_components):
            values[i, :] = probe(component)


class LoadableFunctionalFieldReconstructor(_FunctionalFieldReconstructorBase):
    def __init__(self, file, field_components, measurement_manager,
                 KernelSolverClass=_LinearKernelSolver):
        self._basic_setup(field_components, measurement_manager)
        self._load_kernels(file)
        self._process_kernels(KernelSolverClass)

    def _load_kernels(self, file):
        with np.load(file) as fh:
            self._kernel = fh['KERNEL']
            self._pre_kernel = fh['PRE_KERNEL']


class LinearMixture(object):
    def __init__(self, components=()):
        self._components, self._weights = [], []

        try:
            for c, w in components:
                if isinstance(c, LinearMixture):
                    self._append_components_from_mixture(self._components,
                                                         self._weights,
                                                         self._components,
                                                         c * w)
                else:
                    self._components.append(c)
                    self._weights.append(w)

        except TypeError:
            self._components = (components,)
            self._weights = (1,)

        self._prepare_cache_for_dir()

    def _prepare_cache_for_dir(self):
        components = self._components
        self._dir = ({attr for attr in dir(components[0])
                      if all(hasattr(c, attr) for c in components[1:])}
                     if components
                     else ())

    def __getattr__(self, name):
        if name not in self._dir:
            raise AttributeError

        def wrapper(*args, **kwargs):
            return sum(w * getattr(c, name)(*args, **kwargs)
                       for w, c in zip(self._weights,
                                       self._components))

        return wrapper

    def __dir__(self):
        return list(self._dir)

    def __add__(self, other):
        return self._add(other)

    def __radd__(self, other):
        return self._add(other)

    def _add(self, other):
        if other == 0:
            return self

        components = list(self._components)
        weights = list(self._weights)
        self._append_components_from_mixture(components, weights,
                                             self._components,
                                             other)

        return self.__class__(list(zip(components, weights)))

    @staticmethod
    def _append_components_from_mixture(components,
                                        weights,
                                        reference_components,
                                        mixture):
        for c, w in zip(mixture._components,
                        mixture._weights):
            try:
                i = reference_components.index(c)
            except ValueError:
                weights.append(w)
                components.append(c)
            else:
                weights[i] += w

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        return self._mul(other)

    def __rmul__(self, other):
        return self._mul(other)

    def _mul(self, other):
        if other == 1 or self == 0:
            return self

        return self.__class__([(c, w * other)
                               for c, w
                               in zip(self._components,
                                      self._weights)])

    def __truediv__(self, other):
        return self._div(other)

    def _div(self, other):
        return 1. / other * self

    def __eq__(self, other):
        if self:
            return self is other

        return not other

    def __ne__(self, other):
        if self:
            return self is not other

        return bool(other)

    def _bool(self):
        return bool(self._components)

    # Use API appropriate for current Python version
    if sys.version_info.major > 2:
        def __bool__(self):
            return self._bool()

    else:
        def __nonzero__(self):
            return self._bool()

        def __div__(self, other):
            return self._div(other)


class MeasurementManagerBase(FunctionalFieldReconstructor.MeasurementManagerBase):
    """
    Base class for measurement managers.

    An abstract base class for classes implementing measurement handling, i.e.
    probing a field at some ordered measurement points (`probe()` method) and
    loading such values from some other object (e.g. converting the object).

    Objects of this class implement the measurement points as well as probing
    and loading.

    .. deprecated:: 0.2
        The class has been moved to `FunctionalFieldReconstructor` class.
        Use `FunctionalFieldReconstructor.MeasurementManagerBase` instead.

    Attributes
    ----------
        number_of_measurements: int
            A number of measurement points. May be implemented as a property.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            DeprecationWarning(
                'The class has been moved to `FunctionalFieldReconstructor`.  Use `FunctionalFieldReconstructor.MeasurementManagerBase` instead.'))

        super(MeasurementManagerBase, self).__init__(*args, **kwargs)
