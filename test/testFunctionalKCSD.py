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

import unittest
import numpy as np

try:
    import pandas as pd
except:
    pd = None

import kesi as kesi

class FunctionFieldComponent(object):
    def __init__(self, func, fprime):
        self._func = func
        self._fprime = fprime

    def func(self, arg):
        if isinstance(arg, list):
            return list(map(self._func, arg))
        return self._func(arg)

    def fprime(self, arg):
        if isinstance(arg, list):
            return list(map(self._fprime, arg))
        return self._fprime(arg)

class _GivenComponentsAndNodesBase(unittest.TestCase):
    def setUp(self):
        if not hasattr(self, 'FIELD_COMPONENTS'):
            self.skipTest('test in abstract class called')

    def createField(self, name, points, weights={}):
        return {k: sum(getattr(f, name)([k])[0] * weights.get(c, 1)
                       for c, f in self.FIELD_COMPONENTS.items())
                for k in points
                }

    def createReconstructor(self, name, nodes):
        return kesi.FunctionalKernelFieldReconstructor(self.FIELD_COMPONENTS.values(),
                                                       name,
                                                       nodes)

    def _checkApproximation(self, expected, measured, measuredName,
                            regularization_parameter=None):
        reconstructor = self.createReconstructor(measuredName,
                                                 list(measured))
        approximator = self._get_approximator(reconstructor, measured,
                                              regularization_parameter)
        for name in expected:
            field = getattr(approximator, name)

            for k, v in expected[name].items():
                self.assertEqual(v,
                                 field(k))

    def _get_approximator(self, reconstructor, measured,
                          regularization_parameter=None):
        if regularization_parameter is None:
            return reconstructor(measured)

        return reconstructor(measured,
                             regularization_parameter=regularization_parameter)

    def checkWeightedApproximation(self, measuredName, nodes,
                                   names, points, weights={}):
        self._checkApproximation(
            {name: self.createField(name, points, weights=weights)
             for name in names},
            self.createField(measuredName, nodes, weights=weights),
            measuredName)

    def checkReconstructor(self, expected, reconstructor, funcValues,
                           regularization_parameter=None):
        approximator = self._get_approximator(reconstructor,
                                              funcValues,
                                              regularization_parameter)
        for name in expected:
            field = getattr(approximator, name)

            for k, v in expected[name].items():
                self.assertAlmostEqual(v, field(k))


class _GivenSingleComponentSingleNodeBase(_GivenComponentsAndNodesBase):
    NODES = ['zero']

    def testProperlyHandlesTheNode(self):
        approximated = ['func', 'fprime']
        measuredName = 'func'
        self.checkWeightedApproximation(measuredName,
                                        self.NODES,
                                        approximated,
                                        self.NODES,
                                        weights={'1': 2})

    def testExtrapolatesOneOtherPoint(self):
        approximated = ['func', 'fprime']
        measuredName = 'func'
        points = ['one']
        self.checkWeightedApproximation(measuredName,
                                        self.NODES,
                                        approximated,
                                        points)

    def testExtrapolatesManyOtherPoints(self):
        approximated = ['func', 'fprime']
        measuredName = 'func'
        points = ['one', 'two']
        self.checkWeightedApproximation(measuredName,
                                        self.NODES,
                                        approximated,
                                        points)


class GivenSingleConstantFieldComponentSingleNode(_GivenSingleComponentSingleNodeBase):
    FIELD_COMPONENTS = {'1': FunctionFieldComponent(lambda x: 1,
                                                    lambda x: 0)}


class _GivenTwoNodesBase(_GivenComponentsAndNodesBase):
    NODES = ['zero', 'two']

    def testProperlyHandlesTheNodes(self):
        approximated = ['func', 'fprime']
        measuredName = 'func'
        self.checkWeightedApproximation(measuredName,
                                        self.NODES,
                                        approximated,
                                        self.NODES,
                                        weights={'1': 2})

    def testInterpolates(self):
        approximated = ['func', 'fprime']
        measuredName = 'func'
        self.checkWeightedApproximation(measuredName,
                                        self.NODES,
                                        approximated,
                                        ['one'],
                                        weights={'1': 2})

    def testExtrapolatesAndInterpolates(self):
        approximated = ['func', 'fprime']
        measuredName = 'func'
        self.checkWeightedApproximation(measuredName,
                                        self.NODES,
                                        approximated,
                                        ['one', 'three'],
                                        weights={'1': 2})


class GivenTwoNodesAndTwoLinearFieldComponents(_GivenTwoNodesBase):
    FIELD_COMPONENTS = {'1': FunctionFieldComponent(lambda x: 1,
                                                    lambda x: 0),
                        'x': FunctionFieldComponent({'zero': 0,
                                                     'one': 1,
                                                     'two': 2,
                                                     'three': 3}.get,
                                                    {'zero': 1,
                                                     'one': 1,
                                                     'two': 1,
                                                     'three': 1}.get)}

    def testRegularisation(self):
        expected = {'func': {'zero': 0.8,
                             'one': 1.4,
                             },
                    'fprime': {'zero': 0.6,
                               'one': 0.6,
                               },
                    }
        self.checkReconstructor(expected,
                                self.createReconstructor(
                                        'func',
                                        list(expected['func'])),
                                {'zero': 1, 'one': 2},
                                regularization_parameter=1.0)


class GivenTwoNodesAndThreeLinearFieldComponents(_GivenTwoNodesBase):
    FIELD_COMPONENTS = {'1': FunctionFieldComponent(lambda x: 1,
                                                    lambda x: 0),
                        'x': FunctionFieldComponent({'zero': 0,
                                                     'one': 1,
                                                     'two': 2,
                                                     'three': 3}.get,
                                                    {'zero': 1,
                                                     'one': 1,
                                                     'two': 1,
                                                     'three': 1}.get),
                        '1 - x': FunctionFieldComponent({'zero': 1,
                                                         'one': 0,
                                                         'two': -1,
                                                         'three': -2}.get,
                                                        {'zero': -1,
                                                         'one': -1,
                                                         'two': -1,
                                                         'three': -1}.get),
                        }


@unittest.skipIf(pd is None, 'No pandas module')
class WhenCalledWithPandasSeries(GivenTwoNodesAndThreeLinearFieldComponents):
    def createField(self, name, points, weights={}):
        return pd.Series(super(WhenCalledWithPandasSeries,
                               self).createField(name,
                                                 points,
                                                 weights=weights))

    def _checkApproximation(self, expected, measured, measuredName,
                            regularization_parameter=None):
        reconstructor = self.createReconstructor(
                              measuredName,
                              list(measured.index))
        approximator = self._get_approximator(reconstructor,
                                              measured,
                                              regularization_parameter)
        for name in expected:
            field = getattr(approximator, name)
            for k in expected[name].index:
                self.assertEqual(expected[name][k],
                                 field(k))


if __name__ == '__main__':
    unittest.main()