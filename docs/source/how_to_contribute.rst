How to Contribute
=================

Code style
----------
All contributions should be formated according to the PEP8 standard. Slightly more than 80 characters can sometimes be tolerated if increased line width increases readability. All docstrings should follow Google Style Python Docstrings.

Dependencies
------------
Make sure that you use Python >= 3.5 and have TensorFlow >= 2.0 installed.

Unit tests
----------
After any change of QGOpt, one has to check whether all the tests run without errors. Currently, tests check optimization primitives (retractions, vector transports, etc.) for all manifolds and check optimizers' performance on the simple optimization problem on complex Stiefel manifold. For any new functionality, please provide suitable unit tests. Also, if you find a bug, consider adding a test that detects the bug before fixing it.

::

	pytest

running all test files.

::

	pytest test_manifolds.py

running tests of all manifolds except complex Stiefel manifold.

::

	pytest test_stiefel.py

running tests of complex Stiefel manifold.

::

	pytest test_optimizers.py

running tests of optimizers.

