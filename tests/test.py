"""
Functions for testing scripts.
"""

import pcn
import pytest
import numpy

# Things we need tests for [PLEASE ADD TO LIST!]:
#   1. Model structure is connected
#   2. Input dimensions match signal dimensions
#   3. Every frame has exactly one audio window matched to it
#   4. No computation errors in audio spectrograms
#   5. Video conversion to greyscale runs properly (or color handling is correct)

class TestInputs:
    def test_windowmatch(self):

    def test_audiospec(self):

    def test_greyscale(self):

class TestModel:
    def test_connected(self):

    def test_dimensionmatch(self):

# We can also add tests here that look at output parameters, but I've kept things
# to the model structure itself for now.