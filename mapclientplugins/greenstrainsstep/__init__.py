
"""
MAP Client Plugin
"""

__version__ = '0.1.0'
__author__ = 'Jesse Khorasanee'
__stepname__ = 'GreenStrains'
__location__ = 'https://github.com/Tehsurfer/'

# import class that derives itself from the step mountpoint.
from mapclientplugins.greenstrainsstep import step

# Import the resource file when the module is loaded,
# this enables the framework to use the step icon.
from . import resources_rc