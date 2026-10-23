"""Original OpenJTalk UTF-8 digit phase through owned source operations."""
from .digit_program import DEFINITIONS, GLOBALS, FUNCTIONS
from .digit_runtime import Program

_PROGRAM = Program(DEFINITIONS, GLOBALS, FUNCTIONS)


def set_digit(njd):
    _PROGRAM.call('njd_set_digit', [njd])
