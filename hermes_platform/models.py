"""Explicit model boundary for Hermes' declared connector wire vocabulary."""

from agent.sdk_boundary import bundled

if bundled():
    from standalone.web_schema import (
        ConfigDict, Field, RequestModel as BaseModel, ValidationError, field_validator,
    )
else:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
