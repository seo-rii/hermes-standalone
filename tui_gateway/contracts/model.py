"""The declared RPC model vocabulary, selected for the owning distribution."""

from agent.sdk_boundary import bundled

if bundled():
    from standalone.web_schema import (
        ConfigDict, Field, RequestModel as BaseModel, RootModel, ValidationError, model_validator,
    )
else:
    from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError, model_validator
