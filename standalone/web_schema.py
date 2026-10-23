"""Validated JSON request records for Hermes' declared dashboard schemas.

Only the annotation vocabulary used by Hermes is supported. An unsupported
annotation is a programming error, never an instruction to accept arbitrary
input. This module is deliberately not installed as ``pydantic``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields as dataclass_fields
from decimal import Decimal, InvalidOperation
from enum import Enum
import json
import math
import re
import types
from typing import Any, Annotated, Literal, Union, get_args, get_origin, get_type_hints


MISSING = object()


class ValidationError(ValueError):
    def __init__(self, errors):
        self._errors = errors
        super().__init__("Request validation failed")

    def errors(self, **_kwargs):
        return copy.deepcopy(self._errors)


@dataclass
class FieldInfo:
    default: object = MISSING
    default_factory: object = None
    ge: float | None = None
    le: float | None = None
    gt: float | None = None
    lt: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    alias: str | None = None
    discriminator: str | None = None
    exclude: bool = False
    pattern: str | None = None
    description: str | None = None


def Field(default=MISSING, **kwargs):
    return FieldInfo(default=default, **kwargs)


def _declared_field(owner, name, annotation):
    """Combine assignment defaults with field metadata declared in Annotated.

    ACP's generated records declare their wire aliases on Annotated rather
    than the assignment. Keep the annotation intact for validation while
    exposing the same resolved metadata to constructors and serialization.
    """
    assigned = getattr(owner, name, MISSING)
    result = Field()
    if get_origin(annotation) is Annotated:
        for metadata in get_args(annotation)[1:]:
            if isinstance(metadata, FieldInfo):
                for item in dataclass_fields(FieldInfo):
                    value = getattr(metadata, item.name)
                    if value is not MISSING and (item.name == "default" or value is not None and value is not False):
                        setattr(result, item.name, value)
    if isinstance(assigned, FieldInfo):
        for item in dataclass_fields(FieldInfo):
            value = getattr(assigned, item.name)
            if value is not MISSING and (item.name == "default" or value is not None and value is not False):
                setattr(result, item.name, value)
    elif assigned is not MISSING:
        result.default = assigned
    result.annotation = annotation
    return result


class StrictInt:
    pass


class StrictBool:
    pass


class SecretStr:
    def __init__(self, value):
        if not isinstance(value, str):
            raise TypeError("Secret value must be a string")
        self._value = value

    def get_secret_value(self):
        return self._value

    def __str__(self):
        return "**********" if self._value else ""

    def __repr__(self):
        return f"SecretStr({str(self)!r})"

    def __eq__(self, other):
        return isinstance(other, SecretStr) and self._value == other._value


def field_validator(*names, mode="after"):
    if mode not in ("before", "after"):
        raise TypeError("Unsupported dashboard validator mode")

    def decorate(function):
        target = function.__func__ if isinstance(function, classmethod) else function
        target._hermes_validator = (names, mode)
        return function

    return decorate


def ConfigDict(**kwargs):
    unsupported = set(kwargs) - {"extra", "populate_by_name", "from_attributes", "strict"}
    if unsupported:
        raise TypeError("Unsupported Hermes contract options: " + ", ".join(sorted(unsupported)))
    return kwargs


def model_validator(*, mode):
    if mode != "after":
        raise TypeError("Hermes contracts declare only after model validators")

    def decorate(function):
        function._hermes_model_validator = True
        return function

    return decorate


def _fail(location, kind, message):
    raise ValidationError([{"loc": tuple(location), "type": kind, "msg": message}])


def _discriminated(annotation, value, discriminator, location):
    """Select the declared tag before validation so nested refusals retain their paths."""
    alternatives = get_args(annotation)
    if get_origin(annotation) not in (Union, types.UnionType):
        raise TypeError("A discriminator requires a declared union")
    if not isinstance(value, (dict, RequestModel)):
        _fail(location, "model_attributes_type", "Input should be an object")
    tag = value.get(discriminator, MISSING) if isinstance(value, dict) else getattr(value, discriminator, MISSING)
    if tag is MISSING and isinstance(value, dict):
        for alternative in alternatives:
            if isinstance(alternative, type) and issubclass(alternative, RequestModel):
                annotation = alternative._fields().get(discriminator)
                alias = _declared_field(alternative, discriminator, annotation).alias
                if alias and alias in value:
                    tag = value[alias]
                    break
    if tag is MISSING:
        _fail(location, "union_tag_not_found", "The discriminator field is required")
    for alternative in alternatives:
        if not isinstance(alternative, type) or not issubclass(alternative, RequestModel):
            raise TypeError("Hermes discriminated unions require model alternatives")
        declared = alternative._fields().get(discriminator)
        if get_origin(declared) is Annotated:
            declared = get_args(declared)[0]
        if get_origin(declared) is not Literal:
            raise TypeError("Hermes discriminators require literal tag fields")
        if any(type(tag) is type(choice) and tag == choice for choice in get_args(declared)):
            return validate(alternative, value, (*location, tag))
    _fail(location, "union_tag_invalid", "Input tag does not match a declared alternative")


def _populated_fields(value):
    if isinstance(value, RequestModel):
        names = value.model_fields_set & value._fields().keys()
        return len(names) + sum(_populated_fields(getattr(value, name)) for name in names)
    if isinstance(value, dict):
        return sum(_populated_fields(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_populated_fields(item) for item in value)
    return 0


def _field_value(annotation, value, field, location, *, strict=False):
    if field.discriminator:
        base = get_args(annotation)[0] if get_origin(annotation) is Annotated else annotation
        value = _discriminated(base, value, field.discriminator, location)
    else:
        value = validate(annotation, value, location, strict=strict)
    return constrain(value, field, location)


def validate(annotation, value, location=(), *, strict=False):
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation is Any or annotation is object:
        return copy.deepcopy(value)
    if annotation is type(None):
        if value is None:
            return None
        _fail(location, "none_required", "Input should be None")
    if origin is Annotated:
        fields = [metadata for metadata in args[1:] if isinstance(metadata, FieldInfo)]
        discriminator = next((field.discriminator for field in fields if field.discriminator), None)
        result = (_discriminated(args[0], value, discriminator, location) if discriminator
                  else validate(args[0], value, location, strict=strict))
        for metadata in args[1:]:
            if isinstance(metadata, FieldInfo):
                result = constrain(result, metadata, location)
        return result
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        non_null = [alternative for alternative in args if alternative is not type(None)]
        if len(non_null) == 1:
            return validate(non_null[0], value, location, strict=strict)
        ordered = sorted(args, key=lambda typ: not (isinstance(typ, type) and type(value) is typ))
        errors = []
        candidates = []
        model_union = isinstance(value, dict) and all(
            isinstance(item, type) and issubclass(item, RequestModel) for item in non_null
        )
        for alternative in ordered:
            try:
                result = validate(alternative, value, location, strict=strict)
                if not model_union:
                    return result
                candidates.append(result)
            except ValidationError as exc:
                label = getattr(alternative, "__name__", str(alternative))
                errors.extend({**error, "loc": (*location, label, *error["loc"][len(location):])}
                              for error in exc.errors())
        if candidates:
            return max(candidates, key=_populated_fields)
        raise ValidationError(errors)
    if origin is Literal:
        if any(type(value) is type(choice) and value == choice for choice in args):
            return value
        _fail(location, "literal_error", "Input should be one of the declared literal values")
    if isinstance(annotation, type) and issubclass(annotation, RequestModel):
        try:
            return annotation.model_validate(value)
        except ValidationError as exc:
            raise ValidationError([{**error, "loc": (*location, *error["loc"])} for error in exc.errors()]) from None
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            return annotation(value)
        except (ValueError, TypeError):
            _fail(location, "enum", "Input should be a declared enum value")
    if annotation in (str, SecretStr):
        if annotation is SecretStr and isinstance(value, SecretStr):
            return value
        if not isinstance(value, str):
            _fail(location, "string_type", "Input should be a valid string")
        return SecretStr(value) if annotation is SecretStr else value
    if annotation in (bool, StrictBool):
        if isinstance(value, bool):
            return value
        if annotation is bool and not strict:
            normalized = value.lower() if isinstance(value, str) else value
            if normalized in (0, "0", "false", "f", "no", "n", "off"):
                return False
            if normalized in (1, "1", "true", "t", "yes", "y", "on"):
                return True
        _fail(location, "bool_type", "Input should be a valid boolean")
    if annotation in (int, StrictInt, float):
        if annotation is StrictInt or annotation is int and strict:
            if type(value) is not int:
                _fail(location, "int_type", "Input should be a valid integer")
            return value
        if annotation is float and strict:
            if type(value) not in (int, float):
                _fail(location, "float_type", "Input should be a valid number")
            return float(value)
        if not isinstance(value, (str, int, float, Decimal)):
            _fail(location, "number_type", "Input should be a valid number")
        try:
            if annotation is float:
                return float(value)
            number = Decimal(str(int(value) if isinstance(value, bool) else value))
            if number.is_finite() and number == number.to_integral_value():
                return int(number)
        except (InvalidOperation, ValueError, OverflowError):
            pass
        _fail(location, "int_parsing", "Input should be a valid integer")
    if annotation in (list, dict) or origin in (list, dict):
        kind = origin or annotation
        if not isinstance(value, kind):
            _fail(location, kind.__name__ + "_type", "Input should be a valid " + kind.__name__)
        if not args:
            return copy.deepcopy(value)
        if kind is list:
            return [validate(args[0], item, (*location, index), strict=strict) for index, item in enumerate(value)]
        return {validate(args[0], key, (*location, key), strict=strict): validate(args[1], item, (*location, key), strict=strict)
                for key, item in value.items()}
    raise TypeError(f"Unsupported dashboard request annotation: {annotation!r}")


def constrain(value, field, location):
    if value is None:
        return value
    comparisons = {"ge": lambda a, b: a >= b, "le": lambda a, b: a <= b,
                   "gt": lambda a, b: a > b, "lt": lambda a, b: a < b}
    for name, compare in comparisons.items():
        bound = getattr(field, name)
        if bound is not None and not compare(value, bound):
            _fail(location, name, f"Input violates {name}={bound}")
    for name, compare in (("min_length", lambda a, b: a >= b), ("max_length", lambda a, b: a <= b)):
        bound = getattr(field, name)
        if bound is not None and not compare(len(value), bound):
            _fail(location, name, f"Input violates {name}={bound}")
    if field.pattern is not None and (not isinstance(value, str) or re.search(field.pattern, value) is None):
        _fail(location, "string_pattern_mismatch", "Input does not match the required pattern")
    return value


class _ModelFields:
    def __get__(self, instance, owner):
        fields = {}
        for name, annotation in owner._fields().items():
            fields[name] = _declared_field(owner, name, annotation)
        return fields


class RequestModel:
    model_config = {}
    model_fields = _ModelFields()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        config = {}
        for base in reversed(cls.__mro__[1:]):
            config.update(getattr(base, "model_config", {}))
        config.update(cls.__dict__.get("model_config", {}))
        cls.model_config = ConfigDict(**config)

    @classmethod
    def _fields(cls):
        return get_type_hints(cls, include_extras=True)

    @classmethod
    def _validators(cls, name, mode):
        for base in reversed(cls.__mro__):
            for attr, descriptor in vars(base).items():
                target = descriptor.__func__ if isinstance(descriptor, classmethod) else descriptor
                marker = getattr(target, "_hermes_validator", None)
                if marker and name in marker[0] and mode == marker[1]:
                    yield getattr(cls, attr)

    def __init__(self, **data):
        self.model_fields_set = set()
        self._extras = {}
        errors = []
        consumed = set()
        for name, annotation in self._fields().items():
            field = _declared_field(type(self), name, annotation)
            key = field.alias or name
            if key not in data and self.model_config.get("populate_by_name") and name in data:
                key = name
            if key not in data:
                if field.default_factory is not None:
                    value = field.default_factory()
                elif field.default is not MISSING and field.default is not Ellipsis:
                    value = copy.deepcopy(field.default)
                else:
                    errors.append({"loc": (key,), "type": "missing", "msg": "Field required"})
                    continue
            else:
                value = data[key]
                consumed.add(key)
                self.model_fields_set.add(name)
            try:
                for validator in self._validators(name, "before"):
                    value = validator(value)
                value = _field_value(annotation, value, field, (key,), strict=self.model_config.get("strict", False))
                for validator in self._validators(name, "after"):
                    value = validator(value)
                setattr(self, name, value)
            except ValidationError as exc:
                errors.extend(exc.errors())
            except ValueError as exc:
                errors.append({"loc": (name,), "type": "value_error", "msg": str(exc)})
        extra = set(data) - consumed
        if self.model_config.get("extra") == "forbid":
            errors.extend({"loc": (name,), "type": "extra_forbidden", "msg": "Extra inputs are not permitted"}
                          for name in sorted(extra))
        elif self.model_config.get("extra") == "allow":
            self._extras = {name: copy.deepcopy(data[name]) for name in extra}
            self.model_fields_set.update(extra)
        if errors:
            raise ValidationError(errors)
        for base in reversed(type(self).__mro__):
            for name, method in vars(base).items():
                if getattr(method, "_hermes_model_validator", False):
                    try:
                        result = getattr(self, name)()
                    except ValueError as exc:
                        raise ValidationError([{"loc": (), "type": "value_error", "msg": str(exc)}]) from exc
                    if result is not self:
                        raise TypeError("Hermes model validators must return their validated instance")

    def __getattr__(self, name):
        extra = self.__dict__.get("_extras", {})
        if name in extra:
            return extra[name]
        raise AttributeError(name)

    @property
    def model_extra(self):
        return self._extras if self.model_config.get("extra") == "allow" else None

    @classmethod
    def model_validate(cls, value, *, from_attributes=None):
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            if from_attributes if from_attributes is not None else cls.model_config.get("from_attributes"):
                value = {name: getattr(value, name) for name in cls._fields() if hasattr(value, name)}
            else:
                _fail((), "model_type", "Input should be an object")
        return cls(**value)

    @classmethod
    def model_validate_json(cls, value):
        try:
            parsed = json.loads(value)
        except (ValueError, UnicodeError):
            _fail((), "json_invalid", "Invalid JSON")
        return cls.model_validate(parsed)

    def model_dump(self, *, include=None, exclude=None, exclude_none=False, exclude_unset=False,
                   exclude_defaults=False, mode="python", by_alias=False):
        output = {}
        for name, annotation in self._fields().items():
            value = getattr(self, name)
            if (include is not None and name not in include or exclude is not None and name in exclude
                    or exclude_none and value is None or exclude_unset and name not in self.model_fields_set):
                continue
            field = _declared_field(type(self), name, annotation)
            if field.exclude:
                continue
            if exclude_defaults and value == (field.default_factory() if field.default_factory else field.default):
                continue
            output[field.alias if by_alias and field.alias else name] = dump_value(value, mode=mode,
                exclude_none=exclude_none, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, by_alias=by_alias)
        output.update({name: dump_value(value, mode=mode, exclude_none=exclude_none)
                       for name, value in self._extras.items()
                       if not (exclude_none and value is None) and (include is None or name in include)
                       and (exclude is None or name not in exclude)})
        return output

    dict = model_dump

    def model_dump_json(self, **kwargs):
        return json.dumps(self.model_dump(mode="json", **kwargs), ensure_ascii=False)

    def model_copy(self, *, update=None, deep=False):
        result = copy.deepcopy(self) if deep else copy.copy(self)
        result.model_fields_set = self.model_fields_set | set(update or {})
        result._extras = dict(result._extras)
        for name, value in (update or {}).items():
            if name not in self._fields() and self.model_config.get("extra") == "allow":
                result._extras[name] = value
            else:
                setattr(result, name, value)
        return result


class RootModel(RequestModel):
    """A validated unwrapped record, including ACP's generated generic roots."""

    @classmethod
    def __class_getitem__(cls, annotation):
        return type(f"{cls.__name__}[{annotation!r}]", (cls,),
                    {"__module__": cls.__module__, "__annotations__": {"root": annotation}})

    def __init__(self, root=MISSING, **data):
        if root is MISSING:
            super().__init__(**data)
        elif data:
            raise TypeError("A root record accepts one root value")
        else:
            super().__init__(root=root)

    @classmethod
    def model_validate(cls, value):
        if isinstance(value, cls):
            return value
        return cls(root=value)

    def model_dump(self, *, mode="python", exclude_none=False, **kwargs):
        return dump_value(self.root, mode=mode, exclude_none=exclude_none, **kwargs)

    dict = model_dump


def dump_value(value, *, mode="json", exclude_none=False, **kwargs):
    if isinstance(value, RequestModel):
        return value.model_dump(mode=mode, exclude_none=exclude_none, **kwargs)
    if isinstance(value, SecretStr):
        return str(value) if mode == "json" else value
    if isinstance(value, Enum):
        return value.value if mode == "json" else value
    if isinstance(value, dict):
        return {key: dump_value(item, mode=mode, exclude_none=exclude_none, **kwargs) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [dump_value(item, mode=mode, exclude_none=exclude_none, **kwargs) for item in value]
    if mode == "json" and isinstance(value, float) and not math.isfinite(value):
        return None
    return value
