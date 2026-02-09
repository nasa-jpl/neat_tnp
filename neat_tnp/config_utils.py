"""
Utilities for converting between Python objects and dictionaries.
Eliminates manual conversion boilerplate between Python configs and Rust.
"""

def dict_to_obj(d):
    """
    Converts a dictionary to an object recursively.
    Access dict_thing["x"] becomes obj_thing.x
    """
    if not isinstance(d, dict):
        return d
    
    class DictObject:
        def __repr__(self):
            attrs = ', '.join(f'{k}={repr(v)}' for k, v in self.__dict__.items())
            return f'DictObject({attrs})'
    
    obj = DictObject()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(obj, key, dict_to_obj(value))
        elif isinstance(value, list):
            setattr(obj, key, [dict_to_obj(item) if isinstance(item, dict) else item for item in value])
        else:
            setattr(obj, key, value)
    
    return obj

def obj_to_dict(obj):
    """
    Converts an object to a dictionary recursively.
    Ignores callables and attributes starting with "_".
    """
    if not hasattr(obj, '__dict__'):
        return obj
    
    result = {}
    for key, value in obj.__dict__.items():
        # Skip attributes starting with "_" and callables
        if key.startswith('_') or callable(value):
            continue
        
        if hasattr(value, '__dict__') and not isinstance(value, type):
            result[key] = obj_to_dict(value)
        elif isinstance(value, list):
            result[key] = [obj_to_dict(item) if hasattr(item, '__dict__') and not isinstance(item, type) else item for item in value]
        else:
            result[key] = value
    
    return result

def dataclass_to_dict(obj):
    """
    Converts a dataclass to a dictionary recursively.
    Handles dataclasses specifically with better field detection.
    """
    from dataclasses import is_dataclass, fields
    
    if is_dataclass(obj):
        result = {}
        for field in fields(obj):
            value = getattr(obj, field.name)
            if is_dataclass(value):
                result[field.name] = dataclass_to_dict(value)
            elif isinstance(value, list):
                result[field.name] = [dataclass_to_dict(item) if is_dataclass(item) else item for item in value]
            else:
                result[field.name] = value
        return result
    else:
        return obj