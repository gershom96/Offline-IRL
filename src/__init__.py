# import importlib
# import pkgutil

# __all__ = []

# for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
#     full_module_name = f"{__name__}.{module_name}"
#     module = importlib.import_module(full_module_name)
#     globals()[module_name] = module
#     __all__.append(module_name)