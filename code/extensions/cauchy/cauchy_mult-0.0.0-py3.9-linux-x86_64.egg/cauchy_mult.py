def __bootstrap__():
    global __bootstrap__, __loader__, __file__
    import sys, pkg_resources, importlib.util, importlib.machinery
    __file__ = pkg_resources.resource_filename(__name__, 'cauchy_mult.cpython-39-x86_64-linux-gnu.so')
    __loader__ = None; del __bootstrap__, __loader__
    loader = importlib.machinery.SourceFileLoader(__name__, __file__)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    
    #spec = importlib.util.spec_from_file_location(__name__,__file__)
    #mod = importlib.util.module_from_spec(spec)
    #spec.loader.exec_module(mod)
__bootstrap__()
