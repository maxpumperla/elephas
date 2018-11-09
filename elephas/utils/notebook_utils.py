def is_running_in_notebook():
    try:
        cfg = get_ipython().config
        if 'IPKernelApp' in cfg:
            return True
        else:
            return False
    except NameError:
        return False
