import jnius_config
import os

try:
    jnius_config.add_options('-Dorg.bytedeco.javacpp.nopointergc=true')
    jnius_class_path = os.environ.get('ELEPHAS_CLASS_PATH')
    if not jnius_class_path:
        print('WARNING: Environment variable ELEPHAS_CLASS_PATH not set.')
    else:
        jnius_config.set_classpath('.', jnius_class_path)
except:
    print("WARNING: Jnius configuration has already been loaded")
