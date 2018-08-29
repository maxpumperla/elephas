import jnius_config
import os

jnius_config.add_options('-Dorg.bytedeco.javacpp.nopointergc=true')
jnius_class_path = os.environ.get('ELEPHAS_CLASS_PATH')
if not jnius_class_path:
    print('WARNING: Environment variable ELEPHAS_CLASS_PATH not set.')
elif not os.path.exists(jnius_class_path):
    print('WARNING: File not found : {0}'.format(jnius_class_path))
else:
    jnius_config.set_classpath('.', jnius_class_path)
