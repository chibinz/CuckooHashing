project('CuckooHashing', 'c',
  version : '0.1',
  default_options : ['warning_level=3'])

incdir = include_directories('include')

executable('chash',
           'src/cuckoo.c',
           'src/main.c',
           install : true,
           include_directories : incdir)
