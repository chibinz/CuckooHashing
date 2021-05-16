project('CuckooHashing', 'c',
  version : '0.1',
  default_options : ['warning_level=3'])

incdir = include_directories('include')

executable('chash',
           'src/main.c',
           'src/table.c',
           'src/xxhash.c',
           install : true,
           include_directories : incdir)
