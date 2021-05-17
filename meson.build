project('CuckooHashing', 'cuda',
  version : '0.1',
  default_options : ['warning_level=2'])

incdir = include_directories('include')

executable('chash',
           'src/main.cu',
           'src/table.cu',
           'src/xxhash.cu',
           install : true,
           include_directories : incdir)
