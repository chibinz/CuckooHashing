project('CuckooHashing', 'cuda',
  version : '0.1',
  default_options : ['warning_level=2', 'cuda_std=c++14'])

incdir = include_directories('include')

add_global_arguments('-dc', language : 'cuda')

executable('chash',
           'src/main.cu',
           'src/device.cu',
           'src/host.cu',
           'src/multi.cu',
           'src/xxhash.cu',
           install : true,
           include_directories : incdir)
