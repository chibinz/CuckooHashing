project('CuckooHashing', 'cuda',
  version : '0.1',
  default_options : ['warning_level=2', 'cuda_std=c++14'])

incdir = include_directories('include')

add_languages('cpp')
add_global_arguments('-dc', language : 'cuda')

executable('chash',
           'src/bench.cc',
           'src/host.cc',
           'src/main.cc',
           'src/device.cu',
           'src/multi.cu',
           'src/xxhash.cu',
           install : true,
           include_directories : incdir)
