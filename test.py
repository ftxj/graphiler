from __future__ import absolute_import
import sys
import os
  
    
ffi_dir = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


source_dir = os.path.join(ffi_dir, "..", "..", "..")

install_lib_dir = os.path.join(ffi_dir, "..", "..", "..", "..")

print(ffi_dir)


print(source_dir)


print(install_lib_dir)