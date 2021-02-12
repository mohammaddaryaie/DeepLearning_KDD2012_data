from __future__ import print_function
import os
import gc
import psutil
import sys

def Process_mem():
    proc = psutil.Process(os.getpid())
    gc.collect()
    mem0 = proc.memory_info().rss/(1024*1024)
    print ('Process Number: '+str(os.getpid())+' ****  '+ 'Memory Usage:'+ str(mem0) + 'Mb')


def Variable_mem():
    local_vars = list(locals().items())
    sum_mem=0
    max_mem=0
    var_name_max_mem = ''
    for var, obj in local_vars:
        size_obj=sys.getsizeof(obj)
        sum_mem+=size_obj
        if int(size_obj) > int(max_mem):
            max_mem=size_obj
            var_name_max_mem= var

    print('Sum of variable memory usage is: '+str(sum_mem/(1024*1024))+'Mb'+
          ' **** '+'Maximume Varible is:'+ var_name_max_mem+':'+str(max_mem/(1024*1024))+'Mb')

