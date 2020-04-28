import tvm
from tvm import te

n = te.var('n')
A = te.placeholder((n,), name='A')
B = te.placeholder((n,), name='B')
k = te.reduce_axis((10, n), 'k')
C = te.compute((1,), lambda _: te.sum(A[k] * B[k], axis=k), name='C')

s = te.create_schedule(C.op)
print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")
s = s.normalize()
print(tvm.lower(s, [A, B, C], simple_mode=True))