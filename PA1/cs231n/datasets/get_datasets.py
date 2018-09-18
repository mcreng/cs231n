from __future__ import print_function
from six.moves.urllib import request
import tarfile  
import os

url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

file_name = url.split('/')[-1]
u = request.urlopen(url)
f = open(file_name, 'wb')
meta = u.info()
file_size = int(meta.get("Content-Length"))
print("Downloading: %s Bytes: %s" % (file_name, file_size))

file_size_dl = 0
block_sz = 1048576
while True:
    buffer = u.read(block_sz)
    if not buffer:
        break

    file_size_dl += len(buffer)
    f.write(buffer)
    status = "%d  [%3.2f%%]\r" % (file_size_dl, file_size_dl * 100. / file_size)
#    status = status + chr(8)*(len(status)+1)
    print(status,end='')

f.close()

print('Extracting: %s' % file_name)
t = tarfile.open(file_name, "r:*")  
t.extractall(path = '.')  
t.close()  

os.remove(file_name)