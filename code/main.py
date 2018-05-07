import subprocess
print "start preprocess"
subprocess.call('python preprocess.py',shell=True)
print "start train model"
subprocess.call('python train.py',shell=True)
