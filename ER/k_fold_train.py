import os



for i in range(10):
    cmd = 'python mainpro_Oulu.py --model VGG19 --bs 128 --lr 0.01 --fold %d' %(i+1)
    os.system(cmd)
    print('mainpro_Oulu VGG19', i)
print("Train VGG19 ok!")
'''
for i in range(10):
    cmd = 'python mainpro_CK+.py --model VGG19 --bs 128 --lr 0.01 --fold %d' %(i+1)
    os.system(cmd)
    print('mainpro_CK VGG19',i)
print("Train VGG19 ok!")

for i in range(10):
    cmd = 'python mainpro_CK+.py --model Resnet18 --bs 128 --lr 0.005 --fold %d' %(i+1)
    os.system(cmd)
    print('mainpro_CK Resnet18', i)
print("Train VGG19 ok!")


for i in range(10):
    cmd = 'python mainpro_Oulu.py --model Resnet18 --bs 128 --lr 0.01 --fold %d' %(i+1)
    os.system(cmd)
    print('mainpro_Oulu Resnet18', i)
print("Train VGG19 ok!")
'''

