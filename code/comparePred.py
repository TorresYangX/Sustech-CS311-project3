import difflib

file_1 = open('predict\\testlabel.txt','r')
file_2 = open('testlabel.txt','r')

a = file_1.read().splitlines()
b = file_2.read().splitlines()

n = 0
for i in range(len(a)):
    if a[i] != b[i]:
        n += 1

print(n)

file_1.close()
file_2.close()