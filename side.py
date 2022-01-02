with_star = open('acc_with_star', 'r')
no = open('acc_no_star', 'r')

with_lis = with_star.readlines()
no_lis = no.readlines()

for i in range(30):
    w = with_lis[i].split(" ")
    for j in range(30):
        n = no_lis[j].split(" ")
        if (w[3] == n[3]):
            if (float(w[-1]) > float(n[-1])):
                print(with_lis[i])
                print(no_lis[j])
                print("$$$$$$$$$$$$$")
        

        