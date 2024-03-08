import copy

score_mapping = {6: 10000, 7: 36, 8: 720, 9: 360, 10: 80, 11: 252, 12: 108, 13: 72, 14: 54, 15: 180, 16: 72, 17: 180,
                 18: 119, 19: 36, 20: 306, 21: 1080, 22: 144, 23: 1800, 24: 3600}


def solveq1(data):
    ans = 10_000
    score1 = []
    for i in range(3):
        score1.extend([score_mapping[int(data[i][0]) + int(data[i][1]) + int(data[i][2])],
                       score_mapping[int(data[0][i]) + int(data[1][i]) + int(data[2][i])]])
    score1.extend([score_mapping[int(data[0][0]) + int(data[1][1]) + int(data[2][2])],
                   score_mapping[int(data[0][2]) + int(data[1][1]) + int(data[2][0])]])
    ans = max(score1)
    return ans


def solveq2(data):
    ans = 10_000
    score2=[0 for i in range(8)]
    p=0
    num=[]
    exi=[]
    for i in range(3):
        for j in range(3):
            if data[i][j]:
                exi.append(int(data[i][j]))
    for i in range(9):
        if i+1 not in exi:
            num.append(i+1)
    from itertools import permutations
    for data2 in permutations(num):
        data1 = copy.deepcopy(data)
        p+=1
        n=0
        for i in range(3):
            for j in range(3):
                if data1[i][j]==0:
                    data1[i][j]=data2[n]
                    n+=1
                    if n>4:
                        break
        for i in range(3):
            score2[i]+=int(score_mapping[int(data1[i][0]) + int(data1[i][1]) + int(data1[i][2])])
        for i in range(3):
            score2[i+3]+=int(score_mapping[int(data1[0][i]) + int(data1[1][i]) + int(data1[2][i])])
        score2[6]+=int(score_mapping[int(data1[0][0]) + int(data1[1][1]) + int(data1[2][2])])
        score2[7]+=int(score_mapping[int(data1[0][2]) + int(data1[1][1]) + int(data1[2][0])])
    ans=int(max(score2)/p)
    return ans


def main():
    print(solveq1([[7, 6, 9], [4, 5, 3], [2, 1, 8]]))
    print(solveq2([[0, 6, 0], [4, 0, 3], [2, 0, 0]]))


if __name__ == "__main__":
    main()
