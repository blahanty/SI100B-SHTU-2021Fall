def solve(filename):
    from check import CheckWin, Pung, Calc
    from re import split
    file = open(filename)
    lines = file.readlines()
    file.close()
    winner = open('winner.csv', 'w')
    tile = open('tile.csv', 'w')
    battle = open('battle.csv', 'w')
    global pungm, pungh, games, order
    for i in range(len(lines)):
        lines[i] = split('[,\n]', lines[i])
        lines[i].pop()
    dealer = lines[0][0]
    names = {}
    wins = [0, 0, 0, 0]
    points = [50000, 50000, 50000, 50000]
    wintile = {}
    games = 0
    for i in range(4):
        names[lines[1][i]] = []
        order[i] = lines[1][i]
        order[lines[1][i]] = i
    current = [2, 2, 2, 2]
    battle.write('Game %d\n' % games)
    xs.append(games)
    for i in range(4):
        battle.write('%s,%d\n' % (order[i], points[i]))
        ys[i].append(points[i])
    battle.write('\n')
    while 1:
        flag = 0
        try:
            for i in range(4):
                names[order[i]] = []
                for j in range(13):
                    names[order[i]].append(lines[current[i]][i])
                    current[i] += 1
            turnnum = order[dealer]
            for i in range(21):
                for j in range(4):
                    turn = order[turnnum]
                    names[turn].append(lines[current[turnnum]][turnnum])
                    current[turnnum] += 1
                    if CheckWin(names[turn]):
                        wins[turnnum] += 1
                        flag = 1
                        wintile[lines[current[turnnum] - 1][turnnum]] = wintile.get(
                            lines[current[turnnum] - 1][turnnum], 0) + 1
                        winner.write('%s\n' % turn)
                        pung = Pung(names[turn])
                        pungm, pungh = pung[0], pung[1]
                        battle.write('Game %d\n' % (games + 1))
                        Calc(turnnum, points)
                        for k in range(4):
                            battle.write('%s,%d\n' % (order[k], points[k]))
                        battle.write('\n')
                        break
                    names[turn].remove(lines[current[turnnum]][turnnum])
                    current[turnnum] += 1
                    turnnum = (turnnum + 1) % 4
                if i == 20 and not flag:
                    winner.write('Draw\n')
                    battle.write('Game %d\n' % (games + 1))
                    for k in range(4):
                        battle.write('%s,%d\n' % (order[k], points[k]))
                    battle.write('\n')
                    flag = 1
                if flag:
                    games += 1
                    xs.append(games)
                    for k in range(4):
                        ys[k].append(points[k])
                    pungm = pungh = 0
                    dealer = order[(order[dealer] + 1) % 4]
                    break
        except IndexError:
            break
    winner.write('\n')
    for i in range(4):
        winner.write('%s,%.2f%%\n' % (order[i], wins[i] / games * 100))
    wintile = sorted(wintile.items(), key=lambda x: (-x[1], x[0]))
    for i in wintile:
        tile.write('%s,%.2f%%\n' % (i[0], i[1] / games * 100))
    winner.close()
    tile.close()
    battle.close()


from matplotlib import pyplot as plt

pungh = pungm = games = 0
xs = []
ys = [[] for i in range(4)]
order = {}
solve('test.csv')
plt.xticks(xs)
plt.title('Result')
for i in range(4):
    plt.plot(xs, ys[i], label=order[i])
    plt.legend(loc='best')
plt.show()
