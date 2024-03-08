vis = []
def Dfs(flag, num, maj):
    # Win
    if(flag == True and num == 4):
        return True

    for i in range(14): 
        if not vis[i]:
            for j in range(i+1, 14): 
                # They are in the same suit
                if (not vis[j]) and maj[i][1] == maj[j][1]:
                    for k in range(j + 1, 14):
                        if (not vis[k]) and maj[j][1] == maj[k][1]:
                            # Pung
                            if maj[i][0] == maj[j][0] and maj[j][0] == maj[k][0]:
                                vis[i] = vis[j] = vis[k] = True
                                if Dfs(flag, num + 1, maj):
                                    return True
                                vis[i] = vis[j] = vis[k] = False

                            # Chow
                            if maj[i][1] != 'z' and int(maj[i][0]) + 1 == int(maj[j][0]) and int(maj[j][0]) + 1 == int(maj[k][0]):
                                vis[i] = vis[j] = vis[k] = True
                                if Dfs(flag, num + 1, maj):
                                    return True
                                vis[i] = vis[j] = vis[k] = False
                    
                    # Pair
                    if (not flag) and maj[i][0] == maj[j][0]:
                        vis[i] = vis[j] = True
                        if Dfs(True, num, maj):
                            return True
                        vis[i] = vis[j] = False
    # Fail to win
    return False


def CheckWin(maj):
    if len(maj) != 14:
        raise RuntimeError("The number of tiles is NOT equal to 14!")
    global vis
    vis = [False for i in range(14)]
    cnt = {}
    # Sort the series
    for i in range(14):
        for j in range(i + 1, 14):
            if maj[i][1] > maj[j][1]:
                maj[i], maj[j] = maj[j], maj[i]
            elif maj[i][1] == maj[j][1] and maj[i][0] > maj[j][0]:
                maj[i], maj[j] = maj[j], maj[i]
        if maj[i] in cnt:
            cnt[maj[i]] += 1
        else:
            cnt[maj[i]] = 1
        if cnt[maj[i]] > 4:
            raise RuntimeError("There are more than 4 same tiles!")


    return Dfs(False, 0, maj)

