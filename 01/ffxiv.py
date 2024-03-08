def solve(data):
    ans = 16
    zero = zeros = 0
    for i in range(4):
        if data[i][i] == 0:
            zero += 1
            data[i][i] = 2
    zeros = zero
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            for k in range(4):
                if data[i][k] == 0:
                    zeros += 1
                    data[i][k] = 2
            for k in range(4):
                if data[k][j] == 0:
                    zeros += 1
                    data[k][j] = 2
            if ans > zeros:
                ans = zeros
            zeros = zero
            for k in range(4):
                for l in range(4):
                    if data[k][l] == 2:
                        data[k][l] = 0
    zeros = zero = 0
    for i in range(4):
        if data[i][3 - i] == 0:
            zero += 1
            data[i][3 - i] = 2
    zeros = zero
    for i in range(4):
        for j in range(4):
            if i == 3 - j:
                continue
            for k in range(4):
                if data[i][k] == 0:
                    zeros += 1
                    data[i][k] = 2
            for k in range(4):
                if data[k][j] == 0:
                    zeros += 1
                    data[k][j] = 2
            if ans > zeros:
                ans = zeros
            zeros = zero
            for k in range(4):
                for l in range(4):
                    if data[k][l] == 2:
                        data[k][l] = 0
    return ans


def main():
    data = [[1, 1, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]]
    print(solve(data))


if __name__ == "__main__":
    main()
