def selectP(p,group):
    size = len(group)
    N = size * p
    N = int(N)
    plist = []
    for i in range(size-1):
        for j in range(size-i-1):
            if (GetObjectFunctionValue(group[j]) < GetObjectFunctionValue(group[j+1])):
                tmp = group[j]
                group[j] = group[j+1]
                group[j + 1] = tmp
    j = 0
    while(j<=N-1):
        plist.append(group[j])
        j = j + 1
    print(plist)
    return plist

def GetObjectFunctionValue(X):
    # X [x1,x2]
    return X[0] + X[1]

if __name__ == "__main__":
    selectP(0.5,[[1,2],[3,4],[5,6],[7,8]])