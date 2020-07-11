def levenshtein_distance(errs, corrs):
    errs_len = len(errs)
    corrs_len = len(corrs)
    inf = float("inf")
    dp = [[inf for i in range(corrs_len+1)] for j in range(errs_len+1)]
    dp[0][0] = 0

    for i in range(-1, errs_len):
        for j in range(-1, corrs_len):
            if i == -1 and j == -1:
                continue
            elif i >= 0 and j >= 0:
                if errs[i].lower() == corrs[j].lower():
                    dp[i+1][j+1] = min(dp[i][j], dp[i][j+1] + 1, dp[i+1][j] + 1)
                else:
                    dp[i+1][j+1] = min(dp[i][j] + 1, dp[i][j+1] + 1, dp[i+1][j] + 1)
            elif i >= 0:
                dp[i+1][j+1] = dp[i][j+1] + 1
            elif j >= 0:
                dp[i+1][j+1] = dp[i+1][j] + 1

    del_num, ins_num = 0, 0
    while i > 0 and j > 0:
        dp_val = [dp[i-1][j-1], dp[i-1][j], dp[i][j-1]]
        min_idx = dp_val.index(min(dp_val))
        if dp[i][j] == dp[i-1][j-1] and min_idx == 0:
            i -= 1
            j -= 1
            continue
        elif min_idx == 0:
            del_num += 1
            ins_num += 1
            i -= 1
            j -= 1
        elif min_idx == 1:
            del_num += 1
            i -= 1
        else:
            ins_num += 1
            j -= 1
    return del_num, ins_num


if __name__ == "__main__":
    s1 = "車を買う"
    s2 = "車で買う"
    l = levenshtein_distance(s1, s2)
    print(l)
