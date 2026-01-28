import torch

def get_one_hot_tags(tags_dict):
    """
    將標籤字典轉換為統一的 15 維 One-Hot 向量
    [0-1] gender, [2-5] age, [6-9] body, [10-12] season, [13-14] formal
    """
    vec = torch.zeros(15)
    
    # 輔助函式：確保讀取值為字串且小寫
    def get_val(key):
        return str(tags_dict.get(key, "")).lower().strip()

    # 1. Gender [0, 1]
    g = get_val('gender')
    if g == 'male': vec[0] = 1
    elif g == 'female': vec[1] = 1
    
    # 2. Age [2-5]
    a = get_val('age')
    if a == 'teenager': vec[2] = 1
    elif a == 'adult': vec[3] = 1
    elif a == 'middle-aged': vec[4] = 1
    elif a == 'elderly': vec[5] = 1
    
    # 3. Body [6-9]
    b = get_val('body')
    if b == 'average': vec[6] = 1
    elif b == 'skinny': vec[7] = 1
    elif b == 'athletic': vec[8] = 1
    elif b == 'plus_size': vec[9] = 1
    
    # 4. Season [10-12]
    s = get_val('season')
    if s == 'summer': vec[10] = 1
    elif s == 'winter': vec[11] = 1
    else: vec[12] = 1 # spring/fall
    
    # 5. Formal [13-14]
    f = get_val('formal')
    if f == 'formal': vec[14] = 1
    else: vec[13] = 1 # casual

    return vec