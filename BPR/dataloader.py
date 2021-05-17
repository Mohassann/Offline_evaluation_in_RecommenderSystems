
def LoadRatingFile_HoldKOut_year(path, splitter, K, selected_year, num_years_added):
    data_filename = path+'data.data.RATING'
    train_filename = path+'train.train.RATING'
    test_filename = path+'test.train.RATING'

    splitter='\t'
    K=1
    num_ratings = 0
    num_item = 0
    import collections

    train = {}
    
    
    test = []
    available_items = set()
    num_ratings = 0
    num_item = 0
    num_user = 0
    with open(data_filename, "r") as f:
        for line in f:
            arr = line.split(splitter)
    
            user, item, year= int(arr[0]), int(arr[1]), int(arr[4])
            if (year < selected_year) or (year > selected_year and year <= selected_year+num_years_added):
                if user in train:
                    train[user].append(item)
                else:
                    train[user] = [item]
                num_ratings += 1
                num_user = max(num_user, user)
                num_item = max(num_item, item)
                available_items.add(item)
                    


    with open(train_filename, "r") as f:
        for line in f:
            arr = line.split(splitter)

            user, item, year= int(arr[0]), int(arr[1]), int(arr[4])
            if (year  == selected_year):
                num_ratings += 1
                if user in train:
                    train[user].append(item)
                else:
                    train[user] = [item]
                num_user = max(num_user, user)
                num_item = max(num_item, item)
                available_items.add(item)

    num_user += 1
    num_item = num_item + 1
    
    test = []
    with open(test_filename, "r") as f:
        for line in f:
            arr = line.split(splitter)
            user, item, year= int(arr[0]), int(arr[1]), int(arr[4]) 
            if (year  == selected_year):
                test.append([user, item, year])

    return train, test, available_items, num_user, num_item, num_ratings



def LoadRatingFile_HoldKOut_year_test(path, splitter, K, selected_year, num_years_added):
    data_filename = path+'data.data.RATING'
    train_filename = path+'train.test.RATING'
    test_filename = path+'test.test.RATING'

    splitter='\t'
    K=1
    num_ratings = 0
    num_item = 0
    import collections

    train = {}
    
    
    test = []
    available_items = set()
    num_ratings = 0
    num_item = 0
    num_user = 0
    with open(data_filename, "r") as f:
        for line in f:
            arr = line.split(splitter)
    
            user, item, year= int(arr[0]), int(arr[1]), int(arr[4])
            if (year < selected_year) or (year > selected_year and year <= selected_year+num_years_added):
                if user in train:
                    train[user].append(item)
                else:
                    train[user] = [item]
                num_ratings += 1
                num_user = max(num_user, user)
                num_item = max(num_item, item)
                available_items.add(item)
                    


    with open(train_filename, "r") as f:
        for line in f:
            arr = line.split(splitter)

            user, item, year= int(arr[0]), int(arr[1]), int(arr[4])
            if (year  == selected_year):
                num_ratings += 1
                if user in train:
                    train[user].append(item)
                else:
                    train[user] = [item]
                num_user = max(num_user, user)
                num_item = max(num_item, item)
                available_items.add(item)

    num_user += 1
    num_item = num_item + 1
    
    test = []
    with open(test_filename, "r") as f:
        for line in f:
            arr = line.split(splitter)
            user, item, year= int(arr[0]), int(arr[1]), int(arr[4]) 
            if (year  == selected_year):
                test.append([user, item, year])

    return train, test, available_items, num_user, num_item, num_ratings
