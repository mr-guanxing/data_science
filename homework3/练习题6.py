import time
import random
def select_sort(num):
    for i in range(len(num)):
        min_index = i
        for j in range(i + 1, len(num)):
            if num[j] < num[min_index]:
                min_index = j
        num[i], num[min_index] = num[min_index], num[i]
    return num

def generate_random_list(size):
    return [random.randint(1, 1000) for _ in range(size)]


start_time = time.time()
random_list = generate_random_list(1024)
select_sort(random_list)
print("sorted list:", random_list)
end_time = time.time()
duration = end_time - start_time
print("it cost {} seconds".format(duration))