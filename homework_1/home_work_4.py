def sqrt_2():
    min_value = 1
    max_value = 2
    epsilon = 1e-6  
    value = (min_value+max_value)/2
    while abs(value**2-2)>epsilon:
        value = (min_value+max_value)/2
        if value * value < 2:
            min_value = value
        else:
            max_value = value  
    return value

result = sqrt_2()
print(f"根号2的近似值是: {result}")
