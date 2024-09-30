def hannuota(num,start,end,mid):
    if num == 1:
        print("{}-->{}".format(start,end))
    else:
        hannuota(num-1,start,mid,end)
        print("{}-->{}".format(start,end))
        hannuota(num-1,mid,end,start)

print("汉诺塔问题：")
hannuota(3,'A','C','B')