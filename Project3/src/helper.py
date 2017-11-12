#!/usr/bin/env python3

def toArrayRep(index, numberOfClasses):
        array = []
        for i in range(numberOfClasses):
            if (int(i) == int(index)):
                array.append(1)
            else:
                array.append(0)
        return array
    
def arrayCompare(arr1, arr2):
    equal = True
    for i in range(len(arr1)):
        if(arr1[i] != arr2[i]):
            equal = False
    return equal
