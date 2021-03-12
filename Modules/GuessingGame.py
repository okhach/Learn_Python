import sys
import random

first = int(sys.argv[1])
second = int(sys.argv[2])

number = random.randint(first, second)
guess = int(input(f"Please guess a number between {first} and {second}: "))

count = 1

while True:
    if count < 5:
        if guess == number:
            print("Congrats!")
            break
        else:
            count = count + 1
            guess = int(input("Please enter another number: "))
    else:
        print(f"You lose! The number is {number}")
        break
