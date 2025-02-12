import copy
import math
import numpy as np
from matplotlib import pyplot as pp

# School miniproject, solving sudoku using simulated annealing

class Sudoku:
    def __init__(self, board):
        self.board = board
        self.fixed = copy.deepcopy(board) 

    def display(self):
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("-" * 21)
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    print("|", end=" ")
                print(self.board[i][j], end=" ")
            print()

    def error_one(self, numbers):
        error=0
        for i in range(9):
            error+=abs(numbers[i]-1)
        return error

    # First method, improved version bellow
    def error_total_1(self):
        error=0
        for i in range(9):
            numbers = [0] * 9
            for j in range(9):
                numbers[self.board[i][j] - 1] += 1
            error += self.error_one(numbers)

        for i in range(9):
            numbers = [0] * 9
            for j in range(9):
                numbers[self.board[j][i] - 1] += 1
            error += self.error_one(numbers)

        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                numbers = [0] * 9
                for k in range(3):
                    for l in range(3):
                        numbers[self.board[i + k][j + l] - 1] += 1
                error += self.error_one(numbers)
        
        return error
    
    # Checking error only across blocks and 1 dimension, in the other its allways correct
    def error_total_2(self):
        error=0
        for i in range(9):
            numbers = [0] * 9
            for j in range(9):
                numbers[self.board[j][i] - 1] += 1
            error += self.error_one(numbers)

        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                numbers = [0] * 9
                for k in range(3):
                    for l in range(3):
                        numbers[self.board[i + k][j + l] - 1] += 1
                error += self.error_one(numbers)
        
        return error

    # First method, improved version bellow
    def solveAnnealing1(self):
        all_numbers = [i for i in range(1, 10) for _ in range(9)]
        for row in self.board:
            for num in row:
                if num in all_numbers:
                    all_numbers.remove(num)
        np.random.shuffle(all_numbers)
        for i in range(9):
            for j in range(9):
                if self.board[i][j]==0:
                    self.board[i][j]=all_numbers.pop()
        
        T0=2
        T=2
        iter_max=1000000
        error=self.error_total_1()
        for iter in range(iter_max):
            T=T0*(1-iter/iter_max)
            #T=0.999999*T
            #T=math.exp(-5*iter/iter_max)
            i,j,k,l=np.random.randint(0,9,size=4)
            while (i==k and j==l) or self.fixed[i][j]!=0 or self.fixed[k][l]!=0:
                i,j,k,l=np.random.randint(0,9,size=4)
            temp=self.board[i][j]
            self.board[i][j]=self.board[k][l]
            self.board[k][l]=temp
            error_new=self.error_total_1()
            if error_new>error and math.exp((error-error_new)/T)<np.random.random():
                self.board[k][l]=self.board[i][j]
                self.board[i][j]=temp
            else:
                error=error_new
            if iter%10000==0 and True:
                self.display()
                print("Completed:", iter/iter_max)
                print("Temperature:", T)
                print("Error: ", error)
            if error==0:
                self.display()
                print("Completed:", iter/iter_max)
                print("Temperature:", T)
                print("Error: ", error)
                return


    # One dimension is solved from the start, change number only across the second
    def solveAnnealing2(self):
        for i in range(9):
            all_numbers = [i for i in range(1, 10)]
            for j in range(9):
                if self.board[i][j] in all_numbers:
                    all_numbers.remove(self.board[i][j])
            np.random.shuffle(all_numbers)
            for j in range(9):
                if self.board[i][j]==0:
                    self.board[i][j]=all_numbers.pop()
        
        errors=[]
        T0=1
        T=1
        iter_max=1000000
        error=self.error_total_2()
        for iter in range(iter_max):
            T=T0*(1-iter/iter_max)
            #T=0.999999*T
            #T=math.exp(-5*iter/iter_max)
            i,k,l=np.random.randint(0,9,size=3)
            while (k==l) or self.fixed[i][k]!=0 or self.fixed[i][l]!=0:
                i,k,l=np.random.randint(0,9,size=3)
            temp=self.board[i][k]
            self.board[i][k]=self.board[i][l]
            self.board[i][l]=temp
            error_new=self.error_total_2()
            if error_new>error and math.exp((error-error_new)/T)<np.random.random():
                self.board[i][l]=self.board[i][k]
                self.board[i][k]=temp
            else:
                error=error_new
                errors.append(error)
            if iter%10000==0 and True:
                self.display()
                print("Completed:", iter/iter_max)
                print("Temperature:", T)
                print("Error: ", error)
            if error==0:
                self.display()
                print("Completed:", iter/iter_max)
                print("Temperature:", T)
                print("Error: ", error)
                break
        plot=pp.plot(errors)
        pp.ylim(0, 100)
        pp.show()


#priklady https://sandiway.arizona.edu/sudoku/examples.html

easy1 = [
    [0,0,0,2,6,0,7,0,1],
    [6,8,0,0,7,0,0,9,0],
    [1,9,0,0,0,4,5,0,0],
    [8,2,0,1,0,0,0,4,0],
    [0,0,4,6,0,2,9,0,0],
    [0,5,0,0,0,3,0,2,8],
    [0,0,9,3,0,0,0,7,4],
    [0,4,0,0,5,0,0,3,6],
    [7,0,3,0,1,8,0,0,0]
]

easy2 = [
    [1,0,0,4,8,9,0,0,6],
    [7,3,0,0,0,0,0,4,0],
    [0,0,0,0,0,1,2,9,5],
    [0,0,7,1,2,0,6,0,0],
    [5,0,0,7,0,3,0,0,8],
    [0,0,6,0,9,5,7,0,0],
    [9,1,4,6,0,0,0,0,0],
    [0,2,0,0,0,0,0,3,7],
    [8,0,0,5,1,2,0,0,4]
]

intermediate = [
    [0,2,0,6,0,8,0,0,0],
    [5,8,0,0,0,9,7,0,0],
    [0,0,0,0,4,0,0,0,0],
    [3,7,0,0,0,0,5,0,0],
    [6,0,0,0,0,0,0,0,4],
    [0,0,8,0,0,0,0,1,3],
    [0,0,0,0,2,0,0,0,0],
    [0,0,9,8,0,0,0,3,6],
    [0,0,0,3,0,6,0,9,0]
]

notfun = [
    [0, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 6, 0, 0, 0, 0, 3],
    [0, 7, 4, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 0, 2],
    [0, 8, 0, 0, 4, 0, 0, 1, 0],
    [6, 0, 0, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 7, 8, 0],
    [5, 0, 0, 0, 0, 9, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 4, 0]
]

sudoku = Sudoku(easy1)
sudoku.solveAnnealing2()
#input("Press Enter to continue...")
