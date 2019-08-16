import numpy as np
import sys 

class SolveSudoku(object):

    def convert_labels_to_grid(self,labels): 
        grid = [[0 for i in range(9)] for j in range(9)]
        count = 0
        for i in range(9): 
            for j in range(9): 
                grid[i][j] = labels[count]
                count += 1
        return grid

    def __init__(self,labels):
        assert len(labels) == 81 
        self.grid = self.convert_labels_to_grid(labels)
        self.count = 0

    def print_grid(self): 
        for i in range(9): 
            t = ""
            for j in range(9): 
                t += str(self.grid[i][j]) + " "
            print(t)  
    
    def find_empty_location(self,l): 
        for row in range(9):
            for col in range(9): 
                if(self.grid[row][col] == 0): 
                    l[0] = row
                    l[1] = col 
                    return True 
        return False 

    def used_in_row(self,row,num): 
        for i in range(9): 
            if(self.grid[row][i] == num): 
                return True 
        return False 

    def used_in_col(self,col,num): 
        for i in range(9): 
            if(self.grid[i][col] == num): 
                return True 
        return False 

    def used_in_box(self,row,col,num): 
        for i in range(3):
            for j in range(3): 
                if(self.grid[i + row][j + col] == num): 
                    return True 
        return False

    def check_safe(self,row,col,num): 
        return not self.used_in_row(row, num) and not self.used_in_col(col, num) and not self.used_in_box((row - row % 3),(col - col % 3),num)

    def solve_sudoku(self): 
        l = [0, 0]
        if not self.find_empty_location(l):
            return True 

        row = l[0]
        col = l[1]

        for num in range(1,10): 
            if(self.check_safe(row, col, num)): 
                self.grid[row][col] = num 

                if self.solve_sudoku():
                    return True 

                self.grid[row][col] = 0

        return False 

    def solve(self): 
        if self.solve_sudoku(): 
            return self.grid
        print("Cant solve puzzle")
        return None 
        

if __name__ == '__main__':
    labels = [0, 0, 8, 0, 2, 0, 1, 0, 0, 0, 5, 0, 0, 0, 0, 0, 7, 0, 7, 0, 0, 0, 0, 0, 0, 0, 9, 0, 9, 4, 0, 6,
    0, 5, 3, 0, 2, 0, 3, 8, 0, 9, 6, 0, 1, 0, 6, 1, 0, 7, 0, 8, 9, 0, 4, 0, 0, 0, 0, 0, 0, 0, 8, 0, 3, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 5, 0, 4, 0, 3, 0, 0]
    #temp = np.array(labels)
    #grid = np.reshape(temp, (9, 9)) 
    sudoku = SolveSudoku(labels)
    if sudoku.solve_sudoku(): 
        sudoku.print_grid()