# Table of Contents
1. [Introduction](#Introduction)
2. [Problem representation](#Problem-representation)
3. [Input data](#Input-data)
4. [Output data](#Output-data)
5. [Heuristics](#Heuristics)
6. [Code optimizations](#Code-optimizations)
7. [Algorithm comparasion](#Algorithm-comparasion)
8. [Running the program](#Running-the-program)

# Introduction

[Klotski](https://en.wikipedia.org/wiki/Klotski) is a sliding block puzzle. The goal of this game is to move each piece in any of the 4 directions (up, right, down, left) until you manage to get the special piece out of the board (usually through an exit in the top border). 

The original Klotski puzzle consisted of 10 blocks and can be solve in a minimum of 81 moves:

<img src="images/original_puzzle.png" 
     alt="original puzzle" 
     width="200px"
     height="230px"
     style = "position: relative; margin: 15px 1em 0px 1em"/>

# Problem representation

### **Encoding**:
In order to memorize a board we need to encode the configuration:
1. '#' - border of the current state
2. '.' - free space in the board
3. '*' - part of the special piece
4. 'a'-'z' or 'A'-'Z' for other pieces

Each board configuration is stored in the memory via the SearchNode class.

### **Rules**:
* The puzzle has an exit at the top of the board which fits perfectly with the special piece
* Each piece can be moved one step at a time, in any of the 4 directions (top, right, down left) only if it has enoguh free space to move there
* Each piece has to be moved entirely (you cannot break the piece and move only parts of it)
* Only the special piece can exit the puzzle through the exit at the top
* The cost of each move is equal to the size of the pieced move (except the special piece which always has a cost of 1)
* The puzzle is considered solved when the special piece is no longer on the board

### **Solving the problem**:
In order to solve this problem we will represent each set of moves as a path in a graph, using the Graph class. We will try to build this path step by step using diffrent searching algorithms.
