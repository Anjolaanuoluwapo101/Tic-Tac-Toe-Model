import random
import joblib
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#declare global variable
#no global variable was used


#suppress warnings because the Input_X doesn't have feature names and we want to keep the code simple by not adding too much of that
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

# Display the tic-tac-toe board
def display_board(board):
    for i in range(0, 9, 3):
        print(" | ".join(board[i:i+3]))
        if i < 6:
            print("---------")

# this function that allows the model to play
def model_move(model, board):
    empty_cells = [i for i, cell in enumerate(board) if cell == " "]
    X_input = [0 if cell == " " else 1 if cell == "O" else 2 for cell in board]
    #print(X_input)
    move = model.predict([X_input])[0]
    if move in empty_cells:
        return move
    else:
        return random.choice(empty_cells)
        
 
# Function to check for a winfor either the human or AI depending on which player and occupied board passed
def check_win(player, board):
    win_combinations = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
        (0, 4, 8), (2, 4, 6)  # Diagonals
    ]
    for combination in win_combinations:
        if all(board[i] == player for i in combination):
            return True
    return False

#Function to continue the game or quit the game
def continuity(player):	
	play_again = input("Do you want to continue playing? (y/n): ").lower()
	if play_again == "y":
		print("here")
		if player == "X":
			human_turn = False
			return play_game(model,human_turn)#start another new game entirely..the previous play_game function is stopped and the new one starts
		elif player == "O":
			human_turn = True
			return play_game(model,human_turn) #start another new game entirely..the previous play_game function is stopped and the new one starts
	else:
		return False
 
#GAME STARTS and model is first established.....
#model = joblib.load('existing_models/tic-tac-toe-endgame.pkl')  # Load your trained model here
#create a trained model
data = pd.read_csv("datasets/tic-tac-toe-encoded.csv")
#data = pd.read_csv("tictactoe_encodedV2.csv")
# Prepare the data
X = data.iloc[:, :-1]  # Features..9 columns
y = data.iloc[:, -1]   # Target..the last 1 column

# Create a Random Forest classifier

model = RandomForestClassifier(n_estimators=700,random_state=100)
model.fit(X,y)


def play_game(model,human_turn):
    board = [" " for _ in range(9)]
    winner = False

    while " " in board:
    	
    	#responsible for human turn
        if human_turn:
            display_board(board)
            #check if human_move was even an integer..they can be dubious
            try:
            	human_move = int(input("Enter your move (0-8) \n Top-Left = 0, Top-Middle = 1, Top-Right = 2 \n Middle-Left = 3 ..... \n You get the drill : "))
            except ValueError:
            	print("Only integers allowed!!!")
            	continue #skips current iteration for the next,in which the human players get to make a valid move...
            if board[human_move] == " ":
                board[human_move] = "X"
                if check_win("X", board):
                    display_board(board)
                    print("You win!")
                    winner = True
                    continuity("X")
                    break
                else:
                    human_turn = False
            else:
                print("That position is already occupied") #humans can be dubious..by trying to play in the same place that has already being played in
                continue

		#reponsible for AI turn
        else:
            model_idx = model_move(model, board)
            board[model_idx] = "O"
            if check_win("O", board):
                display_board(board)
                print("Model wins!")
                winner = True;
                continuity("O")
                break 
                #if continuity doesn't return False then this never runs and the while loop continues the game
            else:
                human_turn = True
			
    if " " not in board and winner:
        display_board(board)
        print("It's a tie!")
        #we head over to continuity and pass the current player
        if human_turn:
        	continuity("X")
        else:
        	continuity("O")
        
#start game
play_game(model,True)