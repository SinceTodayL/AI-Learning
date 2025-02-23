#include<iostream>
#include<string>
#include<vector>
#include<set>
#include<algorithm>
#include<random>
#include<unordered_map>
using namespace std;

// Compile-time constants for board and gameplay configurations
constexpr int board_length = 12;
constexpr int middle_position = board_length / 2 - 1;
constexpr int condition_of_success = 5;
constexpr int directions[4][2] = { {1, 0}, {0, 1}, {1, 1}, {1, -1} };
constexpr int recursion_depth = 6;
constexpr int next_cell_num = 6;

// Enum for cell states on the board
enum class CellState {
	BLANK,
	MY,
	OPPONENT    // Opponent's piece
};

// Operator to switch between MY and OPPONENT states
CellState operator!(CellState turn) {
	if (turn != CellState::BLANK)
		return (turn == CellState::MY) ? CellState::OPPONENT : CellState::MY;
	else
		return CellState::BLANK;
}

// Class representing a position on the board
class Position {
public:
	int x, y;

	Position() {
		this->x = 0;
		this->y = 0;
	}
	Position(int posx, int posy) : x(posx), y(posy) {}

	inline bool isValid() const {
		return this->x >= 0 && this->x < board_length && this->y >= 0 && this->y <= board_length;
	}

	// Overload == and != operators for position comparison
	inline bool operator==(const Position& pos) const {
		return this->x == pos.x && this->y == pos.y;
	}

	inline bool operator!=(const Position& pos) const {
		return this->x != pos.x || this->y != pos.y;
	}
};

// Inherited from Position, representing a move and its score
class Movement : public Position {
public:
	int score;
	Movement() {
		this->score = 0;
	}
	Movement(int move_score) {
		this->score = move_score;
	}
	Movement(int posx, int posy, int move_score) {
		this->x = posx;
		this->y = posy;
		this->score = move_score;
	}
};

// Namespace for scoring patterns of player's pieces
namespace MyState {

	constexpr int Five = 40;
	constexpr int ActiveFour = 30;
	constexpr int RushFour = 10;
	constexpr int ActiveThree = 5;

	// Patterns representing various scoring positions for MY pieces
	vector<vector<CellState>> FiveState = { {CellState::MY, CellState::MY, CellState::MY, CellState::MY, CellState::MY } };
	vector<vector<CellState>> ActiveFourState = { { CellState::BLANK, CellState::MY, CellState::MY, CellState::MY, CellState::MY, CellState::BLANK } };
	vector<vector<CellState>> RushFourState = {
		{ CellState::OPPONENT, CellState::MY, CellState::MY, CellState::MY, CellState::MY, CellState::BLANK },
		{ CellState::BLANK, CellState::MY, CellState::MY, CellState::MY, CellState::MY, CellState::OPPONENT },
		{ CellState::MY, CellState::BLANK, CellState::MY, CellState::MY, CellState::MY },
		{ CellState::MY, CellState::MY, CellState::BLANK, CellState::MY, CellState::MY },
		{ CellState::MY, CellState::MY, CellState::MY, CellState::BLANK, CellState::MY }
	};
	vector<vector<CellState>> ActiveThreeState = {
		{ CellState::BLANK, CellState::MY, CellState::MY, CellState::MY, CellState::BLANK },
		{ CellState::BLANK, CellState::MY, CellState::MY, CellState::BLANK, CellState::MY, CellState::BLANK },
		{ CellState::BLANK, CellState::MY, CellState::BLANK, CellState::MY, CellState::MY, CellState::BLANK }
	};
	vector<vector<CellState>> SpecialMyRushFourState = {
		{ CellState::MY, CellState::MY, CellState::MY, CellState::MY, CellState::BLANK },
		{ CellState::BLANK, CellState::MY, CellState::MY, CellState::MY, CellState::MY }
	};

	const vector<vector<vector<CellState>>> mystate = { FiveState, ActiveFourState, RushFourState, ActiveThreeState };
	const vector<int> score = { Five, ActiveFour, RushFour, ActiveThree };
}

// Namespace for scoring patterns of opponent's pieces
namespace OpState {

	vector<vector<CellState>> SpecialOpRushFourState = {
		{ CellState::OPPONENT, CellState::OPPONENT, CellState::OPPONENT, CellState::OPPONENT, CellState::BLANK },
		{ CellState::BLANK, CellState::OPPONENT, CellState::OPPONENT, CellState::OPPONENT, CellState::OPPONENT }
	};

	constexpr int Five = -60;
	constexpr int ActiveFour = -40;
	constexpr int RushFour = -20;
	constexpr int ActiveThree = -10;

	const vector<int> score = { Five, ActiveFour, RushFour, ActiveThree };
}


// Describes the entire chessboard and manages game logic
class ChessBoard {
private:
	vector<vector<CellState>> board; // 2D vector representing the board cells and their states

public:
	// Zobrist Hashing table for each board cell and state
	uint64_t zobrist_table[board_length][3]; // Each cell has 3 states: BLANK, MY, OPPONENT
	unordered_map<uint64_t, vector<int>> transposition_table_my; // Stores hash values for player moves
	unordered_map<uint64_t, vector<int>> transposition_table_op; // Stores hash values for opponent moves

	// Initializes the Zobrist hashing table with random values for each cell and state
	void InitializeZobristTable();

	// Constructor: initializes an empty board
	ChessBoard() : board(board_length, vector<CellState>(board_length, CellState::BLANK)) {}

	// Initializes the board with specific starting pieces for each player
	void Init(const CellState& wt);

	// Updates board with opponent's move
	inline void OpMove(const Position& pos);

	// Updates board with player's move
	inline void MyMove(const Position& pos);

	// Checks if a position has any neighboring pieces, indicating it might be a strategic spot
	bool HasNeighbor(int x, int y);

	// Calculates the current score for the given board configuration
	int CalculateCurrentScore(const vector<vector<CellState>>& cur_board);

	// Checks if a specific direction on the board has a winning condition for a player
	bool CheckDirection(const vector<vector<CellState>>& cur_board, int x, int y, int dx, int dy, CellState player);

	// Determines if the game is over based on the current board state
	bool GameIsOver(const vector<vector<CellState>>& cur_board);

	// Evaluates the score based on a line of cells in a specific axis direction
	void AxisScore(vector<CellState> cur_line, CellState who, vector<int>& Type);

	// Returns a list of potential next moves based on current board state and player
	vector<Position> NextCell(vector<vector<CellState>> chessboard, CellState player);

	// Chooses the best move based on board evaluation and scoring
	Position BestMovement();

	// Implements Alpha-Beta pruning to evaluate the best move within a given depth and score range
	Movement AlphaBeta(vector<vector<CellState>>& cur_board, int alpha, int beta, CellState player, int depth);
};

// Initializes the board with four starting pieces, based on initial player state
void ChessBoard::Init(const CellState& wt) {
	board[middle_position][middle_position] = (wt == CellState::MY) ? CellState::OPPONENT : CellState::MY;
	board[middle_position + 1][middle_position + 1] = (wt == CellState::MY) ? CellState::OPPONENT : CellState::MY;
	board[middle_position][middle_position + 1] = (wt != CellState::MY) ? CellState::OPPONENT : CellState::MY;
	board[middle_position + 1][middle_position] = (wt != CellState::MY) ? CellState::OPPONENT : CellState::MY;
}

// Sets up random values for each cell and state in the Zobrist hashing table
void ChessBoard::InitializeZobristTable() {
	random_device rd;
	mt19937_64 gen(rd());
	uniform_int_distribution<uint64_t> dist;

	for (int i = 0; i < board_length; ++i) {
		for (int k = 0; k < 3; ++k) {  // 3 states: BLANK, MY, OPPONENT
			zobrist_table[i][k] = dist(gen);
		}
	}
}

// Records opponent's move on the board if valid
inline void ChessBoard::OpMove(const Position& pos) {
	if (pos.isValid()) {
		this->board[pos.x][pos.y] = CellState::OPPONENT;
	}
}

// Records player's move on the board if valid
inline void ChessBoard::MyMove(const Position& pos) {
	if (pos.isValid()) {
		this->board[pos.x][pos.y] = CellState::MY;
	}
}

// Checks if there are any adjacent pieces around a given position
bool ChessBoard::HasNeighbor(int x, int y) {
	if (board[x][y] == CellState::BLANK) {
		if ((x > 0 && board[x - 1][y] != CellState::BLANK) ||
			(x < board_length - 1 && board[x + 1][y] != CellState::BLANK) ||
			(y > 0 && board[x][y - 1] != CellState::BLANK) ||
			(y < board_length - 1 && board[x][y + 1] != CellState::BLANK) ||
			(x > 0 && y > 0 && board[x - 1][y - 1] != CellState::BLANK) ||
			(x > 0 && y < board_length - 1 && board[x - 1][y + 1] != CellState::BLANK) ||
			(x < board_length - 1 && y > 0 && board[x + 1][y - 1] != CellState::BLANK) ||
			(x < board_length - 1 && y < board_length - 1 && board[x + 1][y + 1] != CellState::BLANK) ||
			(x > 1 && board[x - 2][y] != CellState::BLANK) ||
			(x < board_length - 2 && board[x + 2][y] != CellState::BLANK) ||
			(y > 1 && board[x][y - 2] != CellState::BLANK) ||
			(y < board_length - 2 && board[x][y + 2] != CellState::BLANK) || 
			(x > 1 && y > 1 && board[x - 2][y - 2] != CellState::BLANK) ||
			(x < board_length - 2 && y < board_length - 2 && board[x + 2][y + 2] != CellState::BLANK) ||
			(x > 1 && y < board_length - 2 && board[x - 2][y + 2] != CellState::BLANK) ||
			(x < board_length - 2 && y > 1 && board[x + 2][y - 2] != CellState::BLANK)
			)
			return true;
	}
	return false;
}

// Finds the possible moves based on having neighboring pieces and evaluates each move's score
vector<Position> ChessBoard::NextCell(vector<vector<CellState>> chessboard, CellState player) {
	vector<Position> hasneighbor;  // Tracks cells with nearby pieces
	vector<Movement> nextcell;     // Stores moves with calculated scores
	vector<Position> final_nextcell;

	for (int i = 0; i < board_length; ++i) {
		for (int j = 0; j < board_length; ++j) {
			if (HasNeighbor(i, j))
				hasneighbor.push_back(Position(i, j));
		}
	}

	vector<CellState> cur_line; // Holds a row or column for scoring

	for (Position pos : hasneighbor) {
		vector<int> MyType(MyState::score.size(), 0); // Scoring breakdown for player
		vector<int> OpType(MyState::score.size(), 0); // Scoring breakdown for opponent

		// Simulate placing the player's piece at current position
		chessboard[pos.x][pos.y] = (player == CellState::MY) ? CellState::MY : CellState::OPPONENT;

		// Calculate score by rows and columns
		cur_line = chessboard[pos.x];
		AxisScore(cur_line, CellState::MY, MyType);

		cur_line.clear();
		for (int i = 0; i < board_length; ++i) {
			cur_line.push_back(chessboard[i][pos.y]);
		}
		AxisScore(cur_line, CellState::MY, MyType);

		// Calculate score by diagonals
		cur_line.clear();
		for (int j = max(0, pos.x + pos.y - (board_length - 1)); j <= min(pos.x + pos.y, board_length - 1); ++j) {
			cur_line.push_back(chessboard[pos.x + pos.y - j][j]);
		}
		AxisScore(cur_line, CellState::MY, MyType);

		cur_line.clear();
		for (int j = max(0, -(pos.x - pos.y)); j <= min(board_length - 1, board_length - 1 - (pos.x - pos.y)); ++j) {
			cur_line.push_back(chessboard[pos.x - pos.y + j][j]);
		}
		AxisScore(cur_line, CellState::MY, MyType);

		// Sum up player's score types
		int all_score_my = 0;
		if (MyType[0])
			all_score_my = MyState::score[0];
		else if ((MyType[2] && MyType[3])|| MyType[3] >= 2)
			all_score_my = MyState::score[1];
		else{
			for (int type = 0; type < static_cast<int>(MyType.size()); ++type) {
				all_score_my += MyState::score[type] * MyType[type];
			}
		}

		// Repeat scoring for opponent moves
		chessboard[pos.x][pos.y] = CellState::OPPONENT;

		cur_line.clear();
		cur_line = chessboard[pos.x];
		AxisScore(cur_line, CellState::OPPONENT, OpType);

		cur_line.clear();
		for (int i = 0; i < board_length; ++i) {
			cur_line.push_back(chessboard[i][pos.y]);
		}
		AxisScore(cur_line, CellState::OPPONENT, OpType);


		cur_line.clear();
		for (int j = max(0, pos.x + pos.y - (board_length - 1)); j <= min(pos.x + pos.y, board_length - 1); ++j) {
			cur_line.push_back(chessboard[pos.x + pos.y - j][j]);
		}
		AxisScore(cur_line, CellState::OPPONENT, OpType);

		cur_line.clear();
		for (int j = max(0, -(pos.x - pos.y)); j <= min(board_length - 1, board_length - 1 - (pos.x - pos.y)); ++j) {
			cur_line.push_back(chessboard[pos.x - pos.y + j][j]);
		}
		AxisScore(cur_line, CellState::OPPONENT, OpType);

		// Calculate score differential and add move to list
		int all_score_op = 0;
		if (OpType[0])
			all_score_op = OpState::score[0];
		else if ((OpType[2] && OpType[3])|| OpType[3] >= 2)
			all_score_op = OpState::score[1];
		else{
			for (int type = 0; type < static_cast<int>(MyType.size()); ++type) {
				all_score_op += OpState::score[type] * OpType[type];
			}
		}
		chessboard[pos.x][pos.y] = CellState::BLANK; // Reset to blank
		nextcell.push_back(Movement(pos.x, pos.y, all_score_my - all_score_op));
	}

	// Sort moves by score, keeping the top N moves
	sort(nextcell.begin(), nextcell.end(), [](Movement a, Movement b) { return a.score > b.score; });

	for (int i = 0; i < min(next_cell_num, static_cast<int>(nextcell.size())); ++i) {
		final_nextcell.push_back(nextcell[i]);
	}

	return final_nextcell; // Returns evaluated next moves
}

// Computes the best move based on Alpha-Beta pruning results
Position ChessBoard::BestMovement() {
	Movement best_move = AlphaBeta(this->board, INT_MIN, INT_MAX, CellState::MY, recursion_depth);
	Position best_position(best_move.x, best_move.y);

	return best_position;
}

// Checks for a sequence of 5 identical pieces in a specified direction
bool ChessBoard::CheckDirection(const vector<vector<CellState>>& cur_board,
	int x, int y, int dx, int dy, CellState player) {

	for (int i = 0; i < condition_of_success; ++i) {
		int nx = x + i * dx;
		int ny = y + i * dy;

		// Verify each cell in the direction, return false if any is invalid
		if (nx < 0 || ny < 0 || nx >= board_length || ny >= board_length
			|| cur_board[nx][ny] != player) {
			return false;
		}
	}
	return true; // Returns true if 5 in a row is achieved
}

// Checks if the game has ended with 5 in a row or if the board is full
bool ChessBoard::GameIsOver(const vector<vector<CellState>>& cur_board) {

	bool empty_flag = true;
	for (int i = 0; i < board_length; ++i) {
		for (int j = 0; j < board_length; ++j) {
			if (board[i][j] != CellState::BLANK)
				empty_flag = false;
			// Checks all directions from each position
			for (auto dir : directions)
				if (CheckDirection(cur_board, i, j, dir[0], dir[1], CellState::MY) ||
					CheckDirection(cur_board, i, j, dir[0], dir[1], CellState::OPPONENT))
					return true;
		}
	}
	if (empty_flag)
		return true; // Returns true if board is empty or full
	return false;
}

// Evaluates each row or column by matching specific patterns
void ChessBoard::AxisScore(vector<CellState> cur_line, CellState who, vector<int>& Type) {

	uint64_t hash_value = 0; // Generates hash value for current line
	for (int i = 0; i < static_cast<int>(cur_line.size()); ++i) {
		hash_value ^= this->zobrist_table[i][static_cast<int>(cur_line[i])];
	}

	vector<int> _Type(MyState::mystate.size(), 0);

	// Check if this line configuration exists in the transposition table for player or opponent
	if (who == CellState::MY) {
		if (this->transposition_table_my.find(hash_value) != transposition_table_my.end()) {
			_Type = transposition_table_my[hash_value];
			for (int i = 0; i < static_cast<int>(_Type.size()); ++i)
				Type[i] += _Type[i];
			return; // Return if hash is found
		}
	}
	else if (who == CellState::OPPONENT) {
		if (this->transposition_table_op.find(hash_value) != transposition_table_op.end()) {
			_Type = transposition_table_op[hash_value];
			for (int i = 0; i < static_cast<int>(_Type.size()); ++i)
				Type[i] += _Type[i];
			return;
		}
	}

	// If no hash match, check for specific patterns and add them to the table if they match
	if (static_cast<int>(cur_line.size()) <= condition_of_success) {
		if (who == CellState::MY) {
			for (auto cur_state : MyState::SpecialMyRushFourState) {
				if (cur_line == cur_state) {
					_Type[2]++;
				}
			}
		}
		else {
			for (auto cur_state : OpState::SpecialOpRushFourState) {
				if (cur_line == cur_state) {
					_Type[2]++;
				}
			}
		}
	}

	// Searches for all possible matches along the line
	for (int type = 0; type < static_cast<int>(Type.size()); ++type) {
		for (auto cur_state : MyState::mystate[type]) {
			int fptr = 0, cnt = 0;
			bool flag = false;
			// Scan for pattern match; updates blank cells in cur_line if matched
			for (fptr = 0; fptr < static_cast<int>(cur_line.size()); ++fptr) {
				if (who == CellState::MY ? (cur_line[fptr] == cur_state[cnt]) : (cur_line[fptr] == !cur_state[cnt])) {
					while (who == CellState::MY ? (cur_line[fptr] == cur_state[cnt]) : (cur_line[fptr] == !cur_state[cnt])) {
						++fptr;
						++cnt;
						if (cnt == static_cast<int>(cur_state.size())) {
							for (int i = fptr - cnt; i < fptr; ++i)
								cur_line[i] = CellState::BLANK;
							_Type[type]++;
							flag = true;
							break;
						}
						else if (fptr >= static_cast<int>(cur_line.size()))
							break;
					}
					if (flag) {
						cnt = 0;
						fptr--;
					}
					else {
						fptr -= cnt;
						cnt = 0;
						flag = false;
					}
				}
			}
		}
	}

	// Save results in transposition table for future lookups
	if (who == CellState::MY) {
		transposition_table_my[hash_value] = _Type;
	}
	else if (who == CellState::OPPONENT) {
		transposition_table_op[hash_value] = _Type;
	}
	for (int i = 0; i < static_cast<int>(_Type.size()); ++i) {
		Type[i] += _Type[i];
	}
}

// Computes the total score for the current board by analyzing lines in all directions
int ChessBoard::CalculateCurrentScore(const vector<vector<CellState>>& cur_board) {

	vector<CellState> cur_line;
	vector<int> MyType(MyState::mystate.size(), 0);
	vector<int> OpType(MyState::mystate.size(), 0);

	// Evaluate each row
	for (int i = 0; i < board_length; ++i) {
		cur_line = cur_board[i];
		AxisScore(cur_line, CellState::MY, MyType);
		AxisScore(cur_line, CellState::OPPONENT, OpType);
	}

	// Evaluate each column
	for (int j = 0; j < board_length; ++j) {
		cur_line.clear();
		for (int i = 0; i < board_length; ++i) {
			cur_line.push_back(cur_board[i][j]);
		}
		AxisScore(cur_line, CellState::MY, MyType);
		AxisScore(cur_line, CellState::OPPONENT, OpType);
	}

	// Evaluate each main diagonal (x + y = constant)
	for (int all = condition_of_success - 1; all <= 2 * (board_length - 1) - condition_of_success + 1; ++all) {
		cur_line.clear();
		for (int j = max(0, all - (board_length - 1)); j <= min(all, board_length - 1); ++j) {
			cur_line.push_back(cur_board[all - j][j]);
		}
		AxisScore(cur_line, CellState::MY, MyType);
		AxisScore(cur_line, CellState::OPPONENT, OpType);
	}

	// Evaluate each anti-diagonal (x - y = constant)
	for (int minus = -(board_length - condition_of_success); minus <= board_length - condition_of_success; ++minus) {
		cur_line.clear();
		for (int j = max(0, -minus); j <= min(board_length - 1, board_length - 1 - minus); ++j) {
			cur_line.push_back(cur_board[minus + j][j]);
		}
		AxisScore(cur_line, CellState::MY, MyType);
		AxisScore(cur_line, CellState::OPPONENT, OpType);
	}

	int all_score = 0;

	// Special scoring conditions based on player and opponent scores
	if (MyType[0])
		return MyState::score[0];
	else if (OpType[0])
		return OpState::score[0];
	else if (OpType[1])
		return OpState::score[1] * OpType[1];
	else if (MyType[1])
		return MyState::score[1] * MyType[1];
	else if ((OpType[2] && OpType[3]) || OpType[3] >= 2)
		return OpState::score[1];
	else if ((MyType[2] && MyType[3]) || MyType[3] >= 2)
		return MyState::score[1];

	// Calculate final score based on types and weights
	for (int type = 0; type < static_cast<int>(MyType.size()); ++type) {
		all_score += MyState::score[type] * MyType[type];
	}
	for (int type = 0; type < static_cast<int>(MyType.size()); ++type) {
		all_score += OpState::score[type] * OpType[type];
	}

	return all_score;
}

// Core algorithm: Alpha-Beta pruning for minimax search
Movement ChessBoard::AlphaBeta(vector<vector<CellState>>& cur_board, int alpha, int beta, CellState player, int depth) {

	// Check for game over or depth limit; return score if reached
	if (GameIsOver(cur_board) || depth <= 0) {
		Movement tmpClass(0, 0, CalculateCurrentScore(cur_board));
		return tmpClass;
	}

	// Get possible next moves based on current board state
	vector<Position> cur_next_cell = NextCell(cur_board, player);

	// If no next moves available, return current board score
	if (cur_next_cell.empty()) {
		Movement tmpClass(0, 0, CalculateCurrentScore(cur_board));
		return tmpClass;
	}

	Movement best_move;

	// Initialize best score depending on player type
	if (player == CellState::MY)
		best_move.score = INT_MIN;
	else
		best_move.score = INT_MAX;

	// Evaluate each move in cur_next_cell using Alpha-Beta pruning
	for (const Position& pos : cur_next_cell) {
		int cur_x = pos.x, cur_y = pos.y;
		cur_board[cur_x][cur_y] = (player == CellState::MY) ? CellState::MY : CellState::OPPONENT;

		// Recursively call AlphaBeta on the new board state with updated depth
		Movement cur_move = AlphaBeta(cur_board, alpha, beta, !player, depth - 1);

		cur_board[cur_x][cur_y] = CellState::BLANK; // Undo move after evaluation
		cur_move.x = cur_x;
		cur_move.y = cur_y;

		// Update best move and Alpha-Beta bounds based on current player
		if (player == CellState::MY && cur_move.score > best_move.score) {
			best_move = cur_move;
			alpha = max(alpha, best_move.score);
		}
		if (player == CellState::OPPONENT && cur_move.score < best_move.score) {
			best_move = cur_move;
			beta = min(beta, best_move.score);
		}

		// Prune branches where beta <= alpha
		if (beta <= alpha)
			break;
	}
	return best_move;
}

// Exception class for handling invalid inputs
class TriggerWrong : public exception {
private:
	string message;
public:
	TriggerWrong(const string& s) :message(s) {}
	virtual const char* what() const noexcept {
		return message.c_str();
	}
};

int main() {

	ChessBoard Board;
	string Instruct;
	bool flag = true;

	try {
		while (flag) {
			cin >> Instruct;

			// Start command initializes board with first move as specified
			if (Instruct == "START") {
				int move_first;
				cin >> move_first;
				if (move_first == 1)
					Board.Init(CellState::MY);
				else if (move_first == 2)
					Board.Init(CellState::OPPONENT);
				else
					throw TriggerWrong("Wrong in 'START'!");

				Board.InitializeZobristTable();
				cout << "OK" << endl;
			}

			// Place command allows opponent to make a move on board
			else if (Instruct == "PLACE") {
				int x, y;
				cin >> x >> y;
				Board.OpMove(Position(x, y));
			}

			// Turn command computes and outputs AI's best move
			else if (Instruct == "TURN") {
				Position my_movement = Board.BestMovement();
				Board.MyMove(my_movement);
				cout << my_movement.x << " " << my_movement.y << endl;
			}

			// End command stops the game loop
			else if (Instruct == "END") {
				int who_win;
				cin >> who_win;
				flag = false;
			}

			// Handle invalid instruction inputs
			else {
				throw TriggerWrong("Wrong in Instruction input!");
			}
		}
	}
	catch (const TriggerWrong& tw) {
		cerr << tw.what() << endl;
	}
	catch (const std::out_of_range& oor) {
		cerr << oor.what() << endl;
	}

	return 0;
}

/*
影响程序性能的因素：
1. 棋型分数评估 
2. 递归深度
3. 邻居数量
*/

/*
程序相对于dfs的优化：
1. AlphaBeta剪枝
2. 采用局部评估函数让剪枝更高效、减少遍历位置的数量
3. Zobrist哈希表减少对重复棋形的计算
*/