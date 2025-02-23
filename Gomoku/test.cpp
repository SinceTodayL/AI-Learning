#include<iostream>
#include<string>
#include<vector>
#include<set>
using namespace std;

constexpr int board_length = 12;
constexpr int middle_position = board_length / 2 - 1;
constexpr int condition_of_success = 5;
constexpr int directions[4][2] = { {1, 0}, {0, 1}, {1, 1}, {1, -1} };
constexpr int recursion_depth = 1;

enum class CellState {
	BLANK,
	MY,
	OPPONENT    // opponent chess
};
CellState operator!(CellState turn) {
	if (turn != CellState::BLANK)
		return (turn == CellState::MY) ? CellState::OPPONENT : CellState::MY;
	else
		return CellState::BLANK;
}
enum class CellState {
	MY,
	OPPONENT
};
CellState operator!(CellState turn) {
	return (turn == CellState::MY) ? CellState::OPPONENT : CellState::MY;
}


class Position {
public:
	int x, y;
	Position() {
		this->x = 0;
		this->y = 0;
	}
	Position(int posx, int posy) : x(posx), y(posy) {}

	bool isValid() const {
		return this->x >= 0 && this->x < board_length && this->y >= 0 && this->y <= board_length;
	}
	bool operator==(const Position& pos) const {
		return this->x == pos.x && this->y == pos.y;
	}
};
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


class PositionCompare {
public:
	bool operator()(Position p1, Position p2) const {
		return (abs(p1.x - middle_position) + abs(p1.y - middle_position))
			< (abs(p2.x - middle_position) + abs(p2.y - middle_position));
	}
};


class ChessBoard {
private:
	vector<vector<CellState>> board;

	multiset<Position, PositionCompare> blankcell;
public:
	ChessBoard() : board(board_length, vector<CellState>(board_length, CellState::BLANK)) {
		for (int i = 0; i < board_length; ++i)
			for (int j = 0; j < board_length; ++j)
				this->blankcell.insert(Position(i, j));
	}

	void Init(const CellState& wt) {    // black is the first turn 
		board[middle_position][middle_position] = (wt == CellState::MY) ? CellState::OPPONENT : CellState::MY;
		board[middle_position + 1][middle_position + 1] = (wt == CellState::MY) ? CellState::OPPONENT : CellState::MY;
		board[middle_position][middle_position + 1] = (wt != CellState::MY) ? CellState::OPPONENT : CellState::MY;
		board[middle_position + 1][middle_position] = (wt != CellState::MY) ? CellState::OPPONENT : CellState::MY;

		for (auto it = blankcell.begin(); it != blankcell.end(); /* no increment here */) {
			if (*it == Position(middle_position, middle_position) ||
				*it == Position(middle_position + 1, middle_position + 1) ||
				*it == Position(middle_position, middle_position + 1) ||
				*it == Position(middle_position + 1, middle_position)) {
				it = blankcell.erase(it);
			}
			else {
				++it;
			}
		}
	}

	void OpMove(const Position& pos) {
		if (pos.isValid()) {
			this->board[pos.x][pos.y] = CellState::OPPONENT;
		}
		for (auto it = blankcell.begin(); it != blankcell.end(); ) {
			if (*it == pos)
				it = blankcell.erase(it);
			else
				++it;
		}
	}
	void MyMove(const Position& pos) {
		if (pos.isValid()) {
			this->board[pos.x][pos.y] = CellState::MY;
		}
		for (auto it = blankcell.begin(); it != blankcell.end(); ) {
			if (*it == pos)
				it = blankcell.erase(it);
			else
				++it;
		}
	}

	int CalculateCurrentScore(const vector<vector<CellState>>& cur_board);

	bool CheckDirection(const vector<vector<CellState>>& cur_board,
		int x, int y, int dx, int dy, CellState player);
	bool GameIsOver(const vector<vector<CellState>>& cur_board);

	Position BestMovement() {
		Movement best_move = AlphaBeta(this->board, this->blankcell, INT_MIN, INT_MAX, CellState::MY, recursion_depth);
		Position best_position(best_move.x, best_move.y);

		return best_position;
	}

	Movement AlphaBeta(vector<vector<CellState>>& cur_board, multiset<Position, PositionCompare>& cur_blank_cell,
		int alpha, int beta, CellState wt, int depth) {

		//cout << "AlphaBeta: " << (*cur_blank_cell.begin()).x << " " << (*cur_blank_cell.begin()).y << " depth: " << depth << endl;

		if (cur_blank_cell.empty() || GameIsOver(cur_board) || depth <= 0) {
			Movement tmpClass(0, 0, CalculateCurrentScore(cur_board));
			return tmpClass;
		}

		Movement best_move;

		if (wt == CellState::MY)
			best_move.score = INT_MIN;
		else
			best_move.score = INT_MAX;

		for (const Position& pos : cur_blank_cell) {
			int cur_x = pos.x, cur_y = pos.y;
			cur_board[cur_x][cur_y] = (wt == CellState::MY) ? CellState::MY : CellState::OPPONENT;

			multiset<Position, PositionCompare> next_blank_cell = cur_blank_cell;
			for (auto it = next_blank_cell.begin(); it != next_blank_cell.end(); ) {
				if (*it == pos)
					it = next_blank_cell.erase(it);
				else
					++it;
			}
			Movement cur_move = AlphaBeta(cur_board, next_blank_cell, alpha, beta, !wt, depth - 1);

			cur_board[cur_x][cur_y] = CellState::BLANK;
			cur_move.x = cur_x;
			cur_move.y = cur_y;

			if (wt == CellState::MY && cur_move.score > best_move.score) {
				best_move = cur_move;
				alpha = max(alpha, best_move.score);
			}
			if (wt == CellState::OPPONENT && cur_move.score < best_move.score) {
				best_move = cur_move;
				beta = min(beta, best_move.score);
			}

			if (beta <= alpha)
				break;
		}
		return best_move;
	}
};

bool ChessBoard::CheckDirection(const vector<vector<CellState>>& cur_board,
	int x, int y, int dx, int dy, CellState player) {

	for (int i = 0; i < condition_of_success; ++i) {
		int nx = x + i * dx;
		int ny = y + i * dy;

		if (nx < 0 || ny < 0 || nx >= board_length || ny >= board_length
			|| cur_board[nx][ny] != player) {
			return false;
		}
	}
	return true;
}


bool ChessBoard::GameIsOver(const vector<vector<CellState>>& cur_board) {

	for (int i = 0; i < board_length; ++i) {
		for (int j = 0; j < board_length; ++j) {
			for (auto dir : directions)
				if (CheckDirection(cur_board, i, j, dir[0], dir[1], CellState::MY) ||
					CheckDirection(cur_board, i, j, dir[0], dir[1], CellState::OPPONENT))
					return true;
		}
	}
	return false;
}

namespace MyState {

	constexpr int Five = 1000000;
	constexpr int ActiveFour = 250000;
	constexpr int RushFour = 70000;
	constexpr int ActiveThree = 20000;
	constexpr int SleepThree = 10000;
	constexpr int ActiveTwo = 5000;
	constexpr int RushTwo = 100;


	vector<vector<CellState>> FiveState = { {CellState::MY,
		CellState::MY, CellState::MY, CellState::MY, CellState::MY } };


	vector<vector<CellState>> ActiveFourState = { { CellState::BLANK, CellState::MY,
		CellState::MY, CellState::MY, CellState::MY, CellState::BLANK } };


	vector<vector<CellState>> RushFourState = { { CellState::OPPONENT, CellState::MY,
		CellState::MY, CellState::MY, CellState::MY, CellState::BLANK },

	{ CellState::BLANK, CellState::MY,
		CellState::MY, CellState::MY, CellState::MY, CellState::OPPONENT },

	{ CellState::MY, CellState::BLANK,
		CellState::MY, CellState::MY, CellState::MY },

	{ CellState::MY, CellState::MY,
		CellState::BLANK, CellState::MY, CellState::MY },

	{ CellState::MY, CellState::MY,
		CellState::MY, CellState::BLANK, CellState::MY },
	};


	vector<vector<CellState>> ActiveThreeState = { { CellState::BLANK, CellState::MY,
		CellState::MY, CellState::MY, CellState::BLANK} ,

	{ CellState::BLANK, CellState::MY,
		CellState::MY, CellState::BLANK, CellState::MY, CellState::BLANK },

	{ CellState::BLANK, CellState::MY,
		CellState::BLANK, CellState::MY, CellState::MY, CellState::BLANK }
	};


	vector<vector<CellState>> SleepThreeState = { { CellState::OPPONENT, CellState::MY,
		CellState::MY, CellState::BLANK, CellState::MY, CellState::BLANK },

	{ CellState::OPPONENT, CellState::MY,
		CellState::BLANK, CellState::MY, CellState::MY, CellState::BLANK },

	{ CellState::BLANK, CellState::MY,
		CellState::MY, CellState::BLANK, CellState::MY, CellState::OPPONENT },

	{ CellState::BLANK, CellState::MY,
		CellState::BLANK, CellState::MY, CellState::MY, CellState::OPPONENT },

	{ CellState::OPPONENT, CellState::MY,
		CellState::MY, CellState::MY, CellState::BLANK, CellState::BLANK },

	{ CellState::BLANK, CellState::BLANK,
		CellState::MY, CellState::MY, CellState::MY, CellState::OPPONENT },
	};


	vector<vector<CellState>> ActiveTwoState = { { CellState::BLANK, CellState::MY,
	 CellState::MY, CellState::BLANK } };


	vector<vector<CellState>> RushTwoState = { { CellState::OPPONENT, CellState::MY,
		CellState::MY, CellState::BLANK },

	{ CellState::BLANK, CellState::MY, CellState::MY, CellState::OPPONENT } };


	vector<vector<CellState>> SpecialMyRushFourState = { { CellState::MY,
		CellState::MY, CellState::MY, CellState::MY, CellState::BLANK },

		  { CellState::BLANK, CellState::MY,
		CellState::MY, CellState::MY, CellState::MY } };


	const vector<vector<vector<CellState>>> mystate = { FiveState, ActiveFourState, RushFourState,
	ActiveThreeState, SleepThreeState, ActiveTwoState, RushTwoState };

	const vector<int> score = { Five, ActiveFour, RushFour, ActiveThree, SleepThree,
		ActiveTwo, RushTwo };
}


namespace OpState {

	vector<vector<CellState>> SpecialOpRushFourState = { { CellState::OPPONENT,
		CellState::OPPONENT, CellState::OPPONENT, CellState::OPPONENT, CellState::BLANK },

		  { CellState::BLANK, CellState::OPPONENT,
		CellState::OPPONENT, CellState::OPPONENT, CellState::OPPONENT } };

	constexpr int Five = -800000;
	constexpr int ActiveFour = -800000;
	constexpr int RushFour = -800000;
	constexpr int ActiveThree = -80000;
	constexpr int SleepThree = -2000;
	constexpr int ActiveTwo = -400;
	constexpr int RushTwo = -50;

	const vector<int> score = { Five, ActiveFour, RushFour, ActiveThree, SleepThree,
		ActiveTwo, RushTwo };
}

void AxisScore(vector<CellState> cur_line, CellState who, vector<int>& Type) {

	if (static_cast<int>(cur_line.size()) <= condition_of_success) {
		if (who == CellState::MY) {
			for (auto cur_state : MyState::SpecialMyRushFourState) {
				if (cur_line == cur_state) {
					Type[2]++;
				}
			}
		}
		else {
			for (auto cur_state : OpState::SpecialOpRushFourState) {
				if (cur_line == cur_state) {
					Type[2]++;
				}
			}
		}
	}

	for (int type = 0; type < static_cast<int>(Type.size()); ++type) {
		for (auto cur_state : MyState::mystate[type]) {
			int fptr = 0, cnt = 0;
			bool flag = false;
			for (fptr = 0; fptr < static_cast<int>(cur_line.size()); ++fptr) {
				if (who == CellState::MY ? (cur_line[fptr] == cur_state[cnt]) : (cur_line[fptr] == !cur_state[cnt])) {
					while (who == CellState::MY ? (cur_line[fptr] == cur_state[cnt]) : (cur_line[fptr] == !cur_state[cnt])) {
						++fptr;
						++cnt;
						if (cnt == static_cast<int>(cur_state.size())) {
							for (int i = fptr - cnt; i < fptr; ++i)
								cur_line[i] = CellState::BLANK;
							Type[type]++;
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
}

int ChessBoard::CalculateCurrentScore(const vector<vector<CellState>>& cur_board) {

	vector<CellState> cur_line;
	vector<int> MyType(MyState::mystate.size(), 0);
	vector<int> OpType(MyState::mystate.size(), 0);

	// x axis
	for (int i = 0; i < board_length; ++i) {
		cur_line = cur_board[i];
		AxisScore(cur_line, CellState::MY, MyType);
		AxisScore(cur_line, CellState::OPPONENT, OpType);
	}

	// y axis
	for (int j = 0; j < board_length; ++j) {
		cur_line.clear();
		for (int i = 0; i < board_length; ++i) {
			cur_line.push_back(cur_board[i][j]);
		}
		AxisScore(cur_line, CellState::MY, MyType);
		AxisScore(cur_line, CellState::OPPONENT, OpType);
	}

	// x + y = constant axis
	for (int all = condition_of_success - 1; all <= 2 * (board_length - 1) - condition_of_success + 1; ++all) {
		cur_line.clear();
		for (int j = max(0, all - (board_length - 1)); j <= min(all, board_length - 1); ++j) {
			cur_line.push_back(cur_board[all - j][j]);
		}
		AxisScore(cur_line, CellState::MY, MyType);
		AxisScore(cur_line, CellState::OPPONENT, OpType);
	}

	// x - y = constant axis
	for (int minus = -(board_length - condition_of_success); minus <= board_length - condition_of_success; ++minus) {
		cur_line.clear();
		for (int j = max(0, -minus); j <= min(board_length - 1, board_length - 1 - minus); ++j) {
			cur_line.push_back(cur_board[minus + j][j]);
		}
		AxisScore(cur_line, CellState::MY, MyType);
		AxisScore(cur_line, CellState::OPPONENT, OpType);
	}

	// static int cnt = 0;
	// cout << "CalculateCurrentScore: " << ++cnt << ": " << my_score + op_score << endl;
	int all_score = 0;
	for (int type = 0; type < static_cast<int>(MyType.size()); ++type) {
		all_score += MyState::score[type] * MyType[type];
	}
	for (int type = 0; type < static_cast<int>(MyType.size()); ++type) {
		all_score += OpState::score[type] * OpType[type];
	}

	if (MyType[2] && MyType[3])
		all_score += (MyState::ActiveThree + MyState::RushFour);

	return all_score;
}

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
			if (Instruct == "START") {
				int move_first;
				cin >> move_first;
				if (move_first == 1)
					Board.Init(CellState::MY);
				else if (move_first == 2)
					Board.Init(CellState::OPPONENT);
				else
					throw TriggerWrong("Wrong in 'START'!");
				cout << "OK" << endl;
			}
			else if (Instruct == "PLACE") {
				int x, y;
				cin >> x >> y;
				Board.OpMove(Position(x, y));
			}
			else if (Instruct == "TURN") {
				Position my_movement = Board.BestMovement();
				Board.MyMove(my_movement);
				cout << my_movement.x << " " << my_movement.y << endl;
			}
			else if (Instruct == "END") {
				int who_win;
				cin >> who_win;
				flag = false;
			}
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