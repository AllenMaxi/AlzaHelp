import React, { useState, useCallback, useRef } from 'react';
import { Grid3X3, RotateCcw, Trophy, Lightbulb, Undo2, Volume2, VolumeX, Sparkles, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

// --- Sudoku Generator ---

function shuffleArray(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function generateFullBoard(size) {
  const board = Array.from({ length: size }, () => Array(size).fill(0));
  const boxH = size === 4 ? 2 : size === 6 ? 2 : 3;
  const boxW = size === 4 ? 2 : size === 6 ? 3 : 3;

  function isValid(board, row, col, num) {
    for (let c = 0; c < size; c++) if (board[row][c] === num) return false;
    for (let r = 0; r < size; r++) if (board[r][col] === num) return false;
    const br = Math.floor(row / boxH) * boxH;
    const bc = Math.floor(col / boxW) * boxW;
    for (let r = br; r < br + boxH; r++)
      for (let c = bc; c < bc + boxW; c++)
        if (board[r][c] === num) return false;
    return true;
  }

  function solve(board) {
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        if (board[r][c] === 0) {
          const nums = shuffleArray([...Array(size)].map((_, i) => i + 1));
          for (const n of nums) {
            if (isValid(board, r, c, n)) {
              board[r][c] = n;
              if (solve(board)) return true;
              board[r][c] = 0;
            }
          }
          return false;
        }
      }
    }
    return true;
  }

  solve(board);
  return board;
}

function createPuzzle(size, clueRatio) {
  const solution = generateFullBoard(size);
  const puzzle = solution.map(row => [...row]);
  const totalCells = size * size;
  const cluesToKeep = Math.floor(totalCells * clueRatio);
  const cellsToRemove = totalCells - cluesToKeep;

  const positions = shuffleArray(
    Array.from({ length: totalCells }, (_, i) => [Math.floor(i / size), i % size])
  );

  for (let i = 0; i < cellsToRemove && i < positions.length; i++) {
    const [r, c] = positions[i];
    puzzle[r][c] = 0;
  }

  return { puzzle, solution };
}

const DIFFICULTIES = {
  easy: { size: 4, clueRatio: 0.6, label: 'Easy (4x4)', boxH: 2, boxW: 2 },
  medium: { size: 6, clueRatio: 0.45, label: 'Medium (6x6)', boxH: 2, boxW: 3 },
  hard: { size: 9, clueRatio: 0.33, label: 'Hard (9x9)', boxH: 3, boxW: 3 },
};

const encouragements = [
  "Well done!",
  "Great thinking!",
  "You're getting better!",
  "Excellent!",
  "Keep it up!",
];

export const SudokuGame = () => {
  const [difficulty, setDifficulty] = useState('easy');
  const [gameStarted, setGameStarted] = useState(false);
  const [puzzle, setPuzzle] = useState([]);
  const [solution, setSolution] = useState([]);
  const [board, setBoard] = useState([]);
  const [initial, setInitial] = useState([]);
  const [selected, setSelected] = useState(null);
  const [errors, setErrors] = useState([]);
  const [isComplete, setIsComplete] = useState(false);
  const [hintsUsed, setHintsUsed] = useState(0);
  const [speakEnabled, setSpeakEnabled] = useState(true);
  const [history, setHistory] = useState([]);
  const [gamesWon, setGamesWon] = useState(0);

  const speak = useCallback((text) => {
    if (speakEnabled && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.8;
      window.speechSynthesis.speak(utterance);
    }
  }, [speakEnabled]);

  const startGame = () => {
    const config = DIFFICULTIES[difficulty];
    const { puzzle: p, solution: s } = createPuzzle(config.size, config.clueRatio);
    setPuzzle(p);
    setSolution(s);
    setBoard(p.map(row => [...row]));
    setInitial(p.map(row => row.map(v => v !== 0)));
    setSelected(null);
    setErrors([]);
    setIsComplete(false);
    setHintsUsed(0);
    setHistory([]);
    setGameStarted(true);
    speak("Fill in the empty cells. Each number can only appear once in each row, column, and box.");
  };

  const resetGame = () => {
    setGameStarted(false);
    setIsComplete(false);
    setSelected(null);
    setErrors([]);
  };

  const handleCellClick = (row, col) => {
    if (initial[row]?.[col]) return;
    setSelected({ row, col });
  };

  const handleNumberInput = (num) => {
    if (!selected || initial[selected.row]?.[selected.col]) return;

    const { row, col } = selected;
    const newBoard = board.map(r => [...r]);

    // Save history for undo
    setHistory(prev => [...prev, { row, col, oldVal: newBoard[row][col] }]);

    newBoard[row][col] = num;
    setBoard(newBoard);

    // Validate
    const config = DIFFICULTIES[difficulty];
    const newErrors = validateBoard(newBoard, config.size, config.boxH, config.boxW);
    setErrors(newErrors);

    // Check completion
    const isFull = newBoard.every(r => r.every(v => v !== 0));
    if (isFull && newErrors.length === 0) {
      setIsComplete(true);
      setGamesWon(prev => prev + 1);
      speak("Congratulations! You solved the puzzle! Your brain is getting stronger!");
    }
  };

  const handleClear = () => {
    if (!selected || initial[selected.row]?.[selected.col]) return;
    const { row, col } = selected;
    setHistory(prev => [...prev, { row, col, oldVal: board[row][col] }]);
    const newBoard = board.map(r => [...r]);
    newBoard[row][col] = 0;
    setBoard(newBoard);
    const config = DIFFICULTIES[difficulty];
    setErrors(validateBoard(newBoard, config.size, config.boxH, config.boxW));
  };

  const handleUndo = () => {
    if (history.length === 0) return;
    const last = history[history.length - 1];
    const newBoard = board.map(r => [...r]);
    newBoard[last.row][last.col] = last.oldVal;
    setBoard(newBoard);
    setHistory(prev => prev.slice(0, -1));
    const config = DIFFICULTIES[difficulty];
    setErrors(validateBoard(newBoard, config.size, config.boxH, config.boxW));
  };

  const handleHint = () => {
    if (!selected) {
      speak("Select an empty cell first, then ask for a hint.");
      return;
    }
    const { row, col } = selected;
    if (initial[row][col]) return;

    const answer = solution[row][col];
    const newBoard = board.map(r => [...r]);
    newBoard[row][col] = answer;
    setBoard(newBoard);
    setHintsUsed(prev => prev + 1);

    const msg = encouragements[Math.floor(Math.random() * encouragements.length)];
    speak(`The answer is ${answer}. ${msg}`);

    const config = DIFFICULTIES[difficulty];
    const newErrors = validateBoard(newBoard, config.size, config.boxH, config.boxW);
    setErrors(newErrors);

    const isFull = newBoard.every(r => r.every(v => v !== 0));
    if (isFull && newErrors.length === 0) {
      setIsComplete(true);
      setGamesWon(prev => prev + 1);
      speak("Puzzle complete! Great job!");
    }
  };

  const config = DIFFICULTIES[difficulty];

  return (
    <section className="py-8 sm:py-12">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-2xl">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-4 py-2 mb-4">
            <Grid3X3 className="h-5 w-5 text-primary" />
            <span className="text-base font-medium text-primary">Brain Exercise</span>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl font-bold text-foreground mb-4">
            Sudoku
          </h2>
          <p className="text-accessible text-muted-foreground">
            Exercise your brain with number puzzles
          </p>
        </div>

        {!gameStarted ? (
          <Card className="border-2 border-border shadow-card">
            <CardContent className="p-8 text-center">
              <div className="w-24 h-24 mx-auto mb-6 rounded-full bg-primary/10 flex items-center justify-center">
                <Grid3X3 className="h-12 w-12 text-primary" />
              </div>
              <h3 className="text-2xl font-bold mb-4">Ready to Exercise Your Brain?</h3>
              <p className="text-lg text-muted-foreground mb-6">
                Fill each row, column, and box with unique numbers. Start easy and work your way up!
              </p>

              {/* Difficulty */}
              <div className="mb-6">
                <p className="text-base font-semibold mb-3">Choose difficulty:</p>
                <div className="flex justify-center gap-3 flex-wrap">
                  {Object.entries(DIFFICULTIES).map(([key, val]) => (
                    <Button
                      key={key}
                      variant={difficulty === key ? 'accessible' : 'accessible-outline'}
                      onClick={() => setDifficulty(key)}
                    >
                      {val.label}
                    </Button>
                  ))}
                </div>
              </div>

              {/* Sound Toggle */}
              <div className="flex justify-center items-center gap-3 mb-6">
                <span className="text-base">Voice hints:</span>
                <Button variant="outline" size="icon" onClick={() => setSpeakEnabled(!speakEnabled)} className="h-12 w-12">
                  {speakEnabled ? <Volume2 className="h-6 w-6" /> : <VolumeX className="h-6 w-6" />}
                </Button>
              </div>

              {gamesWon > 0 && (
                <div className="mb-6">
                  <Badge className="bg-accent/20 text-accent-foreground text-base px-4 py-2">
                    Puzzles Solved: {gamesWon}
                  </Badge>
                </div>
              )}

              <Button variant="accessible" size="xl" onClick={startGame} className="gap-3">
                <Sparkles className="h-6 w-6" />
                Start Puzzle
              </Button>
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-6">
            {/* Top Bar */}
            <Card className="border-2 border-border">
              <CardContent className="p-4">
                <div className="flex justify-between items-center flex-wrap gap-2">
                  <div className="flex items-center gap-3">
                    <Badge className="bg-primary text-primary-foreground text-lg px-4 py-1">
                      {config.label}
                    </Badge>
                    {hintsUsed > 0 && (
                      <Badge className="bg-accent text-accent-foreground text-lg px-4 py-1">
                        Hints: {hintsUsed}
                      </Badge>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <Button variant="outline" size="icon" onClick={() => setSpeakEnabled(!speakEnabled)} className="h-10 w-10">
                      {speakEnabled ? <Volume2 className="h-5 w-5" /> : <VolumeX className="h-5 w-5" />}
                    </Button>
                    <Button variant="outline" onClick={resetGame} className="gap-2">
                      <RotateCcw className="h-5 w-5" />
                      New
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Sudoku Grid */}
            <Card className="border-2 border-border shadow-card">
              <CardContent className="p-4 sm:p-6">
                <div className="mx-auto" style={{ maxWidth: config.size === 9 ? '420px' : config.size === 6 ? '360px' : '280px' }}>
                  <div
                    className="grid border-2 border-foreground rounded-lg overflow-hidden"
                    style={{ gridTemplateColumns: `repeat(${config.size}, 1fr)` }}
                  >
                    {board.map((row, r) =>
                      row.map((val, c) => {
                        const isSelected = selected?.row === r && selected?.col === c;
                        const isInitialCell = initial[r]?.[c];
                        const hasError = errors.some(e => e[0] === r && e[1] === c);
                        const isSameValue = selected && val !== 0 && board[selected.row]?.[selected.col] === val;

                        // Box borders
                        const borderRight = (c + 1) % config.boxW === 0 && c < config.size - 1 ? 'border-r-2 border-r-foreground' : 'border-r border-r-border';
                        const borderBottom = (r + 1) % config.boxH === 0 && r < config.size - 1 ? 'border-b-2 border-b-foreground' : 'border-b border-b-border';

                        return (
                          <button
                            key={`${r}-${c}`}
                            onClick={() => handleCellClick(r, c)}
                            className={`
                              aspect-square flex items-center justify-center
                              transition-colors duration-150
                              ${borderRight} ${borderBottom}
                              ${isSelected ? 'bg-primary/20 ring-2 ring-inset ring-primary' : ''}
                              ${!isSelected && isSameValue ? 'bg-primary/10' : ''}
                              ${hasError ? 'bg-destructive/20 text-destructive' : ''}
                              ${isInitialCell ? 'font-bold text-foreground' : 'text-primary font-semibold'}
                              ${!isInitialCell && !isSelected ? 'hover:bg-muted cursor-pointer' : ''}
                              ${isInitialCell ? 'cursor-default' : ''}
                              text-lg sm:text-2xl
                            `}
                          >
                            {val !== 0 ? val : ''}
                          </button>
                        );
                      })
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Number Input Pad */}
            {!isComplete && (
              <Card className="border-2 border-border">
                <CardContent className="p-4">
                  <div className="flex flex-wrap justify-center gap-2 mb-3">
                    {Array.from({ length: config.size }, (_, i) => i + 1).map(num => (
                      <Button
                        key={num}
                        variant="outline"
                        className="w-12 h-12 sm:w-14 sm:h-14 text-xl font-bold hover:bg-primary hover:text-primary-foreground"
                        onClick={() => handleNumberInput(num)}
                        disabled={!selected}
                      >
                        {num}
                      </Button>
                    ))}
                  </div>
                  <div className="flex justify-center gap-2">
                    <Button variant="outline" onClick={handleClear} disabled={!selected} className="gap-2">
                      Clear
                    </Button>
                    <Button variant="outline" onClick={handleUndo} disabled={history.length === 0} className="gap-2">
                      <Undo2 className="h-4 w-4" />
                      Undo
                    </Button>
                    <Button variant="outline" onClick={handleHint} disabled={!selected} className="gap-2 text-accent-foreground">
                      <Lightbulb className="h-4 w-4" />
                      Hint
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Win Screen */}
            {isComplete && (
              <Card className="border-2 border-success bg-success/5 animate-scale-in">
                <CardContent className="p-8 text-center">
                  <div className="flex items-center justify-center gap-3 mb-4">
                    <Trophy className="h-10 w-10 text-success" />
                    <Check className="h-8 w-8 text-success" />
                    <Trophy className="h-10 w-10 text-success" />
                  </div>
                  <h3 className="text-2xl font-bold mb-2">Puzzle Solved!</h3>
                  <p className="text-lg text-muted-foreground mb-2">
                    Great work exercising your brain!
                  </p>
                  {hintsUsed > 0 && (
                    <p className="text-base text-muted-foreground mb-4">
                      Hints used: {hintsUsed}
                    </p>
                  )}
                  <Button variant="accessible" size="lg" onClick={startGame} className="gap-2">
                    <RotateCcw className="h-5 w-5" />
                    Play Again
                  </Button>
                </CardContent>
              </Card>
            )}
          </div>
        )}
      </div>
    </section>
  );
};

// --- Validation helper ---
function validateBoard(board, size, boxH, boxW) {
  const errorCells = [];

  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const val = board[r][c];
      if (val === 0) continue;

      // Check row
      for (let cc = 0; cc < size; cc++) {
        if (cc !== c && board[r][cc] === val) {
          errorCells.push([r, c]);
          break;
        }
      }

      // Check column
      let colErr = false;
      for (let rr = 0; rr < size; rr++) {
        if (rr !== r && board[rr][c] === val) {
          if (!errorCells.some(e => e[0] === r && e[1] === c)) errorCells.push([r, c]);
          colErr = true;
          break;
        }
      }

      // Check box
      if (!colErr) {
        const br = Math.floor(r / boxH) * boxH;
        const bc = Math.floor(c / boxW) * boxW;
        for (let rr = br; rr < br + boxH; rr++) {
          for (let cc = bc; cc < bc + boxW; cc++) {
            if ((rr !== r || cc !== c) && board[rr][cc] === val) {
              if (!errorCells.some(e => e[0] === r && e[1] === c)) errorCells.push([r, c]);
            }
          }
        }
      }
    }
  }

  return errorCells;
}

export default SudokuGame;
