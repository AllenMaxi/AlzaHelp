import React, { useState, useEffect, useCallback } from 'react';
import { Layers, RotateCcw, Trophy, Heart, Volume2, VolumeX, Star, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

const DIFFICULTIES = {
  easy: { pairs: 3, cols: 3, label: 'Easy (6 cards)' },
  medium: { pairs: 5, cols: 4, label: 'Medium (10 cards)' },
  hard: { pairs: 8, cols: 4, label: 'Hard (16 cards)' },
};

const placeholderPhotos = [
  'https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=200&h=200&fit=crop&crop=face',
  'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=200&h=200&fit=crop&crop=face',
  'https://images.unsplash.com/photo-1489424731084-a5d8b219a5bb?w=200&h=200&fit=crop&crop=face',
  'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=200&h=200&fit=crop&crop=face',
  'https://images.unsplash.com/photo-1547425260-76bcadfb4f2c?w=200&h=200&fit=crop&crop=face',
  'https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=200&h=200&fit=crop&crop=face',
  'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=200&h=200&fit=crop&crop=face',
  'https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=200&h=200&fit=crop&crop=face',
];

const encouragements = [
  "Great match! You remembered!",
  "Wonderful! Keep going!",
  "Excellent memory!",
  "That's right! Well done!",
  "Amazing job!",
];

const getPhoto = (member, index) => {
  if (member.photos && member.photos.length > 0) return member.photos[0];
  return placeholderPhotos[index % placeholderPhotos.length];
};

export const MemoryCardGame = ({ familyMembers = [] }) => {
  const [difficulty, setDifficulty] = useState('easy');
  const [gameStarted, setGameStarted] = useState(false);
  const [cards, setCards] = useState([]);
  const [flipped, setFlipped] = useState([]);
  const [matched, setMatched] = useState([]);
  const [moves, setMoves] = useState(0);
  const [bestMoves, setBestMoves] = useState(null);
  const [speakEnabled, setSpeakEnabled] = useState(true);
  const [showCelebration, setShowCelebration] = useState(false);
  const [lockBoard, setLockBoard] = useState(false);

  const speak = useCallback((text) => {
    if (speakEnabled && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.8;
      utterance.pitch = 1;
      window.speechSynthesis.speak(utterance);
    }
  }, [speakEnabled]);

  const buildDeck = useCallback(() => {
    const { pairs } = DIFFICULTIES[difficulty];
    // Use family members if available, fill with placeholders
    const pool = familyMembers.length > 0
      ? familyMembers.map((m, i) => ({ id: m.id || `m${i}`, name: m.name, photo: getPhoto(m, i), relationship: m.relationship_label || m.relationship }))
      : placeholderPhotos.slice(0, pairs).map((p, i) => ({ id: `p${i}`, name: `Person ${i + 1}`, photo: p, relationship: '' }));

    const selected = pool.slice(0, pairs);
    // If not enough, repeat from the beginning
    while (selected.length < pairs) {
      selected.push(pool[selected.length % pool.length]);
    }

    // Create pairs
    const deck = [];
    selected.forEach((person, idx) => {
      deck.push({ ...person, cardId: `${person.id}-a-${idx}`, pairKey: person.id + idx });
      deck.push({ ...person, cardId: `${person.id}-b-${idx}`, pairKey: person.id + idx });
    });

    // Shuffle
    for (let i = deck.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [deck[i], deck[j]] = [deck[j], deck[i]];
    }

    return deck;
  }, [difficulty, familyMembers]);

  const startGame = () => {
    const deck = buildDeck();
    setCards(deck);
    setFlipped([]);
    setMatched([]);
    setMoves(0);
    setShowCelebration(false);
    setLockBoard(false);
    setGameStarted(true);
    speak("Find the matching pairs! Tap a card to flip it.");
  };

  const resetGame = () => {
    setGameStarted(false);
    setCards([]);
    setFlipped([]);
    setMatched([]);
    setMoves(0);
    setShowCelebration(false);
    setLockBoard(false);
  };

  const handleCardClick = (index) => {
    if (lockBoard) return;
    if (flipped.includes(index)) return;
    if (matched.includes(cards[index].pairKey)) return;

    const newFlipped = [...flipped, index];
    setFlipped(newFlipped);

    if (newFlipped.length === 2) {
      setMoves(prev => prev + 1);
      setLockBoard(true);

      const [first, second] = newFlipped;
      if (cards[first].pairKey === cards[second].pairKey) {
        // Match found
        const newMatched = [...matched, cards[first].pairKey];
        setMatched(newMatched);

        const msg = encouragements[Math.floor(Math.random() * encouragements.length)];
        speak(`${msg} That's ${cards[first].name}!`);

        setTimeout(() => {
          setFlipped([]);
          setLockBoard(false);

          // Check win
          if (newMatched.length === cards.length / 2) {
            const totalMoves = moves + 1;
            setShowCelebration(true);
            if (bestMoves === null || totalMoves < bestMoves) {
              setBestMoves(totalMoves);
            }
            speak(`Congratulations! You found all the matches in ${totalMoves} moves!`);
          }
        }, 800);
      } else {
        // No match — flip back
        speak("Not a match. Try again!");
        setTimeout(() => {
          setFlipped([]);
          setLockBoard(false);
        }, 1200);
      }
    }
  };

  // Check if game is won
  const isWon = showCelebration;
  const { cols } = DIFFICULTIES[difficulty];
  const totalCards = DIFFICULTIES[difficulty].pairs * 2;

  if (familyMembers.length < 1 && !gameStarted) {
    return (
      <section className="py-8 sm:py-12">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-2xl">
          <Card className="border-2 border-border shadow-card">
            <CardContent className="p-8 text-center">
              <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-muted flex items-center justify-center">
                <Layers className="h-10 w-10 text-muted-foreground" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Add Family Members First</h3>
              <p className="text-muted-foreground">
                Add at least 1 family member to play the Memory Card Game with their photos.
                You can also play with placeholder images!
              </p>
              <Button variant="accessible" size="lg" className="mt-4" onClick={startGame}>
                Play with Placeholders
              </Button>
            </CardContent>
          </Card>
        </div>
      </section>
    );
  }

  return (
    <section className="py-8 sm:py-12">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-3xl">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-4 py-2 mb-4">
            <Layers className="h-5 w-5 text-primary" />
            <span className="text-base font-medium text-primary">Memory Exercise</span>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl font-bold text-foreground mb-4">
            Memory Match
          </h2>
          <p className="text-accessible text-muted-foreground">
            Find matching pairs of your loved ones
          </p>
        </div>

        {!gameStarted ? (
          <Card className="border-2 border-border shadow-card">
            <CardContent className="p-8 text-center">
              <div className="w-24 h-24 mx-auto mb-6 rounded-full bg-primary/10 flex items-center justify-center">
                <Layers className="h-12 w-12 text-primary" />
              </div>
              <h3 className="text-2xl font-bold mb-4">Ready to Play?</h3>
              <p className="text-lg text-muted-foreground mb-6">
                Flip cards and find matching pairs. This helps strengthen your visual memory!
              </p>

              {/* Difficulty Selection */}
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
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setSpeakEnabled(!speakEnabled)}
                  className="h-12 w-12"
                >
                  {speakEnabled ? <Volume2 className="h-6 w-6" /> : <VolumeX className="h-6 w-6" />}
                </Button>
              </div>

              {bestMoves !== null && (
                <div className="mb-6">
                  <Badge className="bg-accent/20 text-accent-foreground text-base px-4 py-2">
                    Best: {bestMoves} moves
                  </Badge>
                </div>
              )}

              <Button variant="accessible" size="xl" onClick={startGame} className="gap-3">
                <Sparkles className="h-6 w-6" />
                Start Game
              </Button>
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-6">
            {/* Score Bar */}
            <Card className="border-2 border-border">
              <CardContent className="p-4">
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-4">
                    <Badge className="bg-primary text-primary-foreground text-lg px-4 py-1">
                      Moves: {moves}
                    </Badge>
                    <Badge className="bg-accent text-accent-foreground text-lg px-4 py-1">
                      Pairs: {matched.length}/{DIFFICULTIES[difficulty].pairs}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button variant="outline" size="icon" onClick={() => setSpeakEnabled(!speakEnabled)} className="h-10 w-10">
                      {speakEnabled ? <Volume2 className="h-5 w-5" /> : <VolumeX className="h-5 w-5" />}
                    </Button>
                    <Button variant="outline" onClick={resetGame} className="gap-2">
                      <RotateCcw className="h-5 w-5" />
                      Reset
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Card Grid */}
            <div
              className="grid gap-3 sm:gap-4"
              style={{ gridTemplateColumns: `repeat(${Math.min(cols, Math.ceil(Math.sqrt(totalCards)))}, minmax(0, 1fr))` }}
            >
              {cards.map((card, index) => {
                const isFlipped = flipped.includes(index);
                const isMatched = matched.includes(card.pairKey);

                return (
                  <button
                    key={card.cardId}
                    onClick={() => handleCardClick(index)}
                    disabled={isMatched}
                    className={`
                      relative aspect-square rounded-2xl overflow-hidden
                      transition-all duration-500 transform
                      ${isFlipped || isMatched ? 'scale-100' : 'hover:scale-105'}
                      ${isMatched ? 'ring-4 ring-success opacity-80' : ''}
                      ${!isFlipped && !isMatched ? 'cursor-pointer' : ''}
                    `}
                    style={{ perspective: '1000px' }}
                  >
                    <div
                      className={`
                        w-full h-full transition-transform duration-500
                        ${isFlipped || isMatched ? '' : ''}
                      `}
                      style={{
                        transformStyle: 'preserve-3d',
                        transform: isFlipped || isMatched ? 'rotateY(180deg)' : 'rotateY(0deg)',
                      }}
                    >
                      {/* Card Back */}
                      <div
                        className="absolute inset-0 rounded-2xl bg-gradient-to-br from-primary to-primary/70 flex items-center justify-center shadow-card border-2 border-primary/30"
                        style={{ backfaceVisibility: 'hidden' }}
                      >
                        <Heart className="h-10 w-10 sm:h-12 sm:w-12 text-primary-foreground" />
                      </div>

                      {/* Card Front */}
                      <div
                        className="absolute inset-0 rounded-2xl overflow-hidden shadow-card border-2 border-border"
                        style={{ backfaceVisibility: 'hidden', transform: 'rotateY(180deg)' }}
                      >
                        <img
                          src={card.photo}
                          alt={card.name}
                          className="w-full h-full object-cover"
                        />
                        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-2">
                          <p className="text-white text-sm sm:text-base font-bold text-center truncate">
                            {card.name}
                          </p>
                        </div>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Celebration */}
            {isWon && (
              <Card className="border-2 border-success bg-success/5 animate-scale-in">
                <CardContent className="p-8 text-center">
                  <div className="flex items-center justify-center gap-3 mb-4">
                    <Trophy className="h-10 w-10 text-success" />
                    <Star className="h-8 w-8 text-accent" />
                    <Trophy className="h-10 w-10 text-success" />
                  </div>
                  <h3 className="text-2xl font-bold mb-2">You Did It!</h3>
                  <p className="text-lg text-muted-foreground mb-4">
                    All pairs found in <strong>{moves}</strong> moves!
                    {bestMoves && moves <= bestMoves && " That's your best score!"}
                  </p>
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

export default MemoryCardGame;
