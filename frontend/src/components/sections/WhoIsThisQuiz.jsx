import React, { useState, useEffect, useCallback } from 'react';
import { Brain, Check, X, RotateCcw, Trophy, Heart, Volume2, VolumeX, ChevronRight, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';

// Placeholder images for family members without photos
const placeholderImages = {
  spouse: 'https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=400&h=400&fit=crop&crop=face',
  children: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop&crop=face',
  grandchildren: 'https://images.unsplash.com/photo-1489424731084-a5d8b219a5bb?w=400&h=400&fit=crop&crop=face',
  siblings: 'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=400&h=400&fit=crop&crop=face',
  parents: 'https://images.unsplash.com/photo-1547425260-76bcadfb4f2c?w=400&h=400&fit=crop&crop=face',
  friends: 'https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=400&h=400&fit=crop&crop=face',
  other: 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=400&h=400&fit=crop&crop=face',
};

const encouragements = [
  "Wonderful! You remembered!",
  "That's right! Great job!",
  "Excellent! Your memory is strong!",
  "Perfect! You got it!",
  "Amazing! Well done!",
];

const tryAgainMessages = [
  "That's okay, let's try again!",
  "No worries, memory takes practice!",
  "Don't give up, you're doing great!",
  "Let's look at this one together.",
];

export const WhoIsThisQuiz = ({ familyMembers = [] }) => {
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [options, setOptions] = useState([]);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [isCorrect, setIsCorrect] = useState(null);
  const [score, setScore] = useState(0);
  const [totalQuestions, setTotalQuestions] = useState(0);
  const [streak, setStreak] = useState(0);
  const [bestStreak, setBestStreak] = useState(0);
  const [gameStarted, setGameStarted] = useState(false);
  const [showResult, setShowResult] = useState(false);
  const [speakEnabled, setSpeakEnabled] = useState(true);
  const [difficulty, setDifficulty] = useState('easy'); // easy = 2 options, medium = 3, hard = 4

  const getPhoto = (member) => {
    if (member.photos && member.photos.length > 0) {
      return member.photos[0];
    }
    return placeholderImages[member.category] || placeholderImages.other;
  };

  const speak = useCallback((text) => {
    if (speakEnabled && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.8;
      utterance.pitch = 1;
      window.speechSynthesis.speak(utterance);
    }
  }, [speakEnabled]);

  const generateQuestion = useCallback(() => {
    if (familyMembers.length < 2) return;

    // Pick a random family member as the correct answer
    const correctIndex = Math.floor(Math.random() * familyMembers.length);
    const correct = familyMembers[correctIndex];

    // Determine number of options based on difficulty
    const numOptions = difficulty === 'easy' ? 2 : difficulty === 'medium' ? 3 : 4;
    const maxOptions = Math.min(numOptions, familyMembers.length);

    // Get wrong answers
    const wrongOptions = familyMembers
      .filter((_, i) => i !== correctIndex)
      .sort(() => Math.random() - 0.5)
      .slice(0, maxOptions - 1);

    // Combine and shuffle options
    const allOptions = [correct, ...wrongOptions].sort(() => Math.random() - 0.5);

    setCurrentQuestion(correct);
    setOptions(allOptions);
    setSelectedAnswer(null);
    setIsCorrect(null);
    setShowResult(false);

    // Speak the question
    setTimeout(() => speak("Who is this person?"), 500);
  }, [familyMembers, difficulty, speak]);

  const handleAnswer = (member) => {
    if (selectedAnswer !== null) return;

    setSelectedAnswer(member);
    setTotalQuestions(prev => prev + 1);
    const correct = member.id === currentQuestion.id;
    setIsCorrect(correct);
    setShowResult(true);

    if (correct) {
      setScore(prev => prev + 1);
      setStreak(prev => {
        const newStreak = prev + 1;
        if (newStreak > bestStreak) {
          setBestStreak(newStreak);
        }
        return newStreak;
      });
      const encouragement = encouragements[Math.floor(Math.random() * encouragements.length)];
      speak(`${encouragement} This is ${currentQuestion.name}, ${currentQuestion.relationship_label}.`);
    } else {
      setStreak(0);
      const tryAgain = tryAgainMessages[Math.floor(Math.random() * tryAgainMessages.length)];
      speak(`${tryAgain} This is ${currentQuestion.name}, ${currentQuestion.relationship_label}.`);
    }
  };

  const nextQuestion = () => {
    generateQuestion();
  };

  const startGame = () => {
    setGameStarted(true);
    setScore(0);
    setTotalQuestions(0);
    setStreak(0);
    generateQuestion();
  };

  const resetGame = () => {
    setGameStarted(false);
    setScore(0);
    setTotalQuestions(0);
    setStreak(0);
    setCurrentQuestion(null);
    setSelectedAnswer(null);
    setIsCorrect(null);
    setShowResult(false);
  };

  if (familyMembers.length < 2) {
    return (
      <section className="py-8 sm:py-12">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-2xl">
          <Card className="border-2 border-border shadow-card">
            <CardContent className="p-8 text-center">
              <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-muted flex items-center justify-center">
                <Brain className="h-10 w-10 text-muted-foreground" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Add More Family Members</h3>
              <p className="text-muted-foreground">
                Add at least 2 family members to play the "Who Is This?" quiz game.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>
    );
  }

  return (
    <section className="py-8 sm:py-12">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-2xl">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-4 py-2 mb-4">
            <Brain className="h-5 w-5 text-primary" />
            <span className="text-base font-medium text-primary">Memory Exercise</span>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl font-bold text-foreground mb-4">
            Who Is This?
          </h2>
          <p className="text-accessible text-muted-foreground">
            Practice remembering your loved ones
          </p>
        </div>

        {!gameStarted ? (
          /* Start Screen */
          <Card className="border-2 border-border shadow-card">
            <CardContent className="p-8 text-center">
              <div className="w-24 h-24 mx-auto mb-6 rounded-full bg-primary/10 flex items-center justify-center">
                <Brain className="h-12 w-12 text-primary" />
              </div>
              <h3 className="text-2xl font-bold mb-4">Ready to Practice?</h3>
              <p className="text-lg text-muted-foreground mb-6">
                I'll show you photos of your family members. Try to remember who they are!
              </p>

              {/* Difficulty Selection */}
              <div className="mb-6">
                <p className="text-base font-semibold mb-3">Choose difficulty:</p>
                <div className="flex justify-center gap-3">
                  <Button
                    variant={difficulty === 'easy' ? 'accessible' : 'accessible-outline'}
                    onClick={() => setDifficulty('easy')}
                  >
                    Easy (2 choices)
                  </Button>
                  <Button
                    variant={difficulty === 'medium' ? 'accessible' : 'accessible-outline'}
                    onClick={() => setDifficulty('medium')}
                  >
                    Medium (3 choices)
                  </Button>
                  {familyMembers.length >= 4 && (
                    <Button
                      variant={difficulty === 'hard' ? 'accessible' : 'accessible-outline'}
                      onClick={() => setDifficulty('hard')}
                    >
                      Hard (4 choices)
                    </Button>
                  )}
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

              <Button variant="accessible" size="xl" onClick={startGame} className="gap-3">
                <Sparkles className="h-6 w-6" />
                Start Game
              </Button>
            </CardContent>
          </Card>
        ) : (
          /* Game Screen */
          <div className="space-y-6">
            {/* Score Bar */}
            <Card className="border-2 border-border">
              <CardContent className="p-4">
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-4">
                    <Badge className="bg-primary text-primary-foreground text-lg px-4 py-1">
                      Score: {score}/{totalQuestions}
                    </Badge>
                    {streak > 0 && (
                      <Badge className="bg-accent text-accent-foreground text-lg px-4 py-1">
                        🔥 Streak: {streak}
                      </Badge>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={() => setSpeakEnabled(!speakEnabled)}
                      className="h-10 w-10"
                    >
                      {speakEnabled ? <Volume2 className="h-5 w-5" /> : <VolumeX className="h-5 w-5" />}
                    </Button>
                    <Button variant="outline" onClick={resetGame} className="gap-2">
                      <RotateCcw className="h-5 w-5" />
                      Reset
                    </Button>
                  </div>
                </div>
                {totalQuestions > 0 && (
                  <Progress value={(score / totalQuestions) * 100} className="mt-3 h-3" />
                )}
              </CardContent>
            </Card>

            {/* Question Card */}
            {currentQuestion && (
              <Card className={`border-2 shadow-card transition-all duration-500 ${
                showResult 
                  ? isCorrect 
                    ? 'border-success bg-success/5' 
                    : 'border-destructive bg-destructive/5'
                  : 'border-border'
              }`}>
                <CardContent className="p-6">
                  {/* Photo */}
                  <div className="relative w-64 h-64 mx-auto mb-6 rounded-2xl overflow-hidden shadow-elevated">
                    <img
                      src={getPhoto(currentQuestion)}
                      alt="Who is this?"
                      className="w-full h-full object-cover"
                    />
                  </div>

                  <h3 className="text-2xl font-bold text-center mb-6">Who is this person?</h3>

                  {/* Options */}
                  <div className={`grid gap-4 ${options.length === 2 ? 'grid-cols-1 sm:grid-cols-2' : 'grid-cols-1 sm:grid-cols-2'}`}>
                    {options.map((member) => {
                      const isSelected = selectedAnswer?.id === member.id;
                      const isCorrectAnswer = member.id === currentQuestion.id;
                      
                      let buttonClass = 'border-2 h-auto py-4 text-xl font-semibold';
                      if (showResult) {
                        if (isCorrectAnswer) {
                          buttonClass += ' border-success bg-success text-success-foreground';
                        } else if (isSelected && !isCorrectAnswer) {
                          buttonClass += ' border-destructive bg-destructive text-destructive-foreground';
                        } else {
                          buttonClass += ' opacity-50';
                        }
                      } else {
                        buttonClass += ' hover:border-primary hover:bg-primary/10';
                      }

                      return (
                        <Button
                          key={member.id}
                          variant="outline"
                          className={buttonClass}
                          onClick={() => handleAnswer(member)}
                          disabled={selectedAnswer !== null}
                        >
                          {showResult && isCorrectAnswer && <Check className="h-6 w-6 mr-2" />}
                          {showResult && isSelected && !isCorrectAnswer && <X className="h-6 w-6 mr-2" />}
                          {member.name}
                        </Button>
                      );
                    })}
                  </div>

                  {/* Result */}
                  {showResult && (
                    <div className={`mt-6 p-6 rounded-xl animate-scale-in ${
                      isCorrect ? 'bg-success/20' : 'bg-primary/10'
                    }`}>
                      <div className="flex items-center justify-center gap-3 mb-3">
                        {isCorrect ? (
                          <Trophy className="h-8 w-8 text-success" />
                        ) : (
                          <Heart className="h-8 w-8 text-primary" fill="currentColor" />
                        )}
                        <span className="text-xl font-bold">
                          {isCorrect ? 'Correct!' : "Let's learn together!"}
                        </span>
                      </div>
                      <p className="text-center text-lg">
                        This is <strong>{currentQuestion.name}</strong>, {currentQuestion.relationship_label}.
                      </p>
                      {currentQuestion.notes && (
                        <p className="text-center text-muted-foreground mt-2">
                          {currentQuestion.notes}
                        </p>
                      )}
                      <div className="flex justify-center mt-6">
                        <Button variant="accessible" size="lg" onClick={nextQuestion} className="gap-2">
                          Next Person
                          <ChevronRight className="h-5 w-5" />
                        </Button>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Best Streak */}
            {bestStreak > 0 && (
              <div className="text-center">
                <Badge className="bg-accent/20 text-accent-foreground text-base px-4 py-2">
                  🏆 Best Streak: {bestStreak}
                </Badge>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
};

export default WhoIsThisQuiz;
