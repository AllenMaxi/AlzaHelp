import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Pause, SkipBack, SkipForward, Volume2, VolumeX, Film, ChevronLeft, ChevronRight, X, Heart } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Dialog, DialogContent } from '@/components/ui/dialog';

// Placeholder images
const placeholderImages = {
  milestone: 'https://images.unsplash.com/photo-1519741497674-611481863552?w=800&h=600&fit=crop',
  family: 'https://images.unsplash.com/photo-1555252333-9f8e92e65df9?w=800&h=600&fit=crop',
  travel: 'https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=800&h=600&fit=crop',
  celebration: 'https://images.unsplash.com/photo-1530103862676-de8c9debad1d?w=800&h=600&fit=crop',
  other: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&h=600&fit=crop',
};

export const PhotoStories = ({ memories = [], familyMembers = [] }) => {
  const [selectedStory, setSelectedStory] = useState(null);
  const [currentSlide, setCurrentSlide] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [progress, setProgress] = useState(0);
  const intervalRef = useRef(null);
  const speechRef = useRef(null);

  const SLIDE_DURATION = 8000; // 8 seconds per slide

  const getPhoto = (memory) => {
    if (memory.photos && memory.photos.length > 0) {
      return memory.photos[0];
    }
    return placeholderImages[memory.category] || placeholderImages.other;
  };

  const speak = useCallback((text, onEnd) => {
    if (!voiceEnabled || !('speechSynthesis' in window)) {
      if (onEnd) onEnd();
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.75;
    utterance.pitch = 1;
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => {
      setIsSpeaking(false);
      if (onEnd) onEnd();
    };
    utterance.onerror = () => {
      setIsSpeaking(false);
      if (onEnd) onEnd();
    };
    speechRef.current = utterance;
    window.speechSynthesis.speak(utterance);
  }, [voiceEnabled]);

  const stopSpeaking = useCallback(() => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    setIsSpeaking(false);
  }, []);

  const generateNarration = (memory) => {
    const peopleStr = memory.people?.length > 0 
      ? `The people in this memory are: ${memory.people.join(', ')}.` 
      : '';
    const locationStr = memory.location ? `This was at ${memory.location}.` : '';
    
    return `${memory.title}. ${memory.date}. ${locationStr} ${memory.description} ${peopleStr}`;
  };

  const playSlide = useCallback((index) => {
    if (!selectedStory) return;
    
    setCurrentSlide(index);
    setProgress(0);
    stopSpeaking();

    // Generate and speak narration
    const narration = generateNarration(selectedStory);
    speak(narration);

    // Progress animation
    let progressValue = 0;
    const progressInterval = setInterval(() => {
      progressValue += 100 / (SLIDE_DURATION / 100);
      setProgress(Math.min(progressValue, 100));
    }, 100);

    // Clear on cleanup
    intervalRef.current = setTimeout(() => {
      clearInterval(progressInterval);
    }, SLIDE_DURATION);

  }, [selectedStory, speak, stopSpeaking]);

  const startSlideshow = () => {
    setIsPlaying(true);
    playSlide(currentSlide);
  };

  const pauseSlideshow = () => {
    setIsPlaying(false);
    stopSpeaking();
    if (intervalRef.current) {
      clearTimeout(intervalRef.current);
    }
  };

  const nextSlide = () => {
    // For now we only have one slide per memory
    // In future, could iterate through multiple photos
    pauseSlideshow();
  };

  const prevSlide = () => {
    pauseSlideshow();
  };

  const openStory = (memory) => {
    setSelectedStory(memory);
    setCurrentSlide(0);
    setProgress(0);
    setIsPlaying(false);
  };

  const closeStory = () => {
    pauseSlideshow();
    stopSpeaking();
    setSelectedStory(null);
    setCurrentSlide(0);
    setProgress(0);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopSpeaking();
      if (intervalRef.current) {
        clearTimeout(intervalRef.current);
      }
    };
  }, [stopSpeaking]);

  // Auto-play when story opens
  useEffect(() => {
    if (selectedStory && isPlaying) {
      playSlide(currentSlide);
    }
  }, [selectedStory, isPlaying, currentSlide, playSlide]);

  if (memories.length === 0) {
    return (
      <section className="py-8 sm:py-12">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-4xl">
          <Card className="border-2 border-border shadow-card">
            <CardContent className="p-8 text-center">
              <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-muted flex items-center justify-center">
                <Film className="h-10 w-10 text-muted-foreground" />
              </div>
              <h3 className="text-xl font-semibold mb-2">No Stories Yet</h3>
              <p className="text-muted-foreground">
                Add memories to create photo stories you can watch and listen to.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>
    );
  }

  return (
    <section className="py-8 sm:py-12">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-4 py-2 mb-4">
            <Film className="h-5 w-5 text-primary" />
            <span className="text-base font-medium text-primary">Relive Your Memories</span>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl font-bold text-foreground mb-4">
            Photo Stories
          </h2>
          <p className="text-accessible text-muted-foreground max-w-2xl mx-auto">
            Watch and listen to your precious memories come to life
          </p>
        </div>

        {/* Stories Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {memories.map((memory, index) => (
            <Card
              key={memory.id}
              className="group cursor-pointer border-2 border-border shadow-card hover:shadow-elevated transition-all duration-300 hover:-translate-y-1 overflow-hidden animate-scale-in"
              style={{ animationDelay: `${index * 0.1}s` }}
              onClick={() => openStory(memory)}
            >
              <CardContent className="p-0">
                {/* Thumbnail */}
                <div className="relative h-48 overflow-hidden">
                  <img
                    src={getPhoto(memory)}
                    alt={memory.title}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-background via-transparent to-transparent" />
                  
                  {/* Play button overlay */}
                  <div className="absolute inset-0 flex items-center justify-center bg-black/30 opacity-0 group-hover:opacity-100 transition-opacity">
                    <div className="w-16 h-16 rounded-full bg-primary flex items-center justify-center shadow-elevated">
                      <Play className="h-8 w-8 text-primary-foreground ml-1" fill="currentColor" />
                    </div>
                  </div>
                  
                  <Badge className="absolute top-3 right-3 bg-primary text-primary-foreground">
                    {memory.year}
                  </Badge>
                </div>
                
                {/* Info */}
                <div className="p-4">
                  <h3 className="text-lg font-bold text-foreground mb-1 line-clamp-1">{memory.title}</h3>
                  <p className="text-sm text-muted-foreground">{memory.date}</p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Story Viewer Dialog */}
        <Dialog open={!!selectedStory} onOpenChange={() => closeStory()}>
          <DialogContent className="max-w-4xl max-h-[90vh] p-0 overflow-hidden">
            {selectedStory && (
              <div className="relative bg-black">
                {/* Close button */}
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={closeStory}
                  className="absolute top-4 right-4 z-20 bg-black/50 hover:bg-black/70 text-white h-12 w-12"
                >
                  <X className="h-6 w-6" />
                </Button>

                {/* Main Image */}
                <div className="relative aspect-video">
                  <img
                    src={getPhoto(selectedStory)}
                    alt={selectedStory.title}
                    className="w-full h-full object-cover"
                  />
                  
                  {/* Gradient overlay for text readability */}
                  <div className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-black/30" />
                  
                  {/* Speaking indicator */}
                  {isSpeaking && (
                    <div className="absolute top-4 left-4 flex items-center gap-2 bg-primary/90 text-primary-foreground px-4 py-2 rounded-full">
                      <Volume2 className="h-5 w-5 animate-pulse" />
                      <span className="text-sm font-medium">Narrating...</span>
                    </div>
                  )}

                  {/* Title overlay */}
                  <div className="absolute bottom-0 left-0 right-0 p-6">
                    <h3 className="text-3xl font-bold text-white mb-2">{selectedStory.title}</h3>
                    <p className="text-xl text-white/90">{selectedStory.date}</p>
                    {selectedStory.location && (
                      <p className="text-lg text-white/80 mt-1">{selectedStory.location}</p>
                    )}
                  </div>
                </div>

                {/* Progress bar */}
                <Progress value={progress} className="h-1 rounded-none" />

                {/* Controls */}
                <div className="bg-background p-4">
                  <div className="flex items-center justify-center gap-4 mb-4">
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={prevSlide}
                      className="h-12 w-12"
                      disabled
                    >
                      <SkipBack className="h-5 w-5" />
                    </Button>
                    
                    <Button
                      variant="accessible"
                      size="icon"
                      onClick={isPlaying ? pauseSlideshow : startSlideshow}
                      className="h-16 w-16"
                    >
                      {isPlaying ? (
                        <Pause className="h-8 w-8" />
                      ) : (
                        <Play className="h-8 w-8 ml-1" />
                      )}
                    </Button>
                    
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={nextSlide}
                      className="h-12 w-12"
                      disabled
                    >
                      <SkipForward className="h-5 w-5" />
                    </Button>
                  </div>

                  {/* Voice toggle */}
                  <div className="flex justify-center items-center gap-3">
                    <span className="text-sm text-muted-foreground">Voice narration:</span>
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={() => setVoiceEnabled(!voiceEnabled)}
                      className="h-10 w-10"
                    >
                      {voiceEnabled ? <Volume2 className="h-5 w-5" /> : <VolumeX className="h-5 w-5" />}
                    </Button>
                  </div>

                  {/* Description */}
                  <div className="mt-4 p-4 bg-muted rounded-xl">
                    <div className="flex items-start gap-3">
                      <Heart className="h-6 w-6 text-primary shrink-0 mt-1" fill="currentColor" />
                      <div>
                        <p className="text-accessible">{selectedStory.description}</p>
                        {selectedStory.people?.length > 0 && (
                          <p className="text-base text-muted-foreground mt-2">
                            People: {selectedStory.people.join(', ')}
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </DialogContent>
        </Dialog>
      </div>
    </section>
  );
};

export default PhotoStories;
