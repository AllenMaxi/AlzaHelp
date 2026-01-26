import React, { useState, useCallback } from 'react';
import { Calendar, MapPin, Users, Heart, ChevronLeft, ChevronRight, X, ZoomIn, Plus, Trash2, Loader2, Volume2, VolumeX, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { AddMemoryForm } from '@/components/forms/AddMemoryForm';
import { memoriesApi } from '@/services/api';
import { toast } from 'sonner';

const categoryColors = {
  milestone: 'bg-primary text-primary-foreground',
  family: 'bg-family-children text-white',
  travel: 'bg-family-grandchildren text-foreground',
  celebration: 'bg-accent text-accent-foreground',
  other: 'bg-muted text-muted-foreground',
};

// Placeholder images for memories without photos
const placeholderImages = {
  milestone: 'https://images.unsplash.com/photo-1519741497674-611481863552?w=800&h=600&fit=crop',
  family: 'https://images.unsplash.com/photo-1555252333-9f8e92e65df9?w=800&h=600&fit=crop',
  travel: 'https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=800&h=600&fit=crop',
  celebration: 'https://images.unsplash.com/photo-1530103862676-de8c9debad1d?w=800&h=600&fit=crop',
  other: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&h=600&fit=crop',
};

const decades = [
  { label: 'All Time', value: 'all' },
  { label: '1970s', value: '1970' },
  { label: '1980s', value: '1980' },
  { label: '1990s', value: '1990' },
  { label: '2000s', value: '2000' },
  { label: '2010s', value: '2010' },
  { label: '2020s', value: '2020' },
];

export const TimelineSection = ({ memories = [], familyMembers = [], onRefresh, loading }) => {
  const [selectedMemory, setSelectedMemory] = useState(null);
  const [filter, setFilter] = useState('all');
  const [showAddForm, setShowAddForm] = useState(false);
  const [deleting, setDeleting] = useState(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true);

  // Voice narration for "Tell My Story"
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
    window.speechSynthesis.speak(utterance);
  }, [voiceEnabled]);

  const stopSpeaking = useCallback(() => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    setIsSpeaking(false);
  }, []);

  const tellMyStory = (memory) => {
    const peopleStr = memory.people?.length > 0 
      ? `The people in this memory are: ${memory.people.join(', ')}.` 
      : '';
    const locationStr = memory.location ? `This was at ${memory.location}.` : '';
    
    const narration = `${memory.title}. ${memory.date}. ${locationStr} ${memory.description} ${peopleStr}`;
    speak(narration);
    toast.success('Playing your memory story...');
  };

  const filteredMemories = filter === 'all' 
    ? memories 
    : memories.filter(m => {
        const decade = parseInt(filter);
        return m.year >= decade && m.year < decade + 10;
      });

  // Sort by year descending
  const sortedMemories = [...filteredMemories].sort((a, b) => b.year - a.year);

  const handlePrevious = () => {
    const currentFilteredIndex = sortedMemories.findIndex(m => m.id === selectedMemory.id);
    if (currentFilteredIndex > 0) {
      setSelectedMemory(sortedMemories[currentFilteredIndex - 1]);
    }
  };

  const handleNext = () => {
    const currentFilteredIndex = sortedMemories.findIndex(m => m.id === selectedMemory.id);
    if (currentFilteredIndex < sortedMemories.length - 1) {
      setSelectedMemory(sortedMemories[currentFilteredIndex + 1]);
    }
  };

  const handleDelete = async (memoryId, memoryTitle) => {
    if (!window.confirm(`Are you sure you want to delete "${memoryTitle}"?`)) return;
    
    setDeleting(memoryId);
    try {
      await memoriesApi.delete(memoryId);
      toast.success('Memory deleted');
      setSelectedMemory(null);
      onRefresh();
    } catch (error) {
      toast.error('Failed to delete memory');
    } finally {
      setDeleting(null);
    }
  };

  const getPhoto = (memory) => {
    if (memory.photos && memory.photos.length > 0) {
      return memory.photos[0];
    }
    return placeholderImages[memory.category] || placeholderImages.other;
  };

  return (
    <section className="py-8 sm:py-12">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-4 py-2 mb-4">
            <Calendar className="h-5 w-5 text-primary" />
            <span className="text-base font-medium text-primary">Your Story</span>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl font-bold text-foreground mb-4">
            Memory Timeline
          </h2>
          <p className="text-accessible text-muted-foreground max-w-2xl mx-auto">
            A journey through your most precious moments
          </p>
        </div>

        {/* Add button and Decade Filter */}
        <div className="flex flex-wrap justify-center gap-3 mb-10">
          <Button
            variant="accessible"
            onClick={() => setShowAddForm(true)}
            className="gap-2"
          >
            <Plus className="h-5 w-5" />
            Add Memory
          </Button>
          
          <div className="w-full sm:w-auto" />
          
          {decades.map((decade) => (
            <Button
              key={decade.value}
              variant={filter === decade.value ? 'accessible' : 'accessible-outline'}
              onClick={() => setFilter(decade.value)}
            >
              {decade.label}
            </Button>
          ))}
        </div>

        {/* Loading state */}
        {loading && (
          <div className="flex justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        )}

        {/* Empty state */}
        {!loading && memories.length === 0 && (
          <div className="text-center py-12">
            <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-muted flex items-center justify-center">
              <Calendar className="h-10 w-10 text-muted-foreground" />
            </div>
            <h3 className="text-xl font-semibold mb-2">No memories yet</h3>
            <p className="text-muted-foreground mb-6">Add your first memory to start your timeline</p>
            <Button variant="accessible" onClick={() => setShowAddForm(true)}>
              <Plus className="h-5 w-5 mr-2" />
              Add Memory
            </Button>
          </div>
        )}

        {/* Timeline */}
        {!loading && sortedMemories.length > 0 && (
          <div className="relative">
            {/* Timeline line */}
            <div className="absolute left-4 sm:left-1/2 top-0 bottom-0 w-1 bg-border sm:-translate-x-1/2" />
            
            <div className="space-y-8">
              {sortedMemories.map((memory, index) => (
                <div
                  key={memory.id}
                  className={`relative flex items-center gap-6 animate-slide-up ${
                    index % 2 === 0 ? 'sm:flex-row' : 'sm:flex-row-reverse'
                  }`}
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  {/* Timeline dot */}
                  <div className="absolute left-4 sm:left-1/2 w-4 h-4 rounded-full bg-primary border-4 border-background shadow-soft sm:-translate-x-1/2 z-10" />
                  
                  {/* Content */}
                  <div className={`ml-12 sm:ml-0 sm:w-[45%] ${index % 2 === 0 ? 'sm:pr-8' : 'sm:pl-8'}`}>
                    <Card
                      className="cursor-pointer border-2 border-border shadow-card hover:shadow-elevated transition-all duration-300 hover:-translate-y-1 overflow-hidden group"
                      onClick={() => setSelectedMemory(memory)}
                    >
                      <CardContent className="p-0">
                        {/* Photo */}
                        <div className="relative h-48 overflow-hidden">
                          <img
                            src={getPhoto(memory)}
                            alt={memory.title}
                            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                          />
                          <div className="absolute inset-0 bg-gradient-to-t from-background via-transparent to-transparent" />
                          <Badge className={`absolute top-4 right-4 ${categoryColors[memory.category] || categoryColors.other}`}>
                            {memory.category}
                          </Badge>
                          <div className="absolute bottom-4 left-4">
                            <p className="text-lg font-bold text-foreground">{memory.date}</p>
                          </div>
                        </div>
                        
                        {/* Info */}
                        <div className="p-6">
                          <h3 className="text-xl font-bold text-foreground mb-2">{memory.title}</h3>
                          
                          {memory.location && (
                            <div className="flex items-center gap-2 text-muted-foreground mb-3">
                              <MapPin className="h-4 w-4" />
                              <span className="text-sm">{memory.location}</span>
                            </div>
                          )}
                          
                          {memory.people && memory.people.length > 0 && (
                            <div className="flex items-center gap-2 text-muted-foreground">
                              <Users className="h-4 w-4" />
                              <span className="text-sm">{memory.people.join(', ')}</span>
                            </div>
                          )}
                          
                          <Button variant="ghost" className="w-full mt-4 gap-2">
                            <ZoomIn className="h-5 w-5" />
                            View Memory
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                  
                  {/* Date label (desktop) */}
                  <div className={`hidden sm:block sm:w-[45%] ${index % 2 === 0 ? 'sm:pl-8 text-left' : 'sm:pr-8 text-right'}`}>
                    <p className="text-2xl font-bold text-primary">{memory.year}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Add Form Dialog */}
        <Dialog open={showAddForm} onOpenChange={setShowAddForm}>
          <DialogContent className="max-w-lg max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle className="text-2xl font-display flex items-center gap-2">
                <Calendar className="h-6 w-6 text-primary" />
                Add New Memory
              </DialogTitle>
            </DialogHeader>
            <AddMemoryForm 
              familyMembers={familyMembers}
              onSuccess={() => {
                setShowAddForm(false);
                onRefresh();
              }}
              onClose={() => setShowAddForm(false)}
            />
          </DialogContent>
        </Dialog>

        {/* Detail Dialog */}
        <Dialog open={!!selectedMemory} onOpenChange={() => setSelectedMemory(null)}>
          <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
            {selectedMemory && (
              <>
                <DialogHeader>
                  <DialogTitle className="text-3xl font-display">{selectedMemory.title}</DialogTitle>
                  <p className="text-xl text-primary font-medium">
                    {selectedMemory.date}
                  </p>
                </DialogHeader>
                
                <div className="mt-6 space-y-6">
                  {/* Large Photo */}
                  <div className="relative rounded-2xl overflow-hidden shadow-card">
                    <img
                      src={getPhoto(selectedMemory)}
                      alt={selectedMemory.title}
                      className="w-full h-64 sm:h-96 object-cover"
                    />
                  </div>
                  
                  {/* Multiple photos gallery */}
                  {selectedMemory.photos && selectedMemory.photos.length > 1 && (
                    <div className="flex gap-2 overflow-x-auto pb-2">
                      {selectedMemory.photos.map((photo, idx) => (
                        <img
                          key={idx}
                          src={photo}
                          alt={`Memory photo ${idx + 1}`}
                          className="h-20 w-20 object-cover rounded-lg flex-shrink-0"
                        />
                      ))}
                    </div>
                  )}
                  
                  {/* Details */}
                  <div className="space-y-4">
                    {selectedMemory.location && (
                      <div className="flex items-start gap-4 p-4 rounded-xl bg-muted">
                        <MapPin className="h-6 w-6 text-primary mt-1" />
                        <div>
                          <p className="font-semibold text-lg">Location</p>
                          <p className="text-accessible text-muted-foreground">{selectedMemory.location}</p>
                        </div>
                      </div>
                    )}
                    
                    {selectedMemory.people && selectedMemory.people.length > 0 && (
                      <div className="flex items-start gap-4 p-4 rounded-xl bg-muted">
                        <Users className="h-6 w-6 text-primary mt-1" />
                        <div>
                          <p className="font-semibold text-lg">Who Was There</p>
                          <p className="text-accessible text-muted-foreground">{selectedMemory.people.join(', ')}</p>
                        </div>
                      </div>
                    )}
                    
                    <div className="flex items-start gap-4 p-4 rounded-xl bg-primary/10 border-2 border-primary">
                      <Heart className="h-6 w-6 text-primary mt-1" fill="currentColor" />
                      <div>
                        <p className="font-semibold text-lg text-primary">The Story</p>
                        <p className="text-accessible text-foreground mt-2">{selectedMemory.description}</p>
                      </div>
                    </div>
                  </div>
                  
                  {/* Navigation */}
                  <div className="flex gap-4">
                    <Button 
                      variant="accessible-outline" 
                      className="flex-1"
                      onClick={handlePrevious}
                      disabled={sortedMemories.findIndex(m => m.id === selectedMemory.id) === 0}
                    >
                      <ChevronLeft className="h-6 w-6 mr-2" />
                      Previous
                    </Button>
                    <Button 
                      variant="accessible-outline" 
                      className="flex-1"
                      onClick={handleNext}
                      disabled={sortedMemories.findIndex(m => m.id === selectedMemory.id) === sortedMemories.length - 1}
                    >
                      Next
                      <ChevronRight className="h-6 w-6 ml-2" />
                    </Button>
                  </div>
                  
                  <div className="flex gap-4">
                    <Button 
                      variant="accessible" 
                      className="flex-1"
                      onClick={() => setSelectedMemory(null)}
                    >
                      <X className="h-6 w-6 mr-2" />
                      Close
                    </Button>
                    <Button 
                      variant="destructive"
                      className="gap-2"
                      onClick={() => handleDelete(selectedMemory.id, selectedMemory.title)}
                      disabled={deleting === selectedMemory.id}
                    >
                      {deleting === selectedMemory.id ? (
                        <Loader2 className="h-5 w-5 animate-spin" />
                      ) : (
                        <Trash2 className="h-5 w-5" />
                      )}
                    </Button>
                  </div>
                </div>
              </>
            )}
          </DialogContent>
        </Dialog>
      </div>
    </section>
  );
};

export default TimelineSection;
