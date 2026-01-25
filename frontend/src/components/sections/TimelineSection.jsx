import React, { useState } from 'react';
import { Calendar, MapPin, Users, Heart, ChevronLeft, ChevronRight, X, ZoomIn } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';

// Mock memories data
const memories = [
  {
    id: 1,
    title: 'Our Wedding Day',
    date: 'June 15, 1972',
    year: 1972,
    location: 'St. Mary\'s Church, Hometown',
    people: ['Maria', 'Family', 'Friends'],
    description: 'The happiest day of your life! You married Maria at the little church on Maple Avenue. She wore her mother\'s beautiful white wedding dress. It was a sunny summer day and everyone danced until midnight.',
    photo: 'https://images.unsplash.com/photo-1519741497674-611481863552?w=800&h=600&fit=crop',
    category: 'milestone'
  },
  {
    id: 2,
    title: 'Michael Was Born',
    date: 'July 22, 1975',
    year: 1975,
    location: 'General Hospital',
    people: ['Maria', 'Michael'],
    description: 'Your first child, Michael, was born on a warm summer morning. He was 7 pounds, 8 ounces. You stayed up all night watching him sleep, amazed at how perfect he was.',
    photo: 'https://images.unsplash.com/photo-1555252333-9f8e92e65df9?w=800&h=600&fit=crop',
    category: 'family'
  },
  {
    id: 3,
    title: 'Sarah Was Born',
    date: 'November 8, 1978',
    year: 1978,
    location: 'General Hospital',
    people: ['Maria', 'Sarah', 'Michael'],
    description: 'Your daughter Sarah arrived on a crisp autumn day. Michael was so excited to be a big brother. He held her tiny hand and promised to always protect her.',
    photo: 'https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=800&h=600&fit=crop',
    category: 'family'
  },
  {
    id: 4,
    title: 'Family Vacation to Paris',
    date: 'August 1985',
    year: 1985,
    location: 'Paris, France',
    people: ['Maria', 'Michael', 'Sarah'],
    description: 'Your dream trip to Paris! You visited the Eiffel Tower, ate croissants every morning, and took a boat ride on the Seine. Michael was 10 and Sarah was 7. They still talk about this trip.',
    photo: 'https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=800&h=600&fit=crop',
    category: 'travel'
  },
  {
    id: 5,
    title: 'Bought Our House',
    date: 'March 1988',
    year: 1988,
    location: '123 Oak Street, Hometown',
    people: ['Maria', 'Michael', 'Sarah'],
    description: 'After years of saving, you finally bought your dream home - the house on Oak Street with the big backyard and the oak tree. You\'ve lived there for over 35 years now.',
    photo: 'https://images.unsplash.com/photo-1564013799919-ab600027ffc6?w=800&h=600&fit=crop',
    category: 'milestone'
  },
  {
    id: 6,
    title: 'Sarah\'s Graduation',
    date: 'May 2004',
    year: 2004,
    location: 'University Medical School',
    people: ['Maria', 'Michael', 'Sarah'],
    description: 'Sarah graduated from medical school! You were so proud, you cried happy tears. She worked so hard to become a doctor. Now she helps save lives every day.',
    photo: 'https://images.unsplash.com/photo-1523050854058-8df90110c9f1?w=800&h=600&fit=crop',
    category: 'milestone'
  },
  {
    id: 7,
    title: 'First Grandchild - Emma',
    date: 'April 3, 2012',
    year: 2012,
    location: 'City Hospital',
    people: ['Michael', 'Emma'],
    description: 'Your first grandchild, Emma, was born! You became "Grandpa" for the first time. She had the brightest eyes and grabbed your finger with her tiny hand.',
    photo: 'https://images.unsplash.com/photo-1676478966447-b3ca3565d3ad?w=800&h=600&fit=crop',
    category: 'family'
  },
  {
    id: 8,
    title: '50th Wedding Anniversary',
    date: 'June 15, 2022',
    year: 2022,
    location: 'The Grand Ballroom',
    people: ['Maria', 'Michael', 'Sarah', 'Emma', 'James', 'Lily'],
    description: '50 years of marriage! The whole family threw you a beautiful party. You danced with Maria to the same song from your wedding. All your grandchildren were there.',
    photo: 'https://images.unsplash.com/photo-1766808982663-50e31802011c?w=800&h=600&fit=crop',
    category: 'milestone'
  },
];

const categoryColors = {
  milestone: 'bg-primary text-primary-foreground',
  family: 'bg-family-children text-white',
  travel: 'bg-family-grandchildren text-foreground',
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

export const TimelineSection = () => {
  const [selectedMemory, setSelectedMemory] = useState(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [filter, setFilter] = useState('all');

  const filteredMemories = filter === 'all' 
    ? memories 
    : memories.filter(m => {
        const decade = parseInt(filter);
        return m.year >= decade && m.year < decade + 10;
      });

  const handlePrevious = () => {
    const currentFilteredIndex = filteredMemories.findIndex(m => m.id === selectedMemory.id);
    if (currentFilteredIndex > 0) {
      setSelectedMemory(filteredMemories[currentFilteredIndex - 1]);
    }
  };

  const handleNext = () => {
    const currentFilteredIndex = filteredMemories.findIndex(m => m.id === selectedMemory.id);
    if (currentFilteredIndex < filteredMemories.length - 1) {
      setSelectedMemory(filteredMemories[currentFilteredIndex + 1]);
    }
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

        {/* Decade Filter */}
        <div className="flex flex-wrap justify-center gap-3 mb-10">
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

        {/* Timeline */}
        <div className="relative">
          {/* Timeline line */}
          <div className="absolute left-4 sm:left-1/2 top-0 bottom-0 w-1 bg-border sm:-translate-x-1/2" />
          
          <div className="space-y-8">
            {filteredMemories.map((memory, index) => (
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
                          src={memory.photo}
                          alt={memory.title}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-background via-transparent to-transparent" />
                        <Badge className={`absolute top-4 right-4 ${categoryColors[memory.category]}`}>
                          {memory.category}
                        </Badge>
                        <div className="absolute bottom-4 left-4">
                          <p className="text-lg font-bold text-foreground">{memory.date}</p>
                        </div>
                      </div>
                      
                      {/* Info */}
                      <div className="p-6">
                        <h3 className="text-xl font-bold text-foreground mb-2">{memory.title}</h3>
                        
                        <div className="flex items-center gap-2 text-muted-foreground mb-3">
                          <MapPin className="h-4 w-4" />
                          <span className="text-sm">{memory.location}</span>
                        </div>
                        
                        <div className="flex items-center gap-2 text-muted-foreground">
                          <Users className="h-4 w-4" />
                          <span className="text-sm">{memory.people.join(', ')}</span>
                        </div>
                        
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

        {/* Detail Dialog */}
        <Dialog open={!!selectedMemory} onOpenChange={() => setSelectedMemory(null)}>
          <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
            {selectedMemory && (
              <>
                <DialogHeader>
                  <DialogTitle className="text-3xl font-display">{selectedMemory.title}</DialogTitle>
                  <DialogDescription className="text-xl text-primary font-medium">
                    {selectedMemory.date}
                  </DialogDescription>
                </DialogHeader>
                
                <div className="mt-6 space-y-6">
                  {/* Large Photo */}
                  <div className="relative rounded-2xl overflow-hidden shadow-card">
                    <img
                      src={selectedMemory.photo}
                      alt={selectedMemory.title}
                      className="w-full h-64 sm:h-96 object-cover"
                    />
                  </div>
                  
                  {/* Details */}
                  <div className="space-y-4">
                    <div className="flex items-start gap-4 p-4 rounded-xl bg-muted">
                      <MapPin className="h-6 w-6 text-primary mt-1" />
                      <div>
                        <p className="font-semibold text-lg">Location</p>
                        <p className="text-accessible text-muted-foreground">{selectedMemory.location}</p>
                      </div>
                    </div>
                    
                    <div className="flex items-start gap-4 p-4 rounded-xl bg-muted">
                      <Users className="h-6 w-6 text-primary mt-1" />
                      <div>
                        <p className="font-semibold text-lg">Who Was There</p>
                        <p className="text-accessible text-muted-foreground">{selectedMemory.people.join(', ')}</p>
                      </div>
                    </div>
                    
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
                      disabled={filteredMemories.findIndex(m => m.id === selectedMemory.id) === 0}
                    >
                      <ChevronLeft className="h-6 w-6 mr-2" />
                      Previous
                    </Button>
                    <Button 
                      variant="accessible-outline" 
                      className="flex-1"
                      onClick={handleNext}
                      disabled={filteredMemories.findIndex(m => m.id === selectedMemory.id) === filteredMemories.length - 1}
                    >
                      Next
                      <ChevronRight className="h-6 w-6 ml-2" />
                    </Button>
                  </div>
                  
                  <Button 
                    variant="accessible" 
                    className="w-full"
                    onClick={() => setSelectedMemory(null)}
                  >
                    <X className="h-6 w-6 mr-2" />
                    Close
                  </Button>
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
