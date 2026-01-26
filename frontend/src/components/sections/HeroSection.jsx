import React from 'react';
import { Heart, Users, Calendar, MessageCircle, Brain, Film, CalendarDays, Bell } from 'lucide-react';
import { Button } from '@/components/ui/button';

export const HeroSection = ({ setCurrentView, setActiveTab, userName = 'Friend' }) => {
  const quickActions = [
    {
      id: 'family',
      label: 'My Family',
      description: 'See your loved ones',
      icon: Users,
      color: 'bg-family-spouse/20 border-family-spouse hover:bg-family-spouse/30',
      tab: 'family'
    },
    {
      id: 'timeline',
      label: 'My Memories',
      description: 'View your timeline',
      icon: Calendar,
      color: 'bg-family-children/20 border-family-children hover:bg-family-children/30',
      tab: 'timeline'
    },
    {
      id: 'quiz',
      label: 'Memory Game',
      description: 'Practice remembering faces',
      icon: Brain,
      color: 'bg-accent/20 border-accent hover:bg-accent/30',
      tab: 'quiz'
    },
    {
      id: 'stories',
      label: 'Photo Stories',
      description: 'Watch & listen to memories',
      icon: Film,
      color: 'bg-primary/20 border-primary hover:bg-primary/30',
      tab: 'stories'
    },
    {
      id: 'week',
      label: 'My Week',
      description: 'What did I do recently?',
      icon: CalendarDays,
      color: 'bg-family-grandchildren/20 border-family-grandchildren hover:bg-family-grandchildren/30',
      tab: 'week'
    },
    {
      id: 'ask',
      label: 'Ask Me',
      description: 'I can help you remember',
      icon: MessageCircle,
      color: 'bg-success/20 border-success hover:bg-success/30',
      tab: 'assistant'
    },
  ];

  const handleQuickAction = (tab) => {
    setActiveTab(tab);
    setCurrentView('memories');
  };

  return (
    <section className="relative overflow-hidden bg-gradient-hero py-12 sm:py-16 lg:py-20">
      {/* Background decoration */}
      <div className="absolute inset-0 pattern-dots opacity-50" />
      
      <div className="container relative mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid gap-12 lg:grid-cols-2 lg:gap-16 items-center">
          {/* Left content */}
          <div className="text-center lg:text-left animate-fade-in">
            <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-4 py-2 mb-6">
              <Heart className="h-5 w-5 text-primary" fill="currentColor" />
              <span className="text-base font-medium text-primary">Welcome Home</span>
            </div>
            
            <h1 className="font-display text-4xl sm:text-5xl lg:text-6xl font-bold text-foreground leading-tight mb-6">
              Hello, <span className="text-primary">{userName}</span>
            </h1>
            
            <p className="text-accessible text-muted-foreground max-w-xl mx-auto lg:mx-0 mb-8">
              All your precious memories and loved ones are right here. 
              Just tap a button to see your family, play memory games, 
              or ask me anything you'd like to know.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
              <Button 
                variant="accessible" 
                size="xl"
                onClick={() => handleQuickAction('family')}
                className="gap-3"
              >
                <Users className="h-7 w-7" />
                See My Family
              </Button>
              <Button 
                variant="accessible-outline" 
                size="xl"
                onClick={() => handleQuickAction('quiz')}
                className="gap-3"
              >
                <Brain className="h-7 w-7" />
                Play Memory Game
              </Button>
            </div>
          </div>

          {/* Right - Hero Image */}
          <div className="relative animate-slide-up" style={{ animationDelay: '0.2s' }}>
            <div className="relative rounded-3xl overflow-hidden shadow-elevated">
              <img
                src="https://images.unsplash.com/photo-1600779438084-a87b966aab99?crop=entropy&cs=srgb&fm=jpg&ixid=M3w4NTYxODh8MHwxfHNlYXJjaHwxfHxncmFuZHBhcmVudHMlMjBncmFuZGNoaWxkcmVufGVufDB8fHx8MTc2OTM3NzY3N3ww&ixlib=rb-4.1.0&q=85"
                alt="Warm family moment"
                className="w-full h-[400px] object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-background/60 via-transparent to-transparent" />
              <div className="absolute bottom-6 left-6 right-6">
                <p className="text-xl font-semibold text-foreground">Your memories are safe with us</p>
              </div>
            </div>
            
            {/* Floating decoration */}
            <div className="absolute -top-4 -right-4 h-20 w-20 rounded-2xl bg-accent/20 blur-2xl animate-float" />
            <div className="absolute -bottom-4 -left-4 h-16 w-16 rounded-full bg-primary/20 blur-xl animate-float" style={{ animationDelay: '1s' }} />
          </div>
        </div>

        {/* Quick Actions Grid */}
        <div className="mt-16 sm:mt-20">
          <h2 className="text-2xl sm:text-3xl font-bold text-center mb-8 font-display">
            What would you like to do?
          </h2>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {quickActions.map((action, index) => {
              const Icon = action.icon;
              return (
                <button
                  key={action.id}
                  onClick={() => handleQuickAction(action.tab)}
                  className={`group flex flex-col items-center p-8 rounded-2xl border-2 shadow-card hover:shadow-elevated transition-all duration-300 hover:-translate-y-1 animate-scale-in ${action.color}`}
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-background shadow-soft mb-4 group-hover:scale-110 transition-transform duration-300">
                    <Icon className="h-10 w-10 text-primary" />
                  </div>
                  <h3 className="text-xl font-bold text-foreground mb-2">{action.label}</h3>
                  <p className="text-base text-muted-foreground text-center">{action.description}</p>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
