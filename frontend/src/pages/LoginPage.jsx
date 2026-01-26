import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import { Button } from '@/components/ui/button';
import { Heart, LogIn, Users, Calendar, MessageCircle, Shield } from 'lucide-react';

export const LoginPage = () => {
  const { login, isAuthenticated } = useAuth();
  const navigate = useNavigate();

  // Redirect if already authenticated
  React.useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard');
    }
  }, [isAuthenticated, navigate]);

  const features = [
    {
      icon: Users,
      title: 'Family Directory',
      description: 'Keep all your loved ones\' information in one safe place'
    },
    {
      icon: Calendar,
      title: 'Memory Timeline',
      description: 'Store and relive your precious memories anytime'
    },
    {
      icon: MessageCircle,
      title: 'AI Assistant',
      description: 'Ask questions about your family and memories'
    },
    {
      icon: Shield,
      title: 'Safe & Private',
      description: 'Your memories are protected and only visible to you'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-hero">
      {/* Header */}
      <header className="py-6 px-4 sm:px-6 lg:px-8">
        <div className="container mx-auto flex items-center justify-center sm:justify-start">
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary shadow-soft">
              <Heart className="h-7 w-7 text-primary-foreground" fill="currentColor" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-foreground font-display">MemoryKeeper</h1>
              <p className="text-sm text-muted-foreground">Your memories, always close</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 sm:px-6 lg:px-8 py-12 sm:py-20">
        <div className="grid gap-12 lg:grid-cols-2 lg:gap-20 items-center">
          {/* Left - Welcome Text */}
          <div className="text-center lg:text-left">
            <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-4 py-2 mb-6">
              <Heart className="h-5 w-5 text-primary" fill="currentColor" />
              <span className="text-base font-medium text-primary">Welcome</span>
            </div>
            
            <h2 className="font-display text-4xl sm:text-5xl lg:text-6xl font-bold text-foreground leading-tight mb-6">
              Keep Your <span className="text-primary">Memories</span> Safe
            </h2>
            
            <p className="text-accessible text-muted-foreground max-w-xl mx-auto lg:mx-0 mb-8">
              MemoryKeeper helps you store and remember your loved ones, special moments, 
              and important information. Sign in with one click to get started.
            </p>

            <Button 
              variant="accessible" 
              size="xl"
              onClick={login}
              className="gap-3 w-full sm:w-auto"
            >
              <LogIn className="h-7 w-7" />
              Sign in with Google
            </Button>
            
            <p className="text-sm text-muted-foreground mt-4">
              Easy one-click sign in • No password needed
            </p>
          </div>

          {/* Right - Features */}
          <div className="grid gap-4 sm:grid-cols-2">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <div
                  key={feature.title}
                  className="p-6 rounded-2xl bg-card border-2 border-border shadow-card animate-scale-in"
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-primary/10 mb-4">
                    <Icon className="h-7 w-7 text-primary" />
                  </div>
                  <h3 className="text-xl font-bold text-foreground mb-2">{feature.title}</h3>
                  <p className="text-base text-muted-foreground">{feature.description}</p>
                </div>
              );
            })}
          </div>
        </div>

        {/* Hero Image */}
        <div className="mt-16 sm:mt-20 relative rounded-3xl overflow-hidden shadow-elevated max-w-4xl mx-auto">
          <img
            src="https://images.unsplash.com/photo-1600779438084-a87b966aab99?crop=entropy&cs=srgb&fm=jpg&ixid=M3w4NTYxODh8MHwxfHNlYXJjaHwxfHxncmFuZHBhcmVudHMlMjBncmFuZGNoaWxkcmVufGVufDB8fHx8MTc2OTM3NzY3N3ww&ixlib=rb-4.1.0&q=85"
            alt="Family memories"
            className="w-full h-[300px] sm:h-[400px] object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-background/80 via-transparent to-transparent" />
          <div className="absolute bottom-6 left-6 right-6 text-center">
            <p className="text-xl font-semibold text-foreground">
              Your precious memories deserve to be remembered
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border bg-muted/30 py-8 mt-auto">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="flex items-center justify-center gap-2 mb-3">
            <Heart className="h-5 w-5 text-primary" fill="currentColor" />
            <span className="font-display text-xl font-bold text-foreground">MemoryKeeper</span>
          </div>
          <p className="text-base text-muted-foreground">
            Made with love to help you remember what matters most.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default LoginPage;
