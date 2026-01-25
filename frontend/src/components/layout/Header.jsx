import React from 'react';
import { Heart, Sun, Moon, Home, Menu } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Sheet, SheetContent, SheetTrigger, SheetTitle } from '@/components/ui/sheet';

export const Header = ({ currentView, setCurrentView, darkMode, setDarkMode }) => {
  const navItems = [
    { id: 'home', label: 'Home', icon: Home },
    { id: 'memories', label: 'My Memories', icon: Heart },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/50 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex h-20 items-center justify-between">
          {/* Logo */}
          <button 
            onClick={() => setCurrentView('home')}
            className="flex items-center gap-3 hover:opacity-80 transition-opacity"
          >
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary shadow-soft">
              <Heart className="h-7 w-7 text-primary-foreground" fill="currentColor" />
            </div>
            <div className="hidden sm:block">
              <h1 className="text-2xl font-bold text-foreground font-display">MemoryKeeper</h1>
              <p className="text-sm text-muted-foreground">Your memories, always close</p>
            </div>
          </button>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentView === item.id;
              return (
                <Button
                  key={item.id}
                  variant={isActive ? "default" : "ghost"}
                  size="lg"
                  onClick={() => setCurrentView(item.id)}
                  className={`gap-2 ${isActive ? '' : 'text-muted-foreground hover:text-foreground'}`}
                >
                  <Icon className="h-5 w-5" />
                  <span className="text-lg">{item.label}</span>
                </Button>
              );
            })}
          </nav>

          {/* Right side - Theme toggle and mobile menu */}
          <div className="flex items-center gap-3">
            <Button
              variant="outline"
              size="icon"
              onClick={() => setDarkMode(!darkMode)}
              className="h-12 w-12"
              aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {darkMode ? (
                <Sun className="h-6 w-6 text-accent" />
              ) : (
                <Moon className="h-6 w-6 text-primary" />
              )}
            </Button>

            {/* Mobile Menu */}
            <Sheet>
              <SheetTrigger asChild className="md:hidden">
                <Button variant="outline" size="icon" className="h-12 w-12">
                  <Menu className="h-6 w-6" />
                </Button>
              </SheetTrigger>
              <SheetContent side="right" className="w-[300px] sm:w-[350px]">
                <SheetTitle className="text-2xl font-bold mb-8">Menu</SheetTitle>
                <nav className="flex flex-col gap-4 mt-8">
                  {navItems.map((item) => {
                    const Icon = item.icon;
                    const isActive = currentView === item.id;
                    return (
                      <Button
                        key={item.id}
                        variant={isActive ? "accessible" : "accessible-outline"}
                        onClick={() => setCurrentView(item.id)}
                        className="justify-start gap-4"
                      >
                        <Icon className="h-6 w-6" />
                        <span>{item.label}</span>
                      </Button>
                    );
                  })}
                </nav>
              </SheetContent>
            </Sheet>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
