import React, { useState, useEffect } from "react";
import "@/App.css";
import { Header } from "@/components/layout/Header";
import { HeroSection } from "@/components/sections/HeroSection";
import { FamilySection } from "@/components/sections/FamilySection";
import { TimelineSection } from "@/components/sections/TimelineSection";
import { AssistantSection } from "@/components/sections/AssistantSection";
import { RemindersSection } from "@/components/sections/RemindersSection";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Users, Calendar, MessageCircle, Bell, Heart } from "lucide-react";

function App() {
  const [currentView, setCurrentView] = useState('home');
  const [activeTab, setActiveTab] = useState('family');
  const [darkMode, setDarkMode] = useState(false);

  // Apply dark mode class to html element
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  const tabItems = [
    { id: 'family', label: 'Family', icon: Users },
    { id: 'timeline', label: 'Memories', icon: Calendar },
    { id: 'assistant', label: 'Ask Me', icon: MessageCircle },
    { id: 'reminders', label: 'Today', icon: Bell },
  ];

  return (
    <div className="min-h-screen bg-background">
      <Header 
        currentView={currentView} 
        setCurrentView={setCurrentView}
        darkMode={darkMode}
        setDarkMode={setDarkMode}
      />
      
      {currentView === 'home' && (
        <HeroSection 
          setCurrentView={setCurrentView}
          setActiveTab={setActiveTab}
        />
      )}
      
      {currentView === 'memories' && (
        <main className="min-h-[calc(100vh-5rem)]">
          {/* Page Header */}
          <div className="bg-gradient-hero py-8 sm:py-12 border-b border-border">
            <div className="container mx-auto px-4 sm:px-6 lg:px-8 text-center">
              <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-4 py-2 mb-4">
                <Heart className="h-5 w-5 text-primary" fill="currentColor" />
                <span className="text-base font-medium text-primary">Your Safe Space</span>
              </div>
              <h1 className="font-display text-3xl sm:text-4xl lg:text-5xl font-bold text-foreground">
                Your Memories
              </h1>
              <p className="text-accessible text-muted-foreground mt-3 max-w-2xl mx-auto">
                Everything you love is right here. Choose what you'd like to see.
              </p>
            </div>
          </div>

          {/* Tabs Navigation */}
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <div className="sticky top-20 z-40 bg-background/95 backdrop-blur border-b border-border">
              <div className="container mx-auto px-4 sm:px-6 lg:px-8">
                <TabsList className="h-auto p-2 bg-muted/50 rounded-2xl my-4 grid grid-cols-2 sm:grid-cols-4 gap-2">
                  {tabItems.map((tab) => {
                    const Icon = tab.icon;
                    return (
                      <TabsTrigger
                        key={tab.id}
                        value={tab.id}
                        className="flex items-center gap-2 py-4 px-6 text-lg font-semibold rounded-xl data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-card transition-all duration-300"
                      >
                        <Icon className="h-5 w-5" />
                        <span className="hidden sm:inline">{tab.label}</span>
                      </TabsTrigger>
                    );
                  })}
                </TabsList>
              </div>
            </div>

            <TabsContent value="family" className="mt-0 animate-fade-in">
              <FamilySection />
            </TabsContent>
            
            <TabsContent value="timeline" className="mt-0 animate-fade-in">
              <TimelineSection />
            </TabsContent>
            
            <TabsContent value="assistant" className="mt-0 animate-fade-in">
              <AssistantSection />
            </TabsContent>
            
            <TabsContent value="reminders" className="mt-0 animate-fade-in">
              <RemindersSection />
            </TabsContent>
          </Tabs>
        </main>
      )}

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
}

export default App;
