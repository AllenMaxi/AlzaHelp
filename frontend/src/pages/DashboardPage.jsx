import React, { useState, useEffect, useCallback } from "react";
import { useAuth } from "@/context/AuthContext";
import { Header } from "@/components/layout/Header";
import { AICompanion } from "@/components/AICompanion";
import { FamilySection } from "@/components/sections/FamilySection";
import { TimelineSection } from "@/components/sections/TimelineSection";
import { AssistantSection } from "@/components/sections/AssistantSection";
import { RemindersSection } from "@/components/sections/RemindersSection";
import { WhoIsThisQuiz } from "@/components/sections/WhoIsThisQuiz";
import { WeekMemories } from "@/components/sections/WeekMemories";
import { MemoryCardGame } from "@/components/sections/MemoryCardGame";
import { SudokuGame } from "@/components/sections/SudokuGame";
import { NavigationSection } from "@/components/sections/NavigationSection";
import { MedicationSection } from "@/components/sections/MedicationSection";
import { CaregiverPortalSection } from "@/components/sections/CaregiverPortalSection";
import { AdminSection } from "@/components/sections/AdminSection";
import { MoodBehaviorSection } from "@/components/sections/MoodBehaviorSection";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Users, Calendar, MessageCircle, Bell, Brain, CalendarDays, Home, Heart, Layers, Grid3X3, Route, Pill, ShieldCheck, Shield, Smile } from "lucide-react";
import { familyApi, memoriesApi, remindersApi, destinationsApi, medicationsApi } from "@/services/api";

export const DashboardPage = () => {
  const { user } = useAuth();
  const [currentView, setCurrentView] = useState('home');
  const [activeTab, setActiveTab] = useState('family');
  const [darkMode, setDarkMode] = useState(false);
  
  // Data states
  const [familyMembers, setFamilyMembers] = useState([]);
  const [memories, setMemories] = useState([]);
  const [reminders, setReminders] = useState([]);
  const [destinations, setDestinations] = useState([]);
  const [medications, setMedications] = useState([]);
  const [loading, setLoading] = useState(true);

  // Apply dark mode
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  // Load data on mount
  useEffect(() => {
    loadAllData();
  }, []);

  const loadAllData = async () => {
    setLoading(true);
    try {
      const [familyData, memoriesData, remindersData, destinationsData, medicationsData] = await Promise.all([
        familyApi.getAll().catch(() => []),
        memoriesApi.getAll().catch(() => []),
        remindersApi.getAll().catch(() => []),
        destinationsApi.getAll().catch(() => []),
        medicationsApi.getAll().catch(() => []),
      ]);
      setFamilyMembers(familyData);
      setMemories(memoriesData);
      setReminders(remindersData);
      setDestinations(destinationsData);
      setMedications(medicationsData);
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
    }
  };

  const refreshFamily = async () => {
    const data = await familyApi.getAll().catch(() => []);
    setFamilyMembers(data);
  };

  const refreshMemories = async () => {
    const data = await memoriesApi.getAll().catch(() => []);
    setMemories(data);
  };

  const refreshReminders = async () => {
    const data = await remindersApi.getAll().catch(() => []);
    setReminders(data);
  };

  const refreshDestinations = async () => {
    const data = await destinationsApi.getAll().catch(() => []);
    setDestinations(data);
  };

  const refreshMedications = async () => {
    const data = await medicationsApi.getAll().catch(() => []);
    setMedications(data);
  };

  // Voice assistant navigation handler
  const handleVoiceNavigate = useCallback((target) => {
    if (target === 'home') {
      setCurrentView('home');
    } else {
      setCurrentView('memories');
      setActiveTab(target);
    }
  }, []);

  const tabItems = [
    { id: 'family', label: 'Family', icon: Users },
    { id: 'timeline', label: 'Memories', icon: Calendar },
    { id: 'quiz', label: 'Faces Quiz', icon: Brain },
    { id: 'cards', label: 'Match', icon: Layers },
    { id: 'sudoku', label: 'Sudoku', icon: Grid3X3 },
    { id: 'week', label: 'My Week', icon: CalendarDays },
    { id: 'assistant', label: 'Ask Me', icon: MessageCircle },
    { id: 'reminders', label: 'Today', icon: Bell },
    { id: 'mood', label: 'Mood', icon: Smile },
    { id: 'navigation', label: 'Go To', icon: Route },
    { id: 'medications', label: 'Meds', icon: Pill },
    { id: 'caregiver', label: 'Caregiver', icon: ShieldCheck },
    ...(user?.role === 'admin' ? [{ id: 'admin', label: 'Admin', icon: Shield }] : []),
  ];

  return (
    <div className="min-h-screen bg-background">
      <Header 
        currentView={currentView} 
        setCurrentView={setCurrentView}
        darkMode={darkMode}
        setDarkMode={setDarkMode}
        user={user}
      />
      
      {currentView === 'home' && (
        <AICompanion 
          onNavigate={handleVoiceNavigate}
          userName={user?.name?.split(' ')[0] || 'Friend'}
          onRefreshData={loadAllData}
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
                <TabsList className="h-auto p-2 bg-muted/50 rounded-2xl my-4 flex flex-wrap justify-center gap-2">
                  {/* Home button to return to AI Companion */}
                  <button
                    onClick={() => setCurrentView('home')}
                    className="flex items-center gap-2 py-3 px-4 text-base font-semibold rounded-xl bg-accent text-accent-foreground hover:bg-accent/80 transition-all duration-300"
                  >
                    <Home className="h-5 w-5" />
                    <span className="hidden sm:inline">Assistant</span>
                  </button>
                  {tabItems.map((tab) => {
                    const Icon = tab.icon;
                    return (
                      <TabsTrigger
                        key={tab.id}
                        value={tab.id}
                        className="flex items-center gap-2 py-3 px-4 text-base font-semibold rounded-xl data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-card transition-all duration-300"
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
              <FamilySection 
                familyMembers={familyMembers} 
                onRefresh={refreshFamily}
                loading={loading}
              />
            </TabsContent>
            
            <TabsContent value="timeline" className="mt-0 animate-fade-in">
              <TimelineSection 
                memories={memories}
                familyMembers={familyMembers}
                onRefresh={refreshMemories}
                loading={loading}
              />
            </TabsContent>

            <TabsContent value="quiz" className="mt-0 animate-fade-in">
              <WhoIsThisQuiz
                familyMembers={familyMembers}
              />
            </TabsContent>

            <TabsContent value="cards" className="mt-0 animate-fade-in">
              <MemoryCardGame
                familyMembers={familyMembers}
              />
            </TabsContent>

            <TabsContent value="sudoku" className="mt-0 animate-fade-in">
              <SudokuGame />
            </TabsContent>

            <TabsContent value="week" className="mt-0 animate-fade-in">
              <WeekMemories 
                memories={memories}
                reminders={reminders}
                familyMembers={familyMembers}
              />
            </TabsContent>
            
            <TabsContent value="assistant" className="mt-0 animate-fade-in">
              <AssistantSection 
                userName={user?.name?.split(' ')[0] || 'Friend'}
              />
            </TabsContent>
            
            <TabsContent value="reminders" className="mt-0 animate-fade-in">
              <RemindersSection 
                reminders={reminders}
                onRefresh={refreshReminders}
                loading={loading}
              />
            </TabsContent>

            <TabsContent value="mood" className="mt-0 animate-fade-in">
              <MoodBehaviorSection />
            </TabsContent>

            <TabsContent value="navigation" className="mt-0 animate-fade-in">
              <NavigationSection
                destinations={destinations}
                onRefresh={refreshDestinations}
                loading={loading}
              />
            </TabsContent>

            <TabsContent value="medications" className="mt-0 animate-fade-in">
              <MedicationSection
                medications={medications}
                onRefresh={refreshMedications}
                loading={loading}
              />
            </TabsContent>

            <TabsContent value="caregiver" className="mt-0 animate-fade-in">
              <CaregiverPortalSection
                user={user}
                onNavigate={handleVoiceNavigate}
              />
            </TabsContent>

            <TabsContent value="admin" className="mt-0 animate-fade-in">
              <AdminSection
                user={user}
              />
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
};

export default DashboardPage;
