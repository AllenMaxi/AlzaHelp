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
import { Users, Calendar, MessageCircle, Bell, Brain, CalendarDays, Home, Heart, Layers, Grid3X3, Route, Pill, ShieldCheck, Shield, Smile, Download, Trash2, CreditCard, Crown } from "lucide-react";
import { familyApi, memoriesApi, remindersApi, destinationsApi, medicationsApi, accountApi, billingApi } from "@/services/api";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { OnboardingWizard } from "@/components/OnboardingWizard";
import { InstallPrompt } from "@/components/InstallPrompt";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";

export const DashboardPage = () => {
  const { user, logout } = useAuth();
  const { t } = useTranslation();
  const [deletingAccount, setDeletingAccount] = useState(false);
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
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [subscriptionTier, setSubscriptionTier] = useState("free");

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
      // Load billing status
      billingApi.getStatus().then(s => setSubscriptionTier(s.tier || "free")).catch(() => {});
      // Show onboarding if first time and no data
      if (
        !localStorage.getItem("alzahelp_onboarded") &&
        (!familyData || familyData.length === 0) &&
        (!medicationsData || medicationsData.length === 0)
      ) {
        setShowOnboarding(true);
      }
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
    { id: 'family', label: t('dashboard.tabs.family'), icon: Users },
    { id: 'timeline', label: t('dashboard.tabs.timeline'), icon: Calendar },
    { id: 'quiz', label: t('dashboard.tabs.games'), icon: Brain },
    { id: 'cards', label: 'Match', icon: Layers },
    { id: 'sudoku', label: 'Sudoku', icon: Grid3X3 },
    { id: 'week', label: 'My Week', icon: CalendarDays },
    { id: 'assistant', label: t('dashboard.tabs.assistant'), icon: MessageCircle },
    { id: 'reminders', label: t('dashboard.tabs.reminders'), icon: Bell },
    { id: 'mood', label: t('dashboard.tabs.mood'), icon: Smile },
    { id: 'navigation', label: t('dashboard.tabs.navigation'), icon: Route },
    { id: 'medications', label: t('dashboard.tabs.medications'), icon: Pill },
    { id: 'caregiver', label: t('dashboard.tabs.caregiver'), icon: ShieldCheck },
    ...(user?.role === 'admin' ? [{ id: 'admin', label: t('dashboard.tabs.admin'), icon: Shield }] : []),
  ];

  return (
    <div className="min-h-screen bg-background">
      {showOnboarding && (
        <OnboardingWizard
          userName={user?.name?.split(" ")[0] || "Friend"}
          onComplete={() => { setShowOnboarding(false); loadAllData(); }}
        />
      )}
      {user?.is_demo && (
        <div className="bg-amber-50 dark:bg-amber-900/20 border-b border-amber-200 dark:border-amber-800 px-4 py-3">
          <div className="container mx-auto flex items-center justify-between gap-4">
            <p className="text-sm text-amber-800 dark:text-amber-200">
              {t('dashboard.demoBanner')} <strong>{t('dashboard.demoSignup')}</strong>
            </p>
            <Button size="sm" variant="outline" onClick={() => window.location.href = '/login'}>
              {t('common.signUp')}
            </Button>
          </div>
        </div>
      )}
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
              <ErrorBoundary level="section">
                <FamilySection
                  familyMembers={familyMembers}
                  onRefresh={refreshFamily}
                  loading={loading}
                />
              </ErrorBoundary>
            </TabsContent>

            <TabsContent value="timeline" className="mt-0 animate-fade-in">
              <ErrorBoundary level="section">
                <TimelineSection
                  memories={memories}
                  familyMembers={familyMembers}
                  onRefresh={refreshMemories}
                  loading={loading}
                />
              </ErrorBoundary>
            </TabsContent>

            <TabsContent value="quiz" className="mt-0 animate-fade-in">
              <ErrorBoundary level="section">
                <WhoIsThisQuiz familyMembers={familyMembers} />
              </ErrorBoundary>
            </TabsContent>

            <TabsContent value="cards" className="mt-0 animate-fade-in">
              <ErrorBoundary level="section">
                <MemoryCardGame familyMembers={familyMembers} />
              </ErrorBoundary>
            </TabsContent>

            <TabsContent value="sudoku" className="mt-0 animate-fade-in">
              <ErrorBoundary level="section">
                <SudokuGame />
              </ErrorBoundary>
            </TabsContent>

            <TabsContent value="week" className="mt-0 animate-fade-in">
              <ErrorBoundary level="section">
                <WeekMemories
                  memories={memories}
                  reminders={reminders}
                  familyMembers={familyMembers}
                />
              </ErrorBoundary>
            </TabsContent>

            <TabsContent value="assistant" className="mt-0 animate-fade-in">
              <ErrorBoundary level="section">
                <AssistantSection
                  userName={user?.name?.split(' ')[0] || 'Friend'}
                />
              </ErrorBoundary>
            </TabsContent>

            <TabsContent value="reminders" className="mt-0 animate-fade-in">
              <ErrorBoundary level="section">
                <RemindersSection
                  reminders={reminders}
                  onRefresh={refreshReminders}
                  loading={loading}
                />
              </ErrorBoundary>
            </TabsContent>

            <TabsContent value="mood" className="mt-0 animate-fade-in">
              <ErrorBoundary level="section">
                <MoodBehaviorSection />
              </ErrorBoundary>
            </TabsContent>

            <TabsContent value="navigation" className="mt-0 animate-fade-in">
              <ErrorBoundary level="section">
                <NavigationSection
                  destinations={destinations}
                  onRefresh={refreshDestinations}
                  loading={loading}
                />
              </ErrorBoundary>
            </TabsContent>

            <TabsContent value="medications" className="mt-0 animate-fade-in">
              <ErrorBoundary level="section">
                <MedicationSection
                  medications={medications}
                  onRefresh={refreshMedications}
                  loading={loading}
                />
              </ErrorBoundary>
            </TabsContent>

            <TabsContent value="caregiver" className="mt-0 animate-fade-in">
              <ErrorBoundary level="section">
                <CaregiverPortalSection
                  user={user}
                  onNavigate={handleVoiceNavigate}
                  subscriptionTier={subscriptionTier}
                />
              </ErrorBoundary>
            </TabsContent>

            <TabsContent value="admin" className="mt-0 animate-fade-in">
              <ErrorBoundary level="section">
                <AdminSection user={user} />
              </ErrorBoundary>
            </TabsContent>
          </Tabs>
        </main>
      )}

      {/* Subscription Status */}
      <section className="border-t border-border bg-gradient-to-r from-violet-500/5 to-transparent py-6">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <CreditCard className="h-5 w-5" />
            Subscription
          </h3>
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Current plan:</span>
              {subscriptionTier === "premium" ? (
                <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-300 text-sm font-semibold">
                  <Crown className="h-3.5 w-3.5" />
                  Premium
                </span>
              ) : (
                <span className="text-sm font-semibold text-foreground">Free</span>
              )}
            </div>
            {subscriptionTier === "premium" ? (
              <button
                onClick={async () => {
                  try {
                    const { url } = await billingApi.createPortal();
                    window.location.href = url;
                  } catch (e) {
                    alert("Could not open billing portal.");
                  }
                }}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-border bg-background hover:bg-muted transition-colors text-sm font-medium text-foreground"
              >
                Manage Subscription
              </button>
            ) : (
              <button
                onClick={async () => {
                  try {
                    const { url } = await billingApi.createCheckout();
                    window.location.href = url;
                  } catch (e) {
                    alert(e.message || "Could not start checkout.");
                  }
                }}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-700 transition-colors text-sm font-semibold text-white"
              >
                <Crown className="h-4 w-4" />
                Upgrade to Premium — $9.99/mo
              </button>
            )}
          </div>
          {subscriptionTier !== "premium" && (
            <p className="text-xs text-muted-foreground mt-3">
              Premium unlocks external bot access (Telegram/WhatsApp), SMS medication alerts, and more.
            </p>
          )}
        </div>
      </section>

      {/* Settings & Privacy */}
      <section className="border-t border-border bg-muted/20 py-6">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <h3 className="text-lg font-semibold text-foreground mb-4">Your Data & Privacy</h3>
          <div className="flex flex-wrap gap-3">
            <button
              onClick={async () => {
                try {
                  const data = await accountApi.exportData();
                  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url;
                  a.download = `alzahelp-export-${new Date().toISOString().slice(0, 10)}.json`;
                  a.click();
                  URL.revokeObjectURL(url);
                } catch (e) {
                  alert("Failed to export data. Please try again.");
                }
              }}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-border bg-background hover:bg-muted transition-colors text-sm font-medium text-foreground"
            >
              <Download className="h-4 w-4" />
              Export My Data
            </button>
            <button
              onClick={async () => {
                if (deletingAccount) return;
                const confirmed = window.confirm(
                  "Are you sure you want to permanently delete your account and all data? This action cannot be undone."
                );
                if (!confirmed) return;
                const doubleConfirmed = window.confirm(
                  "This will delete ALL your memories, family members, medications, and settings. Type OK to proceed."
                );
                if (!doubleConfirmed) return;
                setDeletingAccount(true);
                try {
                  await accountApi.deleteAccount();
                  localStorage.clear();
                  logout();
                } catch (e) {
                  alert("Failed to delete account. Please try again.");
                  setDeletingAccount(false);
                }
              }}
              disabled={deletingAccount}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-red-300 dark:border-red-800 bg-background hover:bg-red-50 dark:hover:bg-red-950 transition-colors text-sm font-medium text-red-600 dark:text-red-400 disabled:opacity-50"
            >
              <Trash2 className="h-4 w-4" />
              {deletingAccount ? "Deleting..." : "Delete Account"}
            </button>
          </div>
          <p className="text-xs text-muted-foreground mt-3">
            Your data is stored securely. You can export a copy or permanently delete your account at any time.
          </p>
        </div>
      </section>

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
      <InstallPrompt />
    </div>
  );
};

export default DashboardPage;
