import React, { useState } from 'react';
import { Bell, Clock, Pill, Utensils, Heart, Sun, Moon, Check, Plus, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

// Get current time period
const getTimePeriod = () => {
  const hour = new Date().getHours();
  if (hour >= 5 && hour < 12) return 'morning';
  if (hour >= 12 && hour < 17) return 'afternoon';
  if (hour >= 17 && hour < 21) return 'evening';
  return 'night';
};

const getGreeting = () => {
  const period = getTimePeriod();
  const greetings = {
    morning: 'Good Morning',
    afternoon: 'Good Afternoon',
    evening: 'Good Evening',
    night: 'Good Night'
  };
  return greetings[period];
};

// Initial reminders data
const initialReminders = [
  {
    id: 1,
    title: 'Take morning medication',
    time: '8:00 AM',
    icon: Pill,
    category: 'health',
    color: 'bg-destructive/20 border-destructive text-destructive',
    completed: false,
    period: 'morning'
  },
  {
    id: 2,
    title: 'Drink a glass of water',
    time: '9:00 AM',
    icon: Heart,
    category: 'health',
    color: 'bg-family-children/20 border-family-children text-family-children',
    completed: false,
    period: 'morning'
  },
  {
    id: 3,
    title: 'Breakfast time',
    time: '9:30 AM',
    icon: Utensils,
    category: 'meals',
    color: 'bg-family-grandchildren/20 border-family-grandchildren text-foreground',
    completed: false,
    period: 'morning'
  },
  {
    id: 4,
    title: 'Lunch time',
    time: '12:30 PM',
    icon: Utensils,
    category: 'meals',
    color: 'bg-family-grandchildren/20 border-family-grandchildren text-foreground',
    completed: false,
    period: 'afternoon'
  },
  {
    id: 5,
    title: 'Take afternoon medication',
    time: '2:00 PM',
    icon: Pill,
    category: 'health',
    color: 'bg-destructive/20 border-destructive text-destructive',
    completed: false,
    period: 'afternoon'
  },
  {
    id: 6,
    title: 'Dinner time',
    time: '6:00 PM',
    icon: Utensils,
    category: 'meals',
    color: 'bg-family-grandchildren/20 border-family-grandchildren text-foreground',
    completed: false,
    period: 'evening'
  },
  {
    id: 7,
    title: 'Take evening medication',
    time: '8:00 PM',
    icon: Pill,
    category: 'health',
    color: 'bg-destructive/20 border-destructive text-destructive',
    completed: false,
    period: 'evening'
  },
  {
    id: 8,
    title: 'Drink water before bed',
    time: '9:00 PM',
    icon: Heart,
    category: 'health',
    color: 'bg-family-children/20 border-family-children text-family-children',
    completed: false,
    period: 'night'
  },
];

const periodIcons = {
  morning: Sun,
  afternoon: Sun,
  evening: Moon,
  night: Moon
};

const periodLabels = {
  morning: 'Morning',
  afternoon: 'Afternoon',
  evening: 'Evening',
  night: 'Night'
};

export const RemindersSection = () => {
  const [reminders, setReminders] = useState(initialReminders);
  const [filter, setFilter] = useState('all');
  const [newReminder, setNewReminder] = useState({ title: '', time: '', period: 'morning', category: 'health' });
  const [dialogOpen, setDialogOpen] = useState(false);

  const currentPeriod = getTimePeriod();
  
  const filteredReminders = filter === 'all' 
    ? reminders 
    : reminders.filter(r => r.period === filter);

  const completedCount = reminders.filter(r => r.completed).length;
  const totalCount = reminders.length;

  const toggleReminder = (id) => {
    setReminders(prev => 
      prev.map(r => r.id === id ? { ...r, completed: !r.completed } : r)
    );
  };

  const addReminder = () => {
    if (!newReminder.title || !newReminder.time) return;
    
    const iconMap = {
      health: Pill,
      meals: Utensils,
      activity: Heart
    };
    
    const colorMap = {
      health: 'bg-destructive/20 border-destructive text-destructive',
      meals: 'bg-family-grandchildren/20 border-family-grandchildren text-foreground',
      activity: 'bg-family-children/20 border-family-children text-family-children'
    };

    const reminder = {
      id: Date.now(),
      title: newReminder.title,
      time: newReminder.time,
      icon: iconMap[newReminder.category],
      category: newReminder.category,
      color: colorMap[newReminder.category],
      completed: false,
      period: newReminder.period
    };

    setReminders(prev => [...prev, reminder]);
    setNewReminder({ title: '', time: '', period: 'morning', category: 'health' });
    setDialogOpen(false);
  };

  const removeReminder = (id) => {
    setReminders(prev => prev.filter(r => r.id !== id));
  };

  // Group reminders by period
  const groupedReminders = {
    morning: filteredReminders.filter(r => r.period === 'morning'),
    afternoon: filteredReminders.filter(r => r.period === 'afternoon'),
    evening: filteredReminders.filter(r => r.period === 'evening'),
    night: filteredReminders.filter(r => r.period === 'night'),
  };

  const today = new Date().toLocaleDateString('en-US', { 
    weekday: 'long', 
    year: 'numeric', 
    month: 'long', 
    day: 'numeric' 
  });

  return (
    <section className="py-8 sm:py-12">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-4 py-2 mb-4">
            <Bell className="h-5 w-5 text-primary" />
            <span className="text-base font-medium text-primary">{today}</span>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl font-bold text-foreground mb-2">
            {getGreeting()}!
          </h2>
          <p className="text-accessible text-muted-foreground max-w-2xl mx-auto mb-6">
            Here's what you need to do today
          </p>
          
          {/* Progress */}
          <div className="max-w-md mx-auto">
            <div className="flex justify-between text-lg mb-2">
              <span className="font-semibold">Today's Progress</span>
              <span className="font-bold text-primary">{completedCount} / {totalCount}</span>
            </div>
            <div className="h-4 bg-muted rounded-full overflow-hidden">
              <div 
                className="h-full bg-primary rounded-full transition-all duration-500"
                style={{ width: `${(completedCount / totalCount) * 100}%` }}
              />
            </div>
          </div>
        </div>

        {/* Filter and Add */}
        <div className="flex flex-wrap justify-center gap-3 mb-8">
          <Button
            variant={filter === 'all' ? 'accessible' : 'accessible-outline'}
            onClick={() => setFilter('all')}
          >
            All Day
          </Button>
          <Button
            variant={filter === 'morning' ? 'accessible' : 'accessible-outline'}
            onClick={() => setFilter('morning')}
            className={currentPeriod === 'morning' ? 'ring-2 ring-accent ring-offset-2' : ''}
          >
            <Sun className="h-5 w-5 mr-2" />
            Morning
          </Button>
          <Button
            variant={filter === 'afternoon' ? 'accessible' : 'accessible-outline'}
            onClick={() => setFilter('afternoon')}
            className={currentPeriod === 'afternoon' ? 'ring-2 ring-accent ring-offset-2' : ''}
          >
            <Sun className="h-5 w-5 mr-2" />
            Afternoon
          </Button>
          <Button
            variant={filter === 'evening' ? 'accessible' : 'accessible-outline'}
            onClick={() => setFilter('evening')}
            className={currentPeriod === 'evening' ? 'ring-2 ring-accent ring-offset-2' : ''}
          >
            <Moon className="h-5 w-5 mr-2" />
            Evening
          </Button>
          
          <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
            <DialogTrigger asChild>
              <Button variant="accessible" className="gap-2">
                <Plus className="h-5 w-5" />
                Add Reminder
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-md">
              <DialogHeader>
                <DialogTitle className="text-2xl font-display">Add New Reminder</DialogTitle>
              </DialogHeader>
              <div className="space-y-6 mt-4">
                <div>
                  <Label htmlFor="title" className="text-lg">What do you need to remember?</Label>
                  <Input
                    id="title"
                    value={newReminder.title}
                    onChange={(e) => setNewReminder(prev => ({ ...prev, title: e.target.value }))}
                    placeholder="e.g., Take vitamins"
                    className="mt-2 h-14 text-lg"
                  />
                </div>
                <div>
                  <Label htmlFor="time" className="text-lg">What time?</Label>
                  <Input
                    id="time"
                    value={newReminder.time}
                    onChange={(e) => setNewReminder(prev => ({ ...prev, time: e.target.value }))}
                    placeholder="e.g., 10:00 AM"
                    className="mt-2 h-14 text-lg"
                  />
                </div>
                <div>
                  <Label className="text-lg">Time of Day</Label>
                  <Select 
                    value={newReminder.period} 
                    onValueChange={(value) => setNewReminder(prev => ({ ...prev, period: value }))}
                  >
                    <SelectTrigger className="mt-2 h-14 text-lg">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="morning" className="text-lg py-3">Morning</SelectItem>
                      <SelectItem value="afternoon" className="text-lg py-3">Afternoon</SelectItem>
                      <SelectItem value="evening" className="text-lg py-3">Evening</SelectItem>
                      <SelectItem value="night" className="text-lg py-3">Night</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label className="text-lg">Category</Label>
                  <Select 
                    value={newReminder.category} 
                    onValueChange={(value) => setNewReminder(prev => ({ ...prev, category: value }))}
                  >
                    <SelectTrigger className="mt-2 h-14 text-lg">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="health" className="text-lg py-3">Health / Medication</SelectItem>
                      <SelectItem value="meals" className="text-lg py-3">Meals</SelectItem>
                      <SelectItem value="activity" className="text-lg py-3">Activity</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <Button variant="accessible" className="w-full" onClick={addReminder}>
                  <Plus className="h-5 w-5 mr-2" />
                  Add Reminder
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>

        {/* Reminders by Period */}
        <div className="space-y-10">
          {Object.entries(groupedReminders).map(([period, periodReminders]) => {
            if (periodReminders.length === 0 && filter !== 'all') return null;
            if (periodReminders.length === 0) return null;
            
            const PeriodIcon = periodIcons[period];
            const isCurrentPeriod = period === currentPeriod;
            
            return (
              <div key={period} className={`${isCurrentPeriod ? 'ring-2 ring-accent ring-offset-4 rounded-3xl p-6 bg-accent/5' : ''}`}>
                <div className="flex items-center gap-3 mb-6">
                  <div className={`p-3 rounded-xl ${isCurrentPeriod ? 'bg-accent text-accent-foreground' : 'bg-muted'}`}>
                    <PeriodIcon className="h-6 w-6" />
                  </div>
                  <h3 className="text-2xl font-bold text-foreground">{periodLabels[period]}</h3>
                  {isCurrentPeriod && (
                    <Badge className="bg-accent text-accent-foreground">Now</Badge>
                  )}
                </div>
                
                <div className="grid gap-4">
                  {periodReminders.map((reminder, index) => {
                    const Icon = reminder.icon;
                    return (
                      <Card
                        key={reminder.id}
                        className={`border-2 ${reminder.color} ${reminder.completed ? 'opacity-60' : ''} shadow-soft hover:shadow-card transition-all duration-300 animate-scale-in`}
                        style={{ animationDelay: `${index * 0.1}s` }}
                      >
                        <CardContent className="p-6">
                          <div className="flex items-center gap-6">
                            {/* Checkbox */}
                            <Checkbox
                              checked={reminder.completed}
                              onCheckedChange={() => toggleReminder(reminder.id)}
                              className="h-8 w-8 rounded-lg border-2"
                            />
                            
                            {/* Icon */}
                            <div className={`p-3 rounded-xl bg-background shadow-soft`}>
                              <Icon className="h-7 w-7 text-primary" />
                            </div>
                            
                            {/* Content */}
                            <div className="flex-1">
                              <h4 className={`text-xl font-semibold ${reminder.completed ? 'line-through text-muted-foreground' : 'text-foreground'}`}>
                                {reminder.title}
                              </h4>
                              <div className="flex items-center gap-2 mt-1">
                                <Clock className="h-4 w-4 text-muted-foreground" />
                                <span className="text-base text-muted-foreground">{reminder.time}</span>
                              </div>
                            </div>
                            
                            {/* Completion badge */}
                            {reminder.completed && (
                              <Badge className="bg-success text-success-foreground gap-1">
                                <Check className="h-4 w-4" />
                                Done
                              </Badge>
                            )}
                            
                            {/* Remove button */}
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-10 w-10 text-muted-foreground hover:text-destructive"
                              onClick={() => removeReminder(reminder.id)}
                            >
                              <X className="h-5 w-5" />
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>

        {/* Encouragement message */}
        {completedCount === totalCount && totalCount > 0 && (
          <Card className="mt-10 border-2 border-success bg-success/10 shadow-card animate-scale-in">
            <CardContent className="p-8 text-center">
              <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-success flex items-center justify-center">
                <Check className="h-10 w-10 text-success-foreground" />
              </div>
              <h3 className="text-2xl font-bold text-foreground mb-2">Wonderful Job!</h3>
              <p className="text-accessible text-muted-foreground">
                You've completed all your tasks for today. You're doing great!
              </p>
            </CardContent>
          </Card>
        )}
      </div>
    </section>
  );
};

export default RemindersSection;
