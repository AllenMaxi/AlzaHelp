import React, { useState, useEffect } from 'react';
import { CalendarDays, ChevronLeft, ChevronRight, Clock, MapPin, Users, Heart, Pill, Utensils, Check, Camera, MessageCircle, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

// Days of the week
const dayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
const monthNames = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

const getRelativeDay = (daysAgo) => {
  if (daysAgo === 0) return 'Today';
  if (daysAgo === 1) return 'Yesterday';
  if (daysAgo === 2) return 'Two Days Ago';
  return `${daysAgo} Days Ago`;
};

const formatDate = (date) => {
  return `${dayNames[date.getDay()]}, ${monthNames[date.getMonth()]} ${date.getDate()}`;
};

export const WeekMemories = ({ memories = [], reminders = [], familyMembers = [] }) => {
  const [selectedDay, setSelectedDay] = useState(0); // 0 = today, 1 = yesterday, etc.
  const [weekDays, setWeekDays] = useState([]);
  const [dailyNotes, setDailyNotes] = useState({});
  const [newNote, setNewNote] = useState('');
  const [savingNote, setSavingNote] = useState(false);
  const [loadingNotes, setLoadingNotes] = useState(true);

  // Generate week days
  useEffect(() => {
    const days = [];
    const today = new Date();
    
    for (let i = 0; i < 7; i++) {
      const date = new Date(today);
      date.setDate(today.getDate() - i);
      days.push({
        index: i,
        date: date,
        label: getRelativeDay(i),
        formattedDate: formatDate(date),
        dateString: date.toISOString().split('T')[0]
      });
    }
    
    setWeekDays(days);
  }, []);

  // Load daily notes from backend
  useEffect(() => {
    loadDailyNotes();
  }, []);

  const loadDailyNotes = async () => {
    setLoadingNotes(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/daily-notes`, {
        credentials: 'include'
      });
      if (response.ok) {
        const notes = await response.json();
        const notesMap = {};
        notes.forEach(note => {
          notesMap[note.date] = note;
        });
        setDailyNotes(notesMap);
      }
    } catch (error) {
      console.error('Error loading daily notes:', error);
    } finally {
      setLoadingNotes(false);
    }
  };

  const saveDailyNote = async () => {
    if (!newNote.trim()) return;
    
    const selectedDate = weekDays[selectedDay]?.dateString;
    if (!selectedDate) return;
    
    setSavingNote(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/daily-notes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          date: selectedDate,
          note: newNote.trim()
        })
      });
      
      if (response.ok) {
        const savedNote = await response.json();
        setDailyNotes(prev => ({
          ...prev,
          [selectedDate]: savedNote
        }));
        setNewNote('');
        toast.success('Note saved!');
      }
    } catch (error) {
      toast.error('Could not save note');
    } finally {
      setSavingNote(false);
    }
  };

  // Get activities for a specific day
  const getActivitiesForDay = (dateString) => {
    const activities = [];
    
    // Check reminders completed on this day
    reminders.forEach(reminder => {
      if (reminder.completed) {
        activities.push({
          type: 'reminder',
          icon: reminder.category === 'health' ? Pill : reminder.category === 'meals' ? Utensils : Heart,
          title: reminder.title,
          time: reminder.time,
          color: 'bg-success/20 text-success'
        });
      }
    });
    
    // Check if any memories were created on this day
    memories.forEach(memory => {
      const memoryDate = memory.created_at?.split('T')[0];
      if (memoryDate === dateString) {
        activities.push({
          type: 'memory_added',
          icon: Camera,
          title: `Added memory: ${memory.title}`,
          description: memory.description?.substring(0, 100) + '...',
          color: 'bg-primary/20 text-primary'
        });
      }
    });
    
    return activities;
  };

  // Get memories that happened on this date in history
  const getMemoriesOnThisDate = (date) => {
    const month = date.getMonth();
    const day = date.getDate();
    
    return memories.filter(memory => {
      // Try to parse the date from the memory
      const memoryDateStr = memory.date;
      if (!memoryDateStr) return false;
      
      // Check if the month/day matches (for "On this day" feature)
      const monthNames = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'];
      const lowerDate = memoryDateStr.toLowerCase();
      
      const memoryMonth = monthNames.findIndex(m => lowerDate.includes(m));
      const dayMatch = lowerDate.match(/\d+/);
      const memoryDay = dayMatch ? parseInt(dayMatch[0]) : null;
      
      return memoryMonth === month && memoryDay === day;
    });
  };

  const selectedDayData = weekDays[selectedDay];
  const dayActivities = selectedDayData ? getActivitiesForDay(selectedDayData.dateString) : [];
  const onThisDayMemories = selectedDayData ? getMemoriesOnThisDate(selectedDayData.date) : [];
  const dayNote = selectedDayData ? dailyNotes[selectedDayData.dateString] : null;

  return (
    <section className="py-8 sm:py-12">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-4 py-2 mb-4">
            <CalendarDays className="h-5 w-5 text-primary" />
            <span className="text-base font-medium text-primary">Your Recent Days</span>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl font-bold text-foreground mb-4">
            Week Memories
          </h2>
          <p className="text-accessible text-muted-foreground max-w-2xl mx-auto">
            See what you did recently and add notes about your day
          </p>
        </div>

        {/* Day Selector */}
        <div className="flex flex-wrap justify-center gap-2 mb-8">
          {weekDays.map((day) => (
            <Button
              key={day.index}
              variant={selectedDay === day.index ? 'accessible' : 'accessible-outline'}
              onClick={() => setSelectedDay(day.index)}
              className="flex-col h-auto py-3 px-5"
            >
              <span className="text-lg font-bold">{day.label}</span>
              <span className="text-sm opacity-80">{day.formattedDate}</span>
            </Button>
          ))}
        </div>

        {selectedDayData && (
          <div className="max-w-3xl mx-auto space-y-6">
            {/* Day Header */}
            <Card className="border-2 border-primary bg-primary/5">
              <CardContent className="p-6 text-center">
                <h3 className="text-2xl font-bold text-foreground">
                  {selectedDayData.label}
                </h3>
                <p className="text-lg text-muted-foreground">
                  {selectedDayData.formattedDate}
                </p>
              </CardContent>
            </Card>

            {/* On This Day in History */}
            {onThisDayMemories.length > 0 && (
              <Card className="border-2 border-accent bg-accent/10">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-accent-foreground">
                    <Heart className="h-6 w-6" fill="currentColor" />
                    On This Day...
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {onThisDayMemories.map((memory, index) => (
                    <div key={memory.id} className={`${index > 0 ? 'mt-4 pt-4 border-t border-border' : ''}`}>
                      <p className="text-lg font-semibold">{memory.title}</p>
                      <p className="text-base text-muted-foreground">{memory.date} ({memory.year})</p>
                      {memory.location && (
                        <p className="text-sm text-muted-foreground flex items-center gap-1 mt-1">
                          <MapPin className="h-4 w-4" />
                          {memory.location}
                        </p>
                      )}
                    </div>
                  ))}
                </CardContent>
              </Card>
            )}

            {/* Daily Activities */}
            <Card className="border-2 border-border">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Check className="h-6 w-6 text-success" />
                  What Happened {selectedDayData.label}
                </CardTitle>
              </CardHeader>
              <CardContent>
                {dayActivities.length > 0 ? (
                  <div className="space-y-3">
                    {dayActivities.map((activity, index) => {
                      const Icon = activity.icon;
                      return (
                        <div 
                          key={index}
                          className={`flex items-start gap-3 p-3 rounded-lg ${activity.color}`}
                        >
                          <Icon className="h-5 w-5 mt-0.5" />
                          <div>
                            <p className="font-medium">{activity.title}</p>
                            {activity.time && (
                              <p className="text-sm opacity-80">{activity.time}</p>
                            )}
                            {activity.description && (
                              <p className="text-sm opacity-80">{activity.description}</p>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <p className="text-muted-foreground text-center py-4">
                    {selectedDay === 0 
                      ? "Your activities will appear here as you complete them."
                      : "No recorded activities for this day."}
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Daily Notes */}
            <Card className="border-2 border-border">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MessageCircle className="h-6 w-6 text-primary" />
                  Notes for {selectedDayData.label}
                </CardTitle>
              </CardHeader>
              <CardContent>
                {loadingNotes ? (
                  <div className="flex justify-center py-4">
                    <Loader2 className="h-6 w-6 animate-spin text-primary" />
                  </div>
                ) : (
                  <>
                    {dayNote && (
                      <div className="p-4 bg-muted rounded-xl mb-4">
                        <p className="text-accessible">{dayNote.note}</p>
                        <p className="text-sm text-muted-foreground mt-2">
                          Saved at {new Date(dayNote.created_at).toLocaleTimeString()}
                        </p>
                      </div>
                    )}
                    
                    <div className="space-y-3">
                      <Textarea
                        value={newNote}
                        onChange={(e) => setNewNote(e.target.value)}
                        placeholder={`Write about ${selectedDayData.label.toLowerCase()}... What did you do? How did you feel? Who did you see?`}
                        className="min-h-[100px] text-lg"
                      />
                      <Button
                        variant="accessible"
                        onClick={saveDailyNote}
                        disabled={!newNote.trim() || savingNote}
                        className="w-full"
                      >
                        {savingNote ? (
                          <><Loader2 className="h-5 w-5 mr-2 animate-spin" /> Saving...</>
                        ) : (
                          <>Save Note</>
                        )}
                      </Button>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>

            {/* Family Members Seen */}
            {familyMembers.length > 0 && selectedDay === 0 && (
              <Card className="border-2 border-border">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Users className="h-6 w-6 text-family-children" />
                    Did You See Anyone Today?
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground mb-4">
                    Tap anyone you saw or spoke with today:
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {familyMembers.slice(0, 8).map(member => (
                      <Button
                        key={member.id}
                        variant="outline"
                        className="h-auto py-2 px-4"
                        onClick={() => {
                          setNewNote(prev => {
                            const text = prev ? `${prev} I saw ${member.name} today.` : `I saw ${member.name} today.`;
                            return text;
                          });
                          toast.success(`Added ${member.name} to your note`);
                        }}
                      >
                        {member.name}
                      </Button>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        )}
      </div>
    </section>
  );
};

export default WeekMemories;
