import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Mic, MicOff, Volume2, VolumeX, Users, Calendar, Brain, Bell, CalendarDays, MessageCircle, Home } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const WAKE_WORD = "hey mate";
const AVATAR_URL = "https://static.prod-images.emergentagent.com/jobs/7e703976-b1b5-4cac-8bb4-5016cc00ab1e/images/8861c004f3887363a550e65273e509f4b2a83fa5a022f2e61f3c6318cf99be61.png";

// Conversation states for multi-turn dialogs
const CONVERSATION_STATES = {
  IDLE: 'idle',
  LISTENING_WAKE: 'listening_wake',
  LISTENING_COMMAND: 'listening_command',
  PROCESSING: 'processing',
  SPEAKING: 'speaking',
  // Multi-turn states for creating data
  CREATING_MEMORY: 'creating_memory',
  CREATING_FAMILY: 'creating_family',
  CREATING_REMINDER: 'creating_reminder',
};

export const AICompanion = ({ onNavigate, userName = "Friend", onRefreshData }) => {
  const [conversationState, setConversationState] = useState(CONVERSATION_STATES.IDLE);
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [displayText, setDisplayText] = useState('');
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [sessionId] = useState(() => `companion_${Date.now()}`);
  
  // Multi-turn conversation data
  const [pendingData, setPendingData] = useState({});
  const [currentField, setCurrentField] = useState(null);
  
  const recognitionRef = useRef(null);
  const audioRef = useRef(null);
  const silenceTimeoutRef = useRef(null);

  // Field prompts for creating data
  const FIELD_PROMPTS = {
    memory: {
      title: "What would you like to call this memory?",
      date: "When did this happen? You can say something like 'last week' or 'June 2020'.",
      location: "Where did this take place?",
      description: "Tell me more about this memory. What happened?",
      people: "Who was there with you?",
      confirm: "I have all the details. Should I save this memory?"
    },
    family: {
      name: "What is their name?",
      relationship: "How are they related to you? For example, son, daughter, spouse, friend.",
      notes: "Tell me something special about them that you'd like to remember.",
      confirm: "Should I add this person to your family?"
    },
    reminder: {
      title: "What should I remind you about?",
      time: "What time should I remind you? You can say morning, afternoon, or evening.",
      confirm: "Should I set this reminder?"
    }
  };

  // Initialize speech recognition
  useEffect(() => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      console.warn('Speech recognition not supported');
      return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onresult = (event) => {
      let finalTranscript = '';
      let interimTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcriptText = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcriptText;
        } else {
          interimTranscript += transcriptText;
        }
      }

      const currentTranscript = (finalTranscript || interimTranscript).toLowerCase();
      
      // Check for wake word when idle
      if (conversationState === CONVERSATION_STATES.LISTENING_WAKE) {
        if (currentTranscript.includes(WAKE_WORD)) {
          setConversationState(CONVERSATION_STATES.LISTENING_COMMAND);
          setTranscript('');
          speak("Yes, I'm here. How can I help you today?");
          return;
        }
        setTranscript(interimTranscript);
      }
      // Capture command when actively listening
      else if (conversationState === CONVERSATION_STATES.LISTENING_COMMAND && finalTranscript) {
        const command = finalTranscript.replace(/hey mate/gi, '').trim();
        if (command.length > 2) {
          setTranscript(command);
          processCommand(command);
        }
      }
      // Multi-turn conversation - capture field values
      else if ([CONVERSATION_STATES.CREATING_MEMORY, CONVERSATION_STATES.CREATING_FAMILY, CONVERSATION_STATES.CREATING_REMINDER].includes(conversationState)) {
        if (finalTranscript) {
          const value = finalTranscript.trim();
          if (value.length > 1) {
            setTranscript(value);
            handleFieldResponse(value);
          }
        } else {
          setTranscript(interimTranscript);
        }
      }

      // Reset silence timeout
      if (silenceTimeoutRef.current) {
        clearTimeout(silenceTimeoutRef.current);
      }
      if (conversationState === CONVERSATION_STATES.LISTENING_COMMAND) {
        silenceTimeoutRef.current = setTimeout(() => {
          if (conversationState === CONVERSATION_STATES.LISTENING_COMMAND) {
            speak("I didn't hear anything. Say 'Hey Mate' when you need me.");
            setConversationState(CONVERSATION_STATES.LISTENING_WAKE);
            setTranscript('');
          }
        }, 8000);
      }
    };

    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      if (event.error === 'no-speech') {
        // Silently continue
      } else if (event.error !== 'aborted') {
        setIsListening(false);
      }
    };

    recognition.onend = () => {
      if (isListening && conversationState !== CONVERSATION_STATES.SPEAKING) {
        try {
          recognition.start();
        } catch (e) {
          console.error('Failed to restart recognition:', e);
        }
      }
    };

    recognitionRef.current = recognition;

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      if (silenceTimeoutRef.current) {
        clearTimeout(silenceTimeoutRef.current);
      }
    };
  }, [conversationState, isListening]);

  // Auto-start listening on mount
  useEffect(() => {
    const timer = setTimeout(() => {
      startListening();
      speak(`Hello ${userName}! I'm your memory companion. Say "Hey Mate" anytime you need help remembering something, adding a memory, or just want to chat.`);
    }, 1000);
    return () => clearTimeout(timer);
  }, [userName]);

  const startListening = useCallback(() => {
    try {
      recognitionRef.current?.start();
      setIsListening(true);
      setConversationState(CONVERSATION_STATES.LISTENING_WAKE);
    } catch (e) {
      console.error('Failed to start recognition:', e);
    }
  }, []);

  const stopListening = useCallback(() => {
    recognitionRef.current?.stop();
    setIsListening(false);
    setConversationState(CONVERSATION_STATES.IDLE);
    setTranscript('');
  }, []);

  // Text-to-speech using OpenAI TTS
  const speak = async (text) => {
    if (!text) return;
    
    const previousState = conversationState;
    setConversationState(CONVERSATION_STATES.SPEAKING);
    setDisplayText(text);

    // Pause recognition while speaking
    recognitionRef.current?.stop();

    if (!audioEnabled) {
      // Just display the text, then resume
      setTimeout(() => {
        resumeAfterSpeaking(previousState);
      }, 2000);
      return;
    }

    try {
      const res = await fetch(`${BACKEND_URL}/api/tts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ text, voice: 'nova' })
      });

      if (res.ok) {
        const data = await res.json();
        const audio = new Audio(`data:audio/mp3;base64,${data.audio}`);
        audioRef.current = audio;
        
        audio.onended = () => {
          resumeAfterSpeaking(previousState);
        };
        
        audio.onerror = () => {
          fallbackSpeak(text, previousState);
        };
        
        await audio.play();
      } else {
        fallbackSpeak(text, previousState);
      }
    } catch (error) {
      console.error('TTS error:', error);
      fallbackSpeak(text, previousState);
    }
  };

  const fallbackSpeak = (text, previousState) => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.85;
      utterance.onend = () => resumeAfterSpeaking(previousState);
      window.speechSynthesis.speak(utterance);
    } else {
      setTimeout(() => resumeAfterSpeaking(previousState), 2000);
    }
  };

  const resumeAfterSpeaking = (previousState) => {
    setConversationState(
      [CONVERSATION_STATES.CREATING_MEMORY, CONVERSATION_STATES.CREATING_FAMILY, CONVERSATION_STATES.CREATING_REMINDER].includes(previousState)
        ? previousState
        : CONVERSATION_STATES.LISTENING_COMMAND
    );
    if (isListening) {
      try {
        recognitionRef.current?.start();
      } catch (e) {
        console.error('Failed to resume recognition:', e);
      }
    }
  };

  // Process voice command
  const processCommand = async (command) => {
    if (!command.trim()) return;
    
    setConversationState(CONVERSATION_STATES.PROCESSING);
    const lowerCommand = command.toLowerCase();

    // Check for creation intents
    if (lowerCommand.includes('add') || lowerCommand.includes('create') || lowerCommand.includes('new') || lowerCommand.includes('remember')) {
      if (lowerCommand.includes('memory') || lowerCommand.includes('remember')) {
        startCreatingMemory();
        return;
      }
      if (lowerCommand.includes('family') || lowerCommand.includes('person') || lowerCommand.includes('someone')) {
        startCreatingFamily();
        return;
      }
      if (lowerCommand.includes('reminder') || lowerCommand.includes('remind')) {
        startCreatingReminder();
        return;
      }
    }

    // Navigation commands
    const navMap = {
      'family': ['family', 'relatives', 'loved ones'],
      'timeline': ['memories', 'memory', 'timeline', 'remember'],
      'quiz': ['quiz', 'game', 'practice', 'faces'],
      'week': ['week', 'yesterday', 'today', 'recent', 'what did i do'],
      'assistant': ['chat', 'talk', 'help'],
      'reminders': ['reminders', 'schedule', 'tasks'],
      'home': ['home', 'start', 'main']
    };

    for (const [target, keywords] of Object.entries(navMap)) {
      if (keywords.some(kw => lowerCommand.includes(kw))) {
        const responses = {
          family: "Let me show you your family.",
          timeline: "Here are your precious memories.",
          quiz: "Let's practice remembering faces together!",
          week: "Let me show you what you did recently.",
          assistant: "I'm right here to help you.",
          reminders: "Here are your reminders for today.",
          home: "Taking you back home."
        };
        speak(responses[target] || `Going to ${target}.`);
        setTimeout(() => onNavigate(target), 1500);
        return;
      }
    }

    // For questions, use the backend RAG
    try {
      const res = await fetch(`${BACKEND_URL}/api/voice-command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ text: command, session_id: sessionId })
      });

      if (res.ok) {
        const data = await res.json();
        speak(data.response);
        
        if (data.action === 'navigate' && onNavigate) {
          setTimeout(() => onNavigate(data.target), 2000);
        }
      } else {
        speak("I'm not sure about that. Could you ask me differently?");
      }
    } catch (error) {
      speak("I'm having trouble understanding. Please try again.");
    }
  };

  // Start multi-turn memory creation
  const startCreatingMemory = () => {
    setPendingData({});
    setCurrentField('title');
    setConversationState(CONVERSATION_STATES.CREATING_MEMORY);
    speak(FIELD_PROMPTS.memory.title);
  };

  const startCreatingFamily = () => {
    setPendingData({});
    setCurrentField('name');
    setConversationState(CONVERSATION_STATES.CREATING_FAMILY);
    speak(FIELD_PROMPTS.family.name);
  };

  const startCreatingReminder = () => {
    setPendingData({});
    setCurrentField('title');
    setConversationState(CONVERSATION_STATES.CREATING_REMINDER);
    speak(FIELD_PROMPTS.reminder.title);
  };

  // Handle field response in multi-turn conversation
  const handleFieldResponse = async (value) => {
    const lowerValue = value.toLowerCase();
    
    // Check for cancellation
    if (lowerValue.includes('cancel') || lowerValue.includes('stop') || lowerValue.includes('never mind')) {
      speak("No problem, I've cancelled that. Let me know if you need anything else.");
      setPendingData({});
      setCurrentField(null);
      setConversationState(CONVERSATION_STATES.LISTENING_COMMAND);
      return;
    }

    // Handle confirmation
    if (currentField === 'confirm') {
      if (lowerValue.includes('yes') || lowerValue.includes('sure') || lowerValue.includes('save') || lowerValue.includes('ok')) {
        await saveData();
      } else if (lowerValue.includes('no') || lowerValue.includes('cancel')) {
        speak("Alright, I won't save that. Let me know if you want to try again.");
        setPendingData({});
        setCurrentField(null);
        setConversationState(CONVERSATION_STATES.LISTENING_COMMAND);
      } else {
        speak("Please say yes to save, or no to cancel.");
      }
      return;
    }

    // Store the field value and move to next
    const newData = { ...pendingData, [currentField]: value };
    setPendingData(newData);

    // Determine next field based on conversation state
    let fields, prompts;
    if (conversationState === CONVERSATION_STATES.CREATING_MEMORY) {
      fields = ['title', 'date', 'location', 'description', 'people', 'confirm'];
      prompts = FIELD_PROMPTS.memory;
    } else if (conversationState === CONVERSATION_STATES.CREATING_FAMILY) {
      fields = ['name', 'relationship', 'notes', 'confirm'];
      prompts = FIELD_PROMPTS.family;
    } else if (conversationState === CONVERSATION_STATES.CREATING_REMINDER) {
      fields = ['title', 'time', 'confirm'];
      prompts = FIELD_PROMPTS.reminder;
    }

    const currentIndex = fields.indexOf(currentField);
    const nextField = fields[currentIndex + 1];

    if (nextField) {
      setCurrentField(nextField);
      
      // For confirm, summarize what we have
      if (nextField === 'confirm') {
        let summary = "";
        if (conversationState === CONVERSATION_STATES.CREATING_MEMORY) {
          summary = `Great! Here's the memory: "${newData.title}" from ${newData.date} at ${newData.location}. ${newData.description}. `;
        } else if (conversationState === CONVERSATION_STATES.CREATING_FAMILY) {
          summary = `Got it! ${newData.name}, your ${newData.relationship}. ${newData.notes}. `;
        } else if (conversationState === CONVERSATION_STATES.CREATING_REMINDER) {
          summary = `Okay, I'll remind you: "${newData.title}" in the ${newData.time}. `;
        }
        speak(summary + prompts[nextField]);
      } else {
        speak(`Got it! ${prompts[nextField]}`);
      }
    }
    
    setTranscript('');
  };

  // Save the collected data
  const saveData = async () => {
    setConversationState(CONVERSATION_STATES.PROCESSING);
    
    try {
      let endpoint, body;
      
      if (conversationState === CONVERSATION_STATES.CREATING_MEMORY) {
        endpoint = '/api/memories';
        body = {
          title: pendingData.title,
          date: pendingData.date,
          location: pendingData.location || '',
          description: pendingData.description || '',
          people: pendingData.people ? pendingData.people.split(/,|and/).map(p => p.trim()) : [],
          photos: []
        };
      } else if (conversationState === CONVERSATION_STATES.CREATING_FAMILY) {
        endpoint = '/api/family';
        body = {
          name: pendingData.name,
          relationship: pendingData.relationship.toLowerCase(),
          relationship_label: `Your ${pendingData.relationship}`,
          notes: pendingData.notes || '',
          photos: []
        };
      } else if (conversationState === CONVERSATION_STATES.CREATING_REMINDER) {
        endpoint = '/api/reminders';
        const timeMap = { morning: '09:00', afternoon: '14:00', evening: '18:00', night: '20:00' };
        const time = Object.entries(timeMap).find(([k]) => pendingData.time?.toLowerCase().includes(k))?.[1] || '12:00';
        body = {
          title: pendingData.title,
          time: time
        };
      }

      const res = await fetch(`${BACKEND_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(body)
      });

      if (res.ok) {
        speak("Done! I've saved that for you. Is there anything else I can help with?");
        if (onRefreshData) onRefreshData();
        toast.success('Saved successfully!');
      } else {
        speak("I had trouble saving that. Would you like to try again?");
      }
    } catch (error) {
      console.error('Save error:', error);
      speak("Something went wrong. Please try again later.");
    }

    setPendingData({});
    setCurrentField(null);
    setConversationState(CONVERSATION_STATES.LISTENING_COMMAND);
  };

  // Get status message based on state
  const getStatusMessage = () => {
    switch (conversationState) {
      case CONVERSATION_STATES.IDLE:
        return "Tap to start";
      case CONVERSATION_STATES.LISTENING_WAKE:
        return 'Say "Hey Mate" to talk to me';
      case CONVERSATION_STATES.LISTENING_COMMAND:
        return transcript || "I'm listening...";
      case CONVERSATION_STATES.PROCESSING:
        return "Let me think...";
      case CONVERSATION_STATES.SPEAKING:
        return displayText;
      case CONVERSATION_STATES.CREATING_MEMORY:
      case CONVERSATION_STATES.CREATING_FAMILY:
      case CONVERSATION_STATES.CREATING_REMINDER:
        return transcript || "I'm listening...";
      default:
        return "";
    }
  };

  const quickActions = [
    { id: 'family', label: 'Family', icon: Users },
    { id: 'timeline', label: 'Memories', icon: Calendar },
    { id: 'quiz', label: 'Quiz', icon: Brain },
    { id: 'week', label: 'My Week', icon: CalendarDays },
    { id: 'reminders', label: 'Today', icon: Bell },
  ];

  return (
    <div className="min-h-[80vh] flex flex-col items-center justify-center px-4 py-8">
      {/* Avatar Section */}
      <div className="relative mb-8">
        {/* Animated rings when listening/speaking */}
        {(conversationState === CONVERSATION_STATES.LISTENING_COMMAND || 
          conversationState === CONVERSATION_STATES.CREATING_MEMORY ||
          conversationState === CONVERSATION_STATES.CREATING_FAMILY ||
          conversationState === CONVERSATION_STATES.CREATING_REMINDER) && (
          <>
            <div className="absolute inset-0 rounded-full bg-success/20 animate-ping" style={{ animationDuration: '2s' }} />
            <div className="absolute inset-[-10px] rounded-full border-4 border-success/40 animate-pulse" />
          </>
        )}
        {conversationState === CONVERSATION_STATES.SPEAKING && (
          <>
            <div className="absolute inset-0 rounded-full bg-primary/20 animate-ping" style={{ animationDuration: '1s' }} />
            <div className="absolute inset-[-10px] rounded-full border-4 border-primary/40 animate-pulse" />
          </>
        )}
        {conversationState === CONVERSATION_STATES.LISTENING_WAKE && (
          <div className="absolute inset-[-5px] rounded-full border-2 border-muted-foreground/30 animate-pulse" />
        )}
        
        {/* Avatar Image */}
        <div 
          className={`
            relative w-48 h-48 md:w-56 md:h-56 rounded-full overflow-hidden
            border-4 shadow-elevated cursor-pointer transition-all duration-300
            ${conversationState === CONVERSATION_STATES.SPEAKING ? 'border-primary scale-105' : ''}
            ${conversationState === CONVERSATION_STATES.LISTENING_COMMAND ? 'border-success scale-105' : ''}
            ${conversationState === CONVERSATION_STATES.LISTENING_WAKE ? 'border-muted hover:border-primary' : ''}
            ${conversationState === CONVERSATION_STATES.IDLE ? 'border-muted hover:border-primary' : ''}
          `}
          onClick={() => {
            if (!isListening) {
              startListening();
            } else if (conversationState === CONVERSATION_STATES.LISTENING_WAKE) {
              setConversationState(CONVERSATION_STATES.LISTENING_COMMAND);
              speak("Yes, I'm here. How can I help you?");
            }
          }}
        >
          <img 
            src={AVATAR_URL} 
            alt="Memory Companion"
            className="w-full h-full object-cover"
          />
          
          {/* Microphone indicator */}
          {isListening && (
            <div className="absolute bottom-2 right-2 bg-background rounded-full p-2 shadow-lg">
              <Mic className={`h-5 w-5 ${conversationState === CONVERSATION_STATES.LISTENING_COMMAND ? 'text-success animate-pulse' : 'text-muted-foreground'}`} />
            </div>
          )}
        </div>
      </div>

      {/* Status Message */}
      <div className="text-center mb-8 max-w-md">
        <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-3">
          {conversationState === CONVERSATION_STATES.SPEAKING ? "I'm speaking..." :
           conversationState === CONVERSATION_STATES.LISTENING_COMMAND ? "I'm listening..." :
           conversationState === CONVERSATION_STATES.PROCESSING ? "Thinking..." :
           conversationState === CONVERSATION_STATES.CREATING_MEMORY ? "Creating memory..." :
           conversationState === CONVERSATION_STATES.CREATING_FAMILY ? "Adding family..." :
           conversationState === CONVERSATION_STATES.CREATING_REMINDER ? "Setting reminder..." :
           `Hello, ${userName}!`}
        </h2>
        <p className={`text-lg md:text-xl ${
          conversationState === CONVERSATION_STATES.SPEAKING ? 'text-primary' :
          conversationState === CONVERSATION_STATES.LISTENING_COMMAND ? 'text-success' :
          'text-muted-foreground'
        }`}>
          {getStatusMessage()}
        </p>
      </div>

      {/* Control Buttons */}
      <div className="flex items-center gap-4 mb-10">
        <Button
          variant={isListening ? "destructive" : "accessible"}
          size="lg"
          className="gap-2 text-lg px-8 py-6"
          onClick={() => isListening ? stopListening() : startListening()}
        >
          {isListening ? (
            <>
              <MicOff className="h-6 w-6" />
              Stop
            </>
          ) : (
            <>
              <Mic className="h-6 w-6" />
              Start Listening
            </>
          )}
        </Button>
        
        <Button
          variant="outline"
          size="lg"
          className="px-6 py-6"
          onClick={() => setAudioEnabled(!audioEnabled)}
        >
          {audioEnabled ? <Volume2 className="h-6 w-6" /> : <VolumeX className="h-6 w-6" />}
        </Button>
      </div>

      {/* Quick Actions - Backup buttons */}
      <div className="w-full max-w-2xl">
        <p className="text-center text-sm text-muted-foreground mb-4">Or tap to go directly:</p>
        <div className="flex flex-wrap justify-center gap-3">
          {quickActions.map((action) => (
            <Button
              key={action.id}
              variant="outline"
              size="lg"
              className="gap-2 px-6"
              onClick={() => onNavigate(action.id)}
            >
              <action.icon className="h-5 w-5" />
              {action.label}
            </Button>
          ))}
        </div>
      </div>

      {/* Suggested Commands */}
      <div className="mt-10 text-center">
        <p className="text-sm text-muted-foreground mb-3">Try saying:</p>
        <div className="flex flex-wrap justify-center gap-2">
          {[
            "Add a new memory",
            "Who is in my family?",
            "What did I do yesterday?",
            "Add a family member",
            "Set a reminder"
          ].map((suggestion) => (
            <span 
              key={suggestion}
              className="bg-muted px-3 py-1 rounded-full text-sm text-foreground cursor-pointer hover:bg-muted/80 transition-colors"
              onClick={() => processCommand(suggestion)}
            >
              "{suggestion}"
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};

export default AICompanion;
