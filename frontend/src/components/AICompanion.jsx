import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Mic, MicOff, Volume2, VolumeX, Users, Calendar, Brain, Bell, CalendarDays, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const WAKE_WORD = "hey mate";
const AVATAR_URL = "https://static.prod-images.emergentagent.com/jobs/7e703976-b1b5-4cac-8bb4-5016cc00ab1e/images/8861c004f3887363a550e65273e509f4b2a83fa5a022f2e61f3c6318cf99be61.png";

export const AICompanion = ({ onNavigate, userName = "Friend", onRefreshData }) => {
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isAwake, setIsAwake] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [displayText, setDisplayText] = useState('');
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [sessionId] = useState(() => `companion_${Date.now()}`);
  const [micPermission, setMicPermission] = useState('unknown');
  
  // Multi-turn conversation
  const [conversationMode, setConversationMode] = useState(null); // 'memory', 'family', 'reminder'
  const [pendingData, setPendingData] = useState({});
  const [currentField, setCurrentField] = useState(null);
  
  const recognitionRef = useRef(null);
  const audioRef = useRef(null);
  const isListeningRef = useRef(false);

  // Field prompts
  const FIELD_PROMPTS = {
    memory: {
      fields: ['title', 'date', 'location', 'description', 'people'],
      prompts: {
        title: "What would you like to call this memory?",
        date: "When did this happen?",
        location: "Where did this take place?",
        description: "Tell me more about what happened.",
        people: "Who was there with you?",
      }
    },
    family: {
      fields: ['name', 'relationship', 'notes'],
      prompts: {
        name: "What is their name?",
        relationship: "How are they related to you?",
        notes: "Tell me something special about them.",
      }
    },
    reminder: {
      fields: ['title', 'time'],
      prompts: {
        title: "What should I remind you about?",
        time: "What time? Say morning, afternoon, or evening.",
      }
    }
  };

  // Request microphone permission
  const requestMicPermission = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(track => track.stop());
      setMicPermission('granted');
      return true;
    } catch (err) {
      console.error('Microphone permission denied:', err);
      setMicPermission('denied');
      toast.error('Please allow microphone access to use voice commands');
      return false;
    }
  };

  // Initialize speech recognition
  const initRecognition = useCallback(() => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      toast.error('Voice recognition is not supported in this browser');
      return null;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;

    return recognition;
  }, []);

  // Setup recognition event handlers
  useEffect(() => {
    const recognition = initRecognition();
    if (!recognition) return;

    recognition.onstart = () => {
      console.log('Speech recognition started');
      isListeningRef.current = true;
    };

    recognition.onresult = (event) => {
      let finalTranscript = '';
      let interimTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const text = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += text;
        } else {
          interimTranscript += text;
        }
      }

      const heard = (finalTranscript || interimTranscript).toLowerCase().trim();
      console.log('Heard:', heard, 'Final:', !!finalTranscript);

      // Always show what we're hearing
      setTranscript(interimTranscript || finalTranscript);

      // Check for wake word if not awake
      if (!isAwake && heard.includes(WAKE_WORD)) {
        console.log('Wake word detected!');
        setIsAwake(true);
        setTranscript('');
        speak("Yes, I'm listening. How can I help you?");
        return;
      }

      // Process final transcript when awake
      if (isAwake && finalTranscript && !isSpeaking && !isProcessing) {
        const command = finalTranscript.replace(/hey mate/gi, '').trim();
        if (command.length > 2) {
          console.log('Processing command:', command);
          
          if (conversationMode && currentField) {
            handleFieldResponse(command);
          } else {
            processCommand(command);
          }
        }
      }
    };

    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      if (event.error === 'not-allowed') {
        setMicPermission('denied');
        toast.error('Microphone access denied. Please enable it in browser settings.');
      } else if (event.error !== 'no-speech' && event.error !== 'aborted') {
        // Try to restart on other errors
        setTimeout(() => {
          if (isListeningRef.current) {
            try { recognition.start(); } catch(e) {}
          }
        }, 1000);
      }
    };

    recognition.onend = () => {
      console.log('Speech recognition ended, isListening:', isListeningRef.current);
      // Auto-restart if we should still be listening
      if (isListeningRef.current && !isSpeaking) {
        setTimeout(() => {
          try {
            recognition.start();
          } catch (e) {
            console.log('Could not restart recognition:', e);
          }
        }, 100);
      }
    };

    recognitionRef.current = recognition;

    return () => {
      isListeningRef.current = false;
      try { recognition.stop(); } catch(e) {}
    };
  }, [isAwake, conversationMode, currentField, isSpeaking, isProcessing]);

  // Start listening
  const startListening = async () => {
    if (micPermission !== 'granted') {
      const granted = await requestMicPermission();
      if (!granted) return;
    }

    if (!recognitionRef.current) {
      recognitionRef.current = initRecognition();
    }

    try {
      isListeningRef.current = true;
      recognitionRef.current.start();
      setIsListening(true);
      toast.success('Listening! Say "Hey Mate" to talk to me.');
    } catch (e) {
      console.error('Failed to start recognition:', e);
      // If already started, that's fine
      if (e.message?.includes('already started')) {
        setIsListening(true);
      }
    }
  };

  // Stop listening
  const stopListening = () => {
    isListeningRef.current = false;
    setIsListening(false);
    setIsAwake(false);
    setTranscript('');
    try { recognitionRef.current?.stop(); } catch(e) {}
  };

  // Text-to-speech
  const speak = async (text) => {
    if (!text) return;
    
    setIsSpeaking(true);
    setDisplayText(text);

    // Pause recognition while speaking
    try { recognitionRef.current?.stop(); } catch(e) {}

    const onDone = () => {
      setIsSpeaking(false);
      // Resume listening after speaking
      if (isListeningRef.current) {
        setTimeout(() => {
          try { recognitionRef.current?.start(); } catch(e) {}
        }, 300);
      }
    };

    if (!audioEnabled) {
      setTimeout(onDone, Math.min(text.length * 50, 3000));
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
        audio.onended = onDone;
        audio.onerror = () => {
          console.log('Audio error, using fallback');
          fallbackSpeak(text, onDone);
        };
        await audio.play();
      } else {
        fallbackSpeak(text, onDone);
      }
    } catch (error) {
      console.error('TTS error:', error);
      fallbackSpeak(text, onDone);
    }
  };

  // Browser TTS fallback
  const fallbackSpeak = (text, onDone) => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.pitch = 1;
      utterance.onend = onDone;
      utterance.onerror = onDone;
      window.speechSynthesis.speak(utterance);
    } else {
      setTimeout(onDone, 2000);
    }
  };

  // Stop speaking
  const stopSpeaking = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    window.speechSynthesis?.cancel();
    setIsSpeaking(false);
  };

  // Process voice command
  const processCommand = async (command) => {
    if (!command.trim()) return;
    
    setIsProcessing(true);
    setIsAwake(false);
    setTranscript('');
    const lowerCommand = command.toLowerCase();

    // Check for creation intents
    if (lowerCommand.includes('add') || lowerCommand.includes('create') || lowerCommand.includes('new') || lowerCommand.includes('remember this')) {
      if (lowerCommand.includes('memory') || lowerCommand.includes('remember')) {
        startConversation('memory');
        setIsProcessing(false);
        return;
      }
      if (lowerCommand.includes('family') || lowerCommand.includes('person') || lowerCommand.includes('someone')) {
        startConversation('family');
        setIsProcessing(false);
        return;
      }
      if (lowerCommand.includes('reminder') || lowerCommand.includes('remind')) {
        startConversation('reminder');
        setIsProcessing(false);
        return;
      }
    }

    // Navigation commands
    const navMap = {
      'family': ['family', 'relatives', 'loved ones', 'who is my'],
      'timeline': ['memories', 'memory', 'timeline', 'remember'],
      'quiz': ['quiz', 'game', 'practice', 'faces'],
      'week': ['week', 'yesterday', 'today', 'recent', 'what did i do'],
      'assistant': ['chat', 'talk', 'help me'],
      'reminders': ['reminders', 'schedule', 'tasks', 'medications'],
      'home': ['home', 'start', 'main', 'back']
    };

    for (const [target, keywords] of Object.entries(navMap)) {
      if (keywords.some(kw => lowerCommand.includes(kw))) {
        const responses = {
          family: "Let me show you your family.",
          timeline: "Here are your precious memories.",
          quiz: "Let's practice remembering faces!",
          week: "Let me show you what you did recently.",
          assistant: "I'm here to help you.",
          reminders: "Here are your reminders.",
          home: "Taking you home."
        };
        speak(responses[target]);
        setTimeout(() => {
          onNavigate(target);
          setIsProcessing(false);
        }, 1500);
        return;
      }
    }

    // For questions, use backend RAG
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
        if (data.action === 'navigate') {
          setTimeout(() => onNavigate(data.target), 2000);
        }
      } else {
        speak("I'm not sure about that. Could you try asking differently?");
      }
    } catch (error) {
      speak("I'm having trouble connecting. Please try again.");
    }
    
    setIsProcessing(false);
  };

  // Start multi-turn conversation
  const startConversation = (mode) => {
    setConversationMode(mode);
    setPendingData({});
    const firstField = FIELD_PROMPTS[mode].fields[0];
    setCurrentField(firstField);
    setIsAwake(true);
    speak(`Great! Let's add a new ${mode}. ${FIELD_PROMPTS[mode].prompts[firstField]}`);
  };

  // Handle field response in conversation
  const handleFieldResponse = async (value) => {
    const lowerValue = value.toLowerCase();
    
    // Check for cancel
    if (lowerValue.includes('cancel') || lowerValue.includes('stop') || lowerValue.includes('never mind')) {
      speak("No problem, I've cancelled that.");
      resetConversation();
      return;
    }

    // Store value
    const newData = { ...pendingData, [currentField]: value };
    setPendingData(newData);
    setTranscript('');

    // Get next field
    const fields = FIELD_PROMPTS[conversationMode].fields;
    const currentIndex = fields.indexOf(currentField);
    const nextField = fields[currentIndex + 1];

    if (nextField) {
      setCurrentField(nextField);
      speak(`Got it! ${FIELD_PROMPTS[conversationMode].prompts[nextField]}`);
    } else {
      // All fields collected, save
      await saveData(newData);
    }
  };

  // Save collected data
  const saveData = async (data) => {
    setIsProcessing(true);
    
    try {
      let endpoint, body;
      
      if (conversationMode === 'memory') {
        endpoint = '/api/memories';
        body = {
          title: data.title,
          date: data.date,
          location: data.location || '',
          description: data.description || '',
          people: data.people ? data.people.split(/,|and/).map(p => p.trim()) : [],
          photos: []
        };
      } else if (conversationMode === 'family') {
        endpoint = '/api/family';
        body = {
          name: data.name,
          relationship: data.relationship.toLowerCase(),
          relationship_label: `Your ${data.relationship}`,
          notes: data.notes || '',
          photos: []
        };
      } else if (conversationMode === 'reminder') {
        endpoint = '/api/reminders';
        const timeMap = { morning: '09:00', afternoon: '14:00', evening: '18:00', night: '20:00' };
        const time = Object.entries(timeMap).find(([k]) => data.time?.toLowerCase().includes(k))?.[1] || '12:00';
        body = { title: data.title, time };
      }

      const res = await fetch(`${BACKEND_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(body)
      });

      if (res.ok) {
        speak(`Done! I've saved that for you. Say "Hey Mate" if you need anything else.`);
        if (onRefreshData) onRefreshData();
        toast.success('Saved successfully!');
      } else {
        speak("I had trouble saving that. Please try again.");
      }
    } catch (error) {
      speak("Something went wrong. Please try again.");
    }

    resetConversation();
    setIsProcessing(false);
  };

  // Reset conversation state
  const resetConversation = () => {
    setConversationMode(null);
    setCurrentField(null);
    setPendingData({});
    setIsAwake(false);
  };

  // Get status text
  const getStatus = () => {
    if (isProcessing) return { title: "Thinking...", subtitle: "Please wait" };
    if (isSpeaking) return { title: "I'm speaking...", subtitle: displayText };
    if (conversationMode) return { 
      title: `Adding ${conversationMode}...`, 
      subtitle: transcript || `Listening for ${currentField}...` 
    };
    if (isAwake) return { title: "I'm listening...", subtitle: transcript || "Tell me what you need" };
    if (isListening) return { title: `Hello, ${userName}!`, subtitle: 'Say "Hey Mate" to talk to me' };
    return { title: `Hello, ${userName}!`, subtitle: "Tap the microphone to start" };
  };

  const status = getStatus();
  const quickActions = [
    { id: 'family', label: 'Family', icon: Users },
    { id: 'timeline', label: 'Memories', icon: Calendar },
    { id: 'quiz', label: 'Quiz', icon: Brain },
    { id: 'week', label: 'My Week', icon: CalendarDays },
    { id: 'reminders', label: 'Today', icon: Bell },
  ];

  return (
    <div className="min-h-[80vh] flex flex-col items-center justify-center px-4 py-8">
      {/* Avatar */}
      <div className="relative mb-6">
        {/* Animated rings */}
        {isAwake && !isSpeaking && (
          <>
            <div className="absolute inset-[-15px] rounded-full border-4 border-success/50 animate-ping" style={{ animationDuration: '1.5s' }} />
            <div className="absolute inset-[-8px] rounded-full border-4 border-success/70 animate-pulse" />
          </>
        )}
        {isSpeaking && (
          <>
            <div className="absolute inset-[-15px] rounded-full border-4 border-primary/50 animate-ping" style={{ animationDuration: '1s' }} />
            <div className="absolute inset-[-8px] rounded-full border-4 border-primary/70 animate-pulse" />
          </>
        )}
        {isListening && !isAwake && !isSpeaking && (
          <div className="absolute inset-[-5px] rounded-full border-2 border-muted-foreground/40 animate-pulse" />
        )}
        
        {/* Avatar image */}
        <div 
          className={`
            relative w-44 h-44 md:w-52 md:h-52 rounded-full overflow-hidden cursor-pointer
            border-4 shadow-elevated transition-all duration-300
            ${isSpeaking ? 'border-primary scale-105' : ''}
            ${isAwake && !isSpeaking ? 'border-success scale-105' : ''}
            ${isListening && !isAwake && !isSpeaking ? 'border-muted-foreground/50' : ''}
            ${!isListening ? 'border-muted hover:border-primary hover:scale-102' : ''}
          `}
          onClick={() => {
            if (!isListening) {
              startListening();
            } else if (!isAwake && !isSpeaking) {
              setIsAwake(true);
              speak("Yes, how can I help you?");
            }
          }}
        >
          <img src={AVATAR_URL} alt="Memory Companion" className="w-full h-full object-cover" />
          
          {/* Mic indicator */}
          <div className={`absolute bottom-1 right-1 rounded-full p-2 shadow-lg transition-colors ${
            isAwake ? 'bg-success' : isListening ? 'bg-background' : 'bg-muted'
          }`}>
            {isProcessing ? (
              <Loader2 className="h-4 w-4 animate-spin text-primary" />
            ) : (
              <Mic className={`h-4 w-4 ${isAwake ? 'text-white animate-pulse' : isListening ? 'text-muted-foreground' : 'text-muted-foreground'}`} />
            )}
          </div>
        </div>
      </div>

      {/* Status */}
      <div className="text-center mb-6 max-w-lg">
        <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-2">{status.title}</h2>
        <p className={`text-lg md:text-xl ${isSpeaking ? 'text-primary' : isAwake ? 'text-success' : 'text-muted-foreground'}`}>
          {status.subtitle}
        </p>
        
        {/* Live transcript indicator */}
        {transcript && (isAwake || conversationMode) && (
          <div className="mt-3 p-3 bg-success/10 rounded-xl border border-success/30">
            <p className="text-success font-medium">🎤 "{transcript}"</p>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3 mb-8">
        <Button
          variant={isListening ? "destructive" : "accessible"}
          size="lg"
          className="gap-2 text-lg px-8 py-6"
          onClick={() => isListening ? stopListening() : startListening()}
          data-testid="mic-toggle-btn"
        >
          {isListening ? <MicOff className="h-6 w-6" /> : <Mic className="h-6 w-6" />}
          {isListening ? 'Stop' : 'Start Listening'}
        </Button>
        
        <Button
          variant="outline"
          size="lg"
          className="px-5 py-6"
          onClick={() => setAudioEnabled(!audioEnabled)}
        >
          {audioEnabled ? <Volume2 className="h-6 w-6" /> : <VolumeX className="h-6 w-6" />}
        </Button>

        {isSpeaking && (
          <Button variant="outline" size="lg" className="px-5 py-6" onClick={stopSpeaking}>
            Stop
          </Button>
        )}
      </div>

      {/* Mic permission warning */}
      {micPermission === 'denied' && (
        <div className="mb-6 p-4 bg-destructive/10 rounded-xl border border-destructive/30 max-w-md text-center">
          <p className="text-destructive font-medium">⚠️ Microphone access is blocked</p>
          <p className="text-sm text-muted-foreground mt-1">Please enable it in your browser settings to use voice commands</p>
        </div>
      )}

      {/* Quick actions */}
      <div className="w-full max-w-2xl">
        <p className="text-center text-sm text-muted-foreground mb-3">Or tap to go directly:</p>
        <div className="flex flex-wrap justify-center gap-2">
          {quickActions.map((action) => (
            <Button
              key={action.id}
              variant="outline"
              size="lg"
              className="gap-2"
              onClick={() => onNavigate(action.id)}
            >
              <action.icon className="h-5 w-5" />
              {action.label}
            </Button>
          ))}
        </div>
      </div>

      {/* Suggestions */}
      <div className="mt-8 text-center">
        <p className="text-sm text-muted-foreground mb-3">Try saying:</p>
        <div className="flex flex-wrap justify-center gap-2 max-w-xl">
          {["Add a new memory", "Who is in my family?", "What did I do yesterday?", "Add a family member", "Set a reminder"].map((cmd) => (
            <button
              key={cmd}
              onClick={() => { setIsAwake(true); processCommand(cmd); }}
              className="bg-muted hover:bg-muted/80 px-3 py-1.5 rounded-full text-sm transition-colors"
            >
              "{cmd}"
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default AICompanion;
