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
  const [isActive, setIsActive] = useState(false); // Active conversation (no need for wake word)
  const [transcript, setTranscript] = useState('');
  const [displayText, setDisplayText] = useState('');
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [micPermission, setMicPermission] = useState('unknown');
  
  // Multi-turn conversation for creating data
  const [creationMode, setCreationMode] = useState(null); // 'memory', 'family', 'reminder'
  const [creationData, setCreationData] = useState({});
  const [creationStep, setCreationStep] = useState(0);
  
  // Refs for callbacks
  const recognitionRef = useRef(null);
  const audioRef = useRef(null);
  const stateRef = useRef({
    isListening: false,
    isSpeaking: false,
    isActive: false,
    creationMode: null,
    creationStep: 0,
    creationData: {}
  });
  const activityTimeoutRef = useRef(null);

  // Keep ref in sync
  useEffect(() => {
    stateRef.current = {
      isListening,
      isSpeaking,
      isActive,
      creationMode,
      creationStep,
      creationData
    };
  }, [isListening, isSpeaking, isActive, creationMode, creationStep, creationData]);

  // Creation flow configuration
  const CREATION_FLOWS = {
    memory: {
      steps: ['title', 'date', 'location', 'description', 'people'],
      questions: {
        title: "What would you like to call this memory?",
        date: "When did this happen?",
        location: "Where was it?",
        description: "What happened? Tell me about it.",
        people: "Who was there?"
      }
    },
    family: {
      steps: ['name', 'relationship', 'notes'],
      questions: {
        name: "What is their name?",
        relationship: "How are they related to you? Like son, daughter, spouse, friend?",
        notes: "What's something special about them you want to remember?"
      }
    },
    reminder: {
      steps: ['title', 'time'],
      questions: {
        title: "What should I remind you about?",
        time: "When? You can say morning, afternoon, or evening."
      }
    }
  };

  // Reset activity timeout - keeps conversation active
  const resetActivityTimeout = useCallback(() => {
    if (activityTimeoutRef.current) {
      clearTimeout(activityTimeoutRef.current);
    }
    // Stay active for 30 seconds after last interaction
    activityTimeoutRef.current = setTimeout(() => {
      if (!stateRef.current.isSpeaking && !stateRef.current.creationMode) {
        setIsActive(false);
        setTranscript('');
      }
    }, 30000);
  }, []);

  // Stop all audio
  const stopAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = '';
      audioRef.current = null;
    }
    window.speechSynthesis?.cancel();
  }, []);

  // Request mic permission
  const requestMicPermission = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(track => track.stop());
      setMicPermission('granted');
      return true;
    } catch (err) {
      setMicPermission('denied');
      toast.error('Please allow microphone access');
      return false;
    }
  };

  // Speak text (TTS)
  const speak = useCallback(async (text, keepActive = true) => {
    if (!text) return;
    
    // Prevent multiple simultaneous speeches
    if (stateRef.current.isSpeaking) {
      console.log('Already speaking, queuing...');
      return;
    }

    console.log('Speaking:', text.substring(0, 50) + '...');
    
    stopAudio();
    setIsSpeaking(true);
    setDisplayText(text);
    
    // Stop recognition while speaking to avoid hearing ourselves
    try { recognitionRef.current?.stop(); } catch(e) {}

    const onComplete = () => {
      console.log('Speech complete');
      setIsSpeaking(false);
      
      if (keepActive) {
        setIsActive(true);
        resetActivityTimeout();
      }
      
      // Resume recognition after delay
      setTimeout(() => {
        if (stateRef.current.isListening && !stateRef.current.isSpeaking) {
          try { recognitionRef.current?.start(); } catch(e) {}
        }
      }, 800); // Longer delay to avoid hearing echo
    };

    if (!audioEnabled) {
      setTimeout(onComplete, Math.min(text.length * 50, 4000));
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
        audio.onended = onComplete;
        audio.onerror = () => browserSpeak(text, onComplete);
        await audio.play();
      } else {
        browserSpeak(text, onComplete);
      }
    } catch (error) {
      browserSpeak(text, onComplete);
    }
  }, [audioEnabled, stopAudio, resetActivityTimeout]);

  // Browser TTS fallback
  const browserSpeak = (text, onComplete) => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.onend = onComplete;
      utterance.onerror = onComplete;
      window.speechSynthesis.speak(utterance);
    } else {
      setTimeout(onComplete, 2000);
    }
  };

  // Start creation flow
  const startCreation = useCallback((type) => {
    console.log('Starting creation:', type);
    setCreationMode(type);
    setCreationData({});
    setCreationStep(0);
    const firstQuestion = CREATION_FLOWS[type].questions[CREATION_FLOWS[type].steps[0]];
    speak(`Okay, let's add a new ${type}. ${firstQuestion}`);
  }, [speak]);

  // Handle creation flow response
  const handleCreationResponse = useCallback(async (response) => {
    const { creationMode: mode, creationStep: step, creationData: data } = stateRef.current;
    if (!mode) return;

    const flow = CREATION_FLOWS[mode];
    const currentField = flow.steps[step];
    const newData = { ...data, [currentField]: response };
    setCreationData(newData);

    const nextStep = step + 1;
    if (nextStep < flow.steps.length) {
      // More questions to ask
      setCreationStep(nextStep);
      const nextField = flow.steps[nextStep];
      speak(`Got it. ${flow.questions[nextField]}`);
    } else {
      // All done - save the data
      await saveCreatedData(mode, newData);
    }
  }, [speak]);

  // Save created data
  const saveCreatedData = async (mode, data) => {
    setIsProcessing(true);
    try {
      let endpoint, body;

      if (mode === 'memory') {
        endpoint = '/api/memories';
        body = {
          title: data.title || 'Untitled Memory',
          date: data.date || 'Unknown date',
          location: data.location || '',
          description: data.description || '',
          people: data.people ? data.people.split(/,|and/).map(p => p.trim()).filter(Boolean) : [],
          photos: []
        };
      } else if (mode === 'family') {
        endpoint = '/api/family';
        body = {
          name: data.name || 'Unknown',
          relationship: (data.relationship || 'family').toLowerCase(),
          relationship_label: `Your ${data.relationship || 'family member'}`,
          notes: data.notes || '',
          photos: []
        };
      } else if (mode === 'reminder') {
        endpoint = '/api/reminders';
        const timeMap = { morning: '09:00', afternoon: '14:00', evening: '18:00', night: '20:00' };
        const timeStr = data.time?.toLowerCase() || '';
        const time = Object.entries(timeMap).find(([k]) => timeStr.includes(k))?.[1] || '12:00';
        body = { title: data.title || 'Reminder', time };
      }

      const res = await fetch(`${BACKEND_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(body)
      });

      if (res.ok) {
        speak(`Done! I've saved that. What else can I help you with?`);
        toast.success(`${mode} added successfully!`);
        onRefreshData?.();
      } else {
        speak("I had trouble saving that. Want to try again?");
      }
    } catch (error) {
      speak("Something went wrong. Please try again.");
    }

    setCreationMode(null);
    setCreationStep(0);
    setCreationData({});
    setIsProcessing(false);
  };

  // Process voice command
  const processCommand = useCallback(async (command) => {
    const lowerCmd = command.toLowerCase();
    console.log('Processing:', command);
    
    setIsProcessing(true);
    setTranscript('');

    // Check for cancel during creation
    if (stateRef.current.creationMode) {
      if (lowerCmd.includes('cancel') || lowerCmd.includes('stop') || lowerCmd.includes('never mind')) {
        setCreationMode(null);
        setCreationStep(0);
        setCreationData({});
        speak("Okay, cancelled. What else can I help with?");
        setIsProcessing(false);
        return;
      }
      // Continue creation flow
      handleCreationResponse(command);
      setIsProcessing(false);
      return;
    }

    // Check for creation intents
    if (lowerCmd.match(/\b(add|create|new|make|save)\b.*\b(memory|remember)\b/) || 
        lowerCmd.match(/\b(remember|record)\b.*\b(this|something)\b/)) {
      startCreation('memory');
      setIsProcessing(false);
      return;
    }
    if (lowerCmd.match(/\b(add|create|new)\b.*\b(family|person|someone|member)\b/)) {
      startCreation('family');
      setIsProcessing(false);
      return;
    }
    if (lowerCmd.match(/\b(add|create|set|new)\b.*\b(reminder|remind|alarm)\b/) ||
        lowerCmd.match(/\bremind me\b/)) {
      startCreation('reminder');
      setIsProcessing(false);
      return;
    }

    // Navigation
    const navPatterns = {
      family: /\b(family|relatives|loved ones|who is my)\b/,
      timeline: /\b(memories|memory|timeline|photos)\b/,
      quiz: /\b(quiz|game|practice|faces)\b/,
      week: /\b(week|yesterday|today|recent|what did i do)\b/,
      reminders: /\b(reminders?|schedule|tasks?|medication)\b/,
      home: /\b(home|back|start)\b/
    };

    for (const [target, pattern] of Object.entries(navPatterns)) {
      if (pattern.test(lowerCmd)) {
        const responses = {
          family: "Here's your family.",
          timeline: "Let me show you your memories.",
          quiz: "Let's practice remembering faces!",
          week: "Here's what you did recently.",
          reminders: "Here are your reminders.",
          home: "Going home."
        };
        speak(responses[target]);
        setTimeout(() => onNavigate(target), 1500);
        setIsProcessing(false);
        return;
      }
    }

    // General question - use backend
    try {
      const res = await fetch(`${BACKEND_URL}/api/voice-command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ text: command, session_id: `voice_${Date.now()}` })
      });

      if (res.ok) {
        const data = await res.json();
        speak(data.response);
        if (data.action === 'navigate') {
          setTimeout(() => onNavigate(data.target), 2000);
        }
      } else {
        speak("I'm not sure about that. Can you ask differently?");
      }
    } catch (error) {
      speak("I'm having trouble. Please try again.");
    }
    
    setIsProcessing(false);
  }, [speak, startCreation, handleCreationResponse, onNavigate]);

  // Initialize recognition
  useEffect(() => {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
      toast.error('Voice not supported in this browser');
      return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    let lastProcessedText = '';
    let processingLock = false;

    recognition.onresult = (event) => {
      // Skip if speaking
      if (stateRef.current.isSpeaking) return;

      let final = '';
      let interim = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const text = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          final += text;
        } else {
          interim += text;
        }
      }

      const heard = (final || interim).trim();
      if (interim) setTranscript(interim);

      // Check wake word if not active
      if (!stateRef.current.isActive && !stateRef.current.creationMode) {
        if (heard.toLowerCase().includes(WAKE_WORD)) {
          console.log('Wake word detected');
          setIsActive(true);
          setTranscript('');
          speak("Yes, I'm here. What do you need?");
          return;
        }
      }

      // Process final transcript
      if (final && (stateRef.current.isActive || stateRef.current.creationMode)) {
        const cleanCmd = final.replace(/hey mate/gi, '').trim();
        
        // Prevent duplicate processing
        if (cleanCmd.length > 2 && cleanCmd !== lastProcessedText && !processingLock) {
          lastProcessedText = cleanCmd;
          processingLock = true;
          setTimeout(() => { processingLock = false; }, 2000);
          
          setTranscript(cleanCmd);
          processCommand(cleanCmd);
        }
      }
    };

    recognition.onerror = (event) => {
      console.error('Recognition error:', event.error);
      if (event.error === 'not-allowed') {
        setMicPermission('denied');
      }
    };

    recognition.onend = () => {
      // Auto restart if listening and not speaking
      if (stateRef.current.isListening && !stateRef.current.isSpeaking) {
        setTimeout(() => {
          if (stateRef.current.isListening && !stateRef.current.isSpeaking) {
            try { recognition.start(); } catch(e) {}
          }
        }, 300);
      }
    };

    recognitionRef.current = recognition;

    return () => {
      try { recognition.stop(); } catch(e) {}
      if (activityTimeoutRef.current) clearTimeout(activityTimeoutRef.current);
    };
  }, [processCommand, speak]);

  // Start listening
  const startListening = async () => {
    if (micPermission !== 'granted') {
      const ok = await requestMicPermission();
      if (!ok) return;
    }

    setIsListening(true);
    try { recognitionRef.current?.start(); } catch(e) {}
    toast.success('Listening! Say "Hey Mate" to start.');
  };

  // Stop listening
  const stopListening = () => {
    setIsListening(false);
    setIsActive(false);
    setTranscript('');
    setCreationMode(null);
    setCreationStep(0);
    setCreationData({});
    stopAudio();
    try { recognitionRef.current?.stop(); } catch(e) {}
  };

  // Status display
  const getStatus = () => {
    if (isProcessing) return { title: "Thinking...", sub: "One moment" };
    if (isSpeaking) return { title: "Speaking...", sub: displayText };
    if (creationMode) {
      const field = CREATION_FLOWS[creationMode].steps[creationStep];
      return { title: `Adding ${creationMode}...`, sub: transcript || `Tell me the ${field}` };
    }
    if (isActive) return { title: "I'm listening!", sub: transcript || "Go ahead, I'm here" };
    if (isListening) return { title: `Hello, ${userName}!`, sub: 'Say "Hey Mate" to talk' };
    return { title: `Hello, ${userName}!`, sub: "Tap Start to begin" };
  };

  const status = getStatus();
  const showActiveState = isActive || creationMode;

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
        {showActiveState && !isSpeaking && (
          <div className="absolute inset-[-12px] rounded-full border-4 border-success animate-pulse" />
        )}
        {isSpeaking && (
          <div className="absolute inset-[-12px] rounded-full border-4 border-primary animate-pulse" />
        )}
        {isListening && !showActiveState && !isSpeaking && (
          <div className="absolute inset-[-6px] rounded-full border-2 border-muted-foreground/40" />
        )}
        
        <div 
          className={`
            relative w-44 h-44 md:w-52 md:h-52 rounded-full overflow-hidden cursor-pointer
            border-4 shadow-elevated transition-all duration-300
            ${isSpeaking ? 'border-primary scale-105' : ''}
            ${showActiveState && !isSpeaking ? 'border-success scale-105' : ''}
            ${isListening && !showActiveState && !isSpeaking ? 'border-muted-foreground/50' : ''}
            ${!isListening ? 'border-muted hover:border-primary' : ''}
          `}
          onClick={() => {
            if (!isListening) startListening();
            else if (!isActive && !isSpeaking) {
              setIsActive(true);
              speak("Yes, how can I help?");
            }
          }}
        >
          <img src={AVATAR_URL} alt="Companion" className="w-full h-full object-cover" />
          
          <div className={`absolute bottom-1 right-1 rounded-full p-2 shadow-lg ${
            showActiveState ? 'bg-success' : isSpeaking ? 'bg-primary' : 'bg-background'
          }`}>
            {isProcessing ? (
              <Loader2 className="h-4 w-4 animate-spin text-white" />
            ) : (
              <Mic className={`h-4 w-4 ${showActiveState || isSpeaking ? 'text-white' : 'text-muted-foreground'}`} />
            )}
          </div>
        </div>
      </div>

      {/* Status */}
      <div className="text-center mb-6 max-w-lg">
        <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-2">{status.title}</h2>
        <p className={`text-lg ${isSpeaking ? 'text-primary' : showActiveState ? 'text-success' : 'text-muted-foreground'}`}>
          {status.sub}
        </p>
        
        {transcript && showActiveState && !isSpeaking && (
          <div className="mt-4 p-3 bg-success/10 rounded-xl border-2 border-success/40">
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
        >
          {isListening ? <MicOff className="h-6 w-6" /> : <Mic className="h-6 w-6" />}
          {isListening ? 'Stop' : 'Start Listening'}
        </Button>
        
        <Button variant="outline" size="lg" className="px-5 py-6" onClick={() => setAudioEnabled(!audioEnabled)}>
          {audioEnabled ? <Volume2 className="h-6 w-6" /> : <VolumeX className="h-6 w-6" />}
        </Button>

        {isSpeaking && (
          <Button variant="outline" size="lg" className="px-5 py-6" onClick={() => { stopAudio(); setIsSpeaking(false); }}>
            Stop
          </Button>
        )}
      </div>

      {micPermission === 'denied' && (
        <div className="mb-6 p-4 bg-destructive/10 rounded-xl border border-destructive/30 text-center">
          <p className="text-destructive font-medium">⚠️ Microphone blocked</p>
        </div>
      )}

      {/* Quick navigation */}
      <div className="w-full max-w-2xl">
        <p className="text-center text-sm text-muted-foreground mb-3">Or tap:</p>
        <div className="flex flex-wrap justify-center gap-2">
          {quickActions.map((a) => (
            <Button key={a.id} variant="outline" size="lg" className="gap-2" onClick={() => onNavigate(a.id)}>
              <a.icon className="h-5 w-5" /> {a.label}
            </Button>
          ))}
        </div>
      </div>

      {/* Suggestions */}
      <div className="mt-8 text-center">
        <p className="text-sm text-muted-foreground mb-3">Try saying:</p>
        <div className="flex flex-wrap justify-center gap-2 max-w-xl">
          {["Add a memory", "Who is Maria?", "Add a family member", "Set a reminder", "Show my week"].map((cmd) => (
            <button
              key={cmd}
              onClick={() => { setIsActive(true); processCommand(cmd); }}
              className="bg-muted hover:bg-muted/80 px-3 py-1.5 rounded-full text-sm"
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
