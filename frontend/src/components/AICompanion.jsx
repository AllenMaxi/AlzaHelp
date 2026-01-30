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
  const [micPermission, setMicPermission] = useState('unknown');
  
  // Multi-turn conversation
  const [conversationMode, setConversationMode] = useState(null);
  const [pendingData, setPendingData] = useState({});
  const [currentField, setCurrentField] = useState(null);
  
  // Refs for stable state access
  const recognitionRef = useRef(null);
  const audioRef = useRef(null);
  const isListeningRef = useRef(false);
  const isSpeakingRef = useRef(false);
  const isAwakeRef = useRef(false);
  const conversationModeRef = useRef(null);
  const currentFieldRef = useRef(null);
  const pendingDataRef = useRef({});
  const wakeWordCooldownRef = useRef(false);
  const sessionIdRef = useRef(`companion_${Date.now()}`);

  // Keep refs in sync
  useEffect(() => { isAwakeRef.current = isAwake; }, [isAwake]);
  useEffect(() => { conversationModeRef.current = conversationMode; }, [conversationMode]);
  useEffect(() => { currentFieldRef.current = currentField; }, [currentField]);
  useEffect(() => { pendingDataRef.current = pendingData; }, [pendingData]);

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

  // Stop all audio immediately
  const stopAllAudio = useCallback(() => {
    // Stop HTML5 audio
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = '';
      audioRef.current = null;
    }
    // Stop browser speech synthesis
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    isSpeakingRef.current = false;
    setIsSpeaking(false);
  }, []);

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

  // Start speech recognition
  const startRecognition = useCallback(() => {
    if (!recognitionRef.current || isSpeakingRef.current) return;
    
    try {
      recognitionRef.current.start();
      console.log('Recognition started');
    } catch (e) {
      if (!e.message?.includes('already started')) {
        console.error('Failed to start recognition:', e);
      }
    }
  }, []);

  // Stop speech recognition
  const stopRecognition = useCallback(() => {
    if (!recognitionRef.current) return;
    try {
      recognitionRef.current.stop();
      console.log('Recognition stopped');
    } catch (e) {}
  }, []);

  // Text-to-speech
  const speak = useCallback(async (text) => {
    if (!text || isSpeakingRef.current) {
      console.log('Skipping speak - already speaking or no text');
      return;
    }
    
    console.log('Speaking:', text);
    
    // Stop everything first
    stopAllAudio();
    stopRecognition();
    
    isSpeakingRef.current = true;
    setIsSpeaking(true);
    setDisplayText(text);

    const onDone = () => {
      console.log('Speech done');
      isSpeakingRef.current = false;
      setIsSpeaking(false);
      
      // Restart recognition after a small delay
      if (isListeningRef.current) {
        setTimeout(() => {
          if (!isSpeakingRef.current) {
            startRecognition();
          }
        }, 500);
      }
    };

    if (!audioEnabled) {
      setTimeout(onDone, Math.min(text.length * 40, 3000));
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
          browserSpeak(text, onDone);
        };
        
        await audio.play();
      } else {
        browserSpeak(text, onDone);
      }
    } catch (error) {
      console.error('TTS error:', error);
      browserSpeak(text, onDone);
    }
  }, [audioEnabled, stopAllAudio, stopRecognition, startRecognition]);

  // Browser TTS fallback
  const browserSpeak = useCallback((text, onDone) => {
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
  }, []);

  // Process voice command
  const processCommand = useCallback(async (command) => {
    if (!command.trim()) return;
    
    console.log('Processing command:', command);
    setIsProcessing(true);
    setTranscript('');
    
    const lowerCommand = command.toLowerCase();

    // Check for creation intents
    if (lowerCommand.includes('add') || lowerCommand.includes('create') || lowerCommand.includes('new') || lowerCommand.includes('remember this')) {
      if (lowerCommand.includes('memory') || lowerCommand.includes('remember')) {
        setConversationMode('memory');
        setPendingData({});
        setCurrentField('title');
        setIsProcessing(false);
        speak(`Great! Let's add a new memory. ${FIELD_PROMPTS.memory.prompts.title}`);
        return;
      }
      if (lowerCommand.includes('family') || lowerCommand.includes('person') || lowerCommand.includes('someone')) {
        setConversationMode('family');
        setPendingData({});
        setCurrentField('name');
        setIsProcessing(false);
        speak(`Great! Let's add a family member. ${FIELD_PROMPTS.family.prompts.name}`);
        return;
      }
      if (lowerCommand.includes('reminder') || lowerCommand.includes('remind')) {
        setConversationMode('reminder');
        setPendingData({});
        setCurrentField('title');
        setIsProcessing(false);
        speak(`Sure! Let's set a reminder. ${FIELD_PROMPTS.reminder.prompts.title}`);
        return;
      }
    }

    // Navigation commands
    const navMap = {
      'family': ['family', 'relatives', 'loved ones'],
      'timeline': ['memories', 'memory', 'timeline'],
      'quiz': ['quiz', 'game', 'practice', 'faces'],
      'week': ['week', 'yesterday', 'recent', 'what did i do'],
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
          reminders: "Here are your reminders.",
          home: "Taking you home."
        };
        
        setIsProcessing(false);
        setIsAwake(false);
        speak(responses[target]);
        setTimeout(() => onNavigate(target), 2000);
        return;
      }
    }

    // For questions, use backend RAG
    try {
      const res = await fetch(`${BACKEND_URL}/api/voice-command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ text: command, session_id: sessionIdRef.current })
      });

      if (res.ok) {
        const data = await res.json();
        setIsAwake(false);
        speak(data.response);
        if (data.action === 'navigate') {
          setTimeout(() => onNavigate(data.target), 2500);
        }
      } else {
        setIsAwake(false);
        speak("I'm not sure about that. Could you try asking differently?");
      }
    } catch (error) {
      setIsAwake(false);
      speak("I'm having trouble connecting. Please try again.");
    }
    
    setIsProcessing(false);
  }, [speak, onNavigate]);

  // Handle field response in conversation
  const handleFieldResponse = useCallback(async (value) => {
    const lowerValue = value.toLowerCase();
    const mode = conversationModeRef.current;
    const field = currentFieldRef.current;
    const data = pendingDataRef.current;
    
    // Check for cancel
    if (lowerValue.includes('cancel') || lowerValue.includes('stop') || lowerValue.includes('never mind')) {
      speak("No problem, cancelled.");
      setConversationMode(null);
      setCurrentField(null);
      setPendingData({});
      setIsAwake(false);
      return;
    }

    // Store value
    const newData = { ...data, [field]: value };
    setPendingData(newData);
    pendingDataRef.current = newData;
    setTranscript('');

    // Get next field
    const fields = FIELD_PROMPTS[mode].fields;
    const currentIndex = fields.indexOf(field);
    const nextField = fields[currentIndex + 1];

    if (nextField) {
      setCurrentField(nextField);
      speak(`Got it! ${FIELD_PROMPTS[mode].prompts[nextField]}`);
    } else {
      // All fields collected - save
      await saveData(mode, newData);
    }
  }, [speak]);

  // Save collected data
  const saveData = useCallback(async (mode, data) => {
    setIsProcessing(true);
    
    try {
      let endpoint, body;
      
      if (mode === 'memory') {
        endpoint = '/api/memories';
        body = {
          title: data.title,
          date: data.date,
          location: data.location || '',
          description: data.description || '',
          people: data.people ? data.people.split(/,|and/).map(p => p.trim()) : [],
          photos: []
        };
      } else if (mode === 'family') {
        endpoint = '/api/family';
        body = {
          name: data.name,
          relationship: data.relationship.toLowerCase(),
          relationship_label: `Your ${data.relationship}`,
          notes: data.notes || '',
          photos: []
        };
      } else if (mode === 'reminder') {
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
        speak("Done! I saved that for you. Say Hey Mate if you need anything else.");
        if (onRefreshData) onRefreshData();
        toast.success('Saved successfully!');
      } else {
        speak("I had trouble saving. Please try again.");
      }
    } catch (error) {
      speak("Something went wrong. Please try again.");
    }

    setConversationMode(null);
    setCurrentField(null);
    setPendingData({});
    setIsAwake(false);
    setIsProcessing(false);
  }, [speak, onRefreshData]);

  // Initialize speech recognition once
  useEffect(() => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      toast.error('Voice recognition not supported in this browser');
      return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      console.log('Recognition started');
    };

    recognition.onresult = (event) => {
      // Don't process if speaking
      if (isSpeakingRef.current) return;
      
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
      
      // Show what we're hearing
      if (interimTranscript) {
        setTranscript(interimTranscript);
      }

      // Check for wake word (with cooldown to prevent multiple triggers)
      if (!isAwakeRef.current && !wakeWordCooldownRef.current && heard.includes(WAKE_WORD)) {
        console.log('Wake word detected!');
        wakeWordCooldownRef.current = true;
        setTimeout(() => { wakeWordCooldownRef.current = false; }, 3000);
        
        setIsAwake(true);
        isAwakeRef.current = true;
        setTranscript('');
        speak("Yes, I'm listening. How can I help?");
        return;
      }

      // Process final command when awake
      if (isAwakeRef.current && finalTranscript && !isSpeakingRef.current) {
        const command = finalTranscript.replace(/hey mate/gi, '').trim();
        
        if (command.length > 2) {
          console.log('Final command:', command);
          setTranscript(command);
          
          // Check if in conversation mode
          if (conversationModeRef.current && currentFieldRef.current) {
            handleFieldResponse(command);
          } else {
            processCommand(command);
          }
        }
      }
    };

    recognition.onerror = (event) => {
      console.error('Recognition error:', event.error);
      if (event.error === 'not-allowed') {
        setMicPermission('denied');
        toast.error('Microphone access denied');
      }
    };

    recognition.onend = () => {
      console.log('Recognition ended, listening:', isListeningRef.current, 'speaking:', isSpeakingRef.current);
      
      // Auto-restart if should be listening and not speaking
      if (isListeningRef.current && !isSpeakingRef.current) {
        setTimeout(() => {
          if (isListeningRef.current && !isSpeakingRef.current) {
            try {
              recognition.start();
            } catch (e) {}
          }
        }, 200);
      }
    };

    recognitionRef.current = recognition;

    return () => {
      isListeningRef.current = false;
      try { recognition.stop(); } catch(e) {}
    };
  }, [speak, processCommand, handleFieldResponse]);

  // Start listening
  const startListening = async () => {
    if (micPermission !== 'granted') {
      const granted = await requestMicPermission();
      if (!granted) return;
    }

    isListeningRef.current = true;
    setIsListening(true);
    startRecognition();
    toast.success('Listening! Say "Hey Mate" to talk to me.');
  };

  // Stop listening
  const stopListening = () => {
    isListeningRef.current = false;
    setIsListening(false);
    setIsAwake(false);
    isAwakeRef.current = false;
    setTranscript('');
    setConversationMode(null);
    setCurrentField(null);
    setPendingData({});
    stopAllAudio();
    stopRecognition();
  };

  // Get status text
  const getStatus = () => {
    if (isProcessing) return { title: "Thinking...", subtitle: "Please wait" };
    if (isSpeaking) return { title: "Speaking...", subtitle: displayText };
    if (conversationMode) return { 
      title: `Adding ${conversationMode}...`, 
      subtitle: transcript || `Waiting for ${currentField}...` 
    };
    if (isAwake) return { title: "I'm listening...", subtitle: transcript || "Tell me what you need" };
    if (isListening) return { title: `Hello, ${userName}!`, subtitle: 'Say "Hey Mate" to talk to me' };
    return { title: `Hello, ${userName}!`, subtitle: "Tap Start Listening to begin" };
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
        {(isAwake || conversationMode) && !isSpeaking && (
          <>
            <div className="absolute inset-[-15px] rounded-full border-4 border-success/40 animate-ping" style={{ animationDuration: '2s' }} />
            <div className="absolute inset-[-8px] rounded-full border-4 border-success animate-pulse" />
          </>
        )}
        {isSpeaking && (
          <>
            <div className="absolute inset-[-15px] rounded-full border-4 border-primary/40 animate-ping" style={{ animationDuration: '1s' }} />
            <div className="absolute inset-[-8px] rounded-full border-4 border-primary animate-pulse" />
          </>
        )}
        {isListening && !isAwake && !isSpeaking && !conversationMode && (
          <div className="absolute inset-[-5px] rounded-full border-2 border-muted-foreground/30 animate-pulse" />
        )}
        
        {/* Avatar image */}
        <div 
          className={`
            relative w-44 h-44 md:w-52 md:h-52 rounded-full overflow-hidden cursor-pointer
            border-4 shadow-elevated transition-all duration-300
            ${isSpeaking ? 'border-primary scale-105' : ''}
            ${(isAwake || conversationMode) && !isSpeaking ? 'border-success scale-105' : ''}
            ${isListening && !isAwake && !isSpeaking ? 'border-muted-foreground/50' : ''}
            ${!isListening ? 'border-muted hover:border-primary' : ''}
          `}
          onClick={() => {
            if (!isListening) {
              startListening();
            } else if (!isAwake && !isSpeaking) {
              setIsAwake(true);
              isAwakeRef.current = true;
              speak("Yes, how can I help you?");
            }
          }}
        >
          <img src={AVATAR_URL} alt="Memory Companion" className="w-full h-full object-cover" />
          
          {/* Status indicator */}
          <div className={`absolute bottom-1 right-1 rounded-full p-2 shadow-lg ${
            isAwake || conversationMode ? 'bg-success' : isSpeaking ? 'bg-primary' : isListening ? 'bg-background' : 'bg-muted'
          }`}>
            {isProcessing ? (
              <Loader2 className="h-4 w-4 animate-spin text-white" />
            ) : (
              <Mic className={`h-4 w-4 ${(isAwake || conversationMode || isSpeaking) ? 'text-white' : 'text-muted-foreground'} ${isAwake ? 'animate-pulse' : ''}`} />
            )}
          </div>
        </div>
      </div>

      {/* Status */}
      <div className="text-center mb-6 max-w-lg">
        <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-2">{status.title}</h2>
        <p className={`text-lg md:text-xl transition-colors ${
          isSpeaking ? 'text-primary' : (isAwake || conversationMode) ? 'text-success' : 'text-muted-foreground'
        }`}>
          {status.subtitle}
        </p>
        
        {/* Live transcript */}
        {transcript && (isAwake || conversationMode) && !isSpeaking && (
          <div className="mt-4 p-3 bg-success/10 rounded-xl border-2 border-success/30 animate-pulse">
            <p className="text-success font-medium">🎤 Heard: "{transcript}"</p>
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
        
        <Button
          variant="outline"
          size="lg"
          className="px-5 py-6"
          onClick={() => setAudioEnabled(!audioEnabled)}
          title={audioEnabled ? 'Mute voice' : 'Unmute voice'}
        >
          {audioEnabled ? <Volume2 className="h-6 w-6" /> : <VolumeX className="h-6 w-6" />}
        </Button>

        {isSpeaking && (
          <Button variant="outline" size="lg" className="px-5 py-6" onClick={stopAllAudio}>
            Stop
          </Button>
        )}
      </div>

      {/* Mic permission warning */}
      {micPermission === 'denied' && (
        <div className="mb-6 p-4 bg-destructive/10 rounded-xl border border-destructive/30 max-w-md text-center">
          <p className="text-destructive font-medium">⚠️ Microphone blocked</p>
          <p className="text-sm text-muted-foreground mt-1">Enable it in browser settings</p>
        </div>
      )}

      {/* Quick navigation */}
      <div className="w-full max-w-2xl">
        <p className="text-center text-sm text-muted-foreground mb-3">Or tap to navigate:</p>
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
              onClick={() => {
                setIsAwake(true);
                isAwakeRef.current = true;
                processCommand(cmd);
              }}
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
