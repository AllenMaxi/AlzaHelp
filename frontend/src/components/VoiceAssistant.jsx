import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Mic, MicOff, Volume2, VolumeX, X, Loader2, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const WAKE_WORD = "hey memory";

export const VoiceAssistant = ({ onNavigate, userName = "Friend" }) => {
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [response, setResponse] = useState('');
  const [wakeWordDetected, setWakeWordDetected] = useState(false);
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [sessionId] = useState(() => `voice_${Date.now()}`);
  
  const recognitionRef = useRef(null);
  const audioRef = useRef(null);
  const silenceTimeoutRef = useRef(null);

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
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript;
        } else {
          interimTranscript += transcript;
        }
      }

      const currentTranscript = (finalTranscript || interimTranscript).toLowerCase();
      
      // Check for wake word
      if (!wakeWordDetected && currentTranscript.includes(WAKE_WORD)) {
        setWakeWordDetected(true);
        setIsExpanded(true);
        setTranscript('');
        speak("Yes, I'm listening. How can I help you?");
        return;
      }

      // If wake word was detected, capture the command
      if (wakeWordDetected && finalTranscript) {
        // Remove wake word from transcript
        const command = finalTranscript.replace(/hey memory/gi, '').trim();
        if (command.length > 2) {
          setTranscript(command);
          processCommand(command);
          setWakeWordDetected(false);
        }
      } else if (wakeWordDetected) {
        setTranscript(interimTranscript.replace(/hey memory/gi, '').trim());
        
        // Reset silence timeout
        if (silenceTimeoutRef.current) {
          clearTimeout(silenceTimeoutRef.current);
        }
        silenceTimeoutRef.current = setTimeout(() => {
          if (wakeWordDetected && !isProcessing) {
            setWakeWordDetected(false);
            setTranscript('');
          }
        }, 5000);
      }
    };

    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      if (event.error !== 'no-speech') {
        setIsListening(false);
      }
    };

    recognition.onend = () => {
      // Restart if we should still be listening
      if (isListening) {
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
  }, [wakeWordDetected, isProcessing, isListening]);

  // Start/stop listening
  const toggleListening = useCallback(() => {
    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
      setWakeWordDetected(false);
      setTranscript('');
    } else {
      try {
        recognitionRef.current?.start();
        setIsListening(true);
        toast.success('Voice assistant activated. Say "Hey Memory" to start!');
      } catch (e) {
        console.error('Failed to start recognition:', e);
        toast.error('Could not start voice recognition');
      }
    }
  }, [isListening]);

  // Text-to-speech using OpenAI TTS
  const speak = async (text) => {
    if (!audioEnabled || !text) return;
    
    setIsSpeaking(true);
    setResponse(text);

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
          setIsSpeaking(false);
        };
        
        audio.onerror = () => {
          setIsSpeaking(false);
          // Fallback to browser TTS
          fallbackSpeak(text);
        };
        
        await audio.play();
      } else {
        // Fallback to browser TTS
        fallbackSpeak(text);
      }
    } catch (error) {
      console.error('TTS error:', error);
      fallbackSpeak(text);
    }
  };

  // Fallback browser TTS
  const fallbackSpeak = (text) => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.85;
      utterance.onend = () => setIsSpeaking(false);
      window.speechSynthesis.speak(utterance);
    } else {
      setIsSpeaking(false);
    }
  };

  // Stop speaking
  const stopSpeaking = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    setIsSpeaking(false);
  };

  // Process voice command
  const processCommand = async (command) => {
    if (!command.trim()) return;
    
    setIsProcessing(true);
    
    try {
      const res = await fetch(`${BACKEND_URL}/api/voice-command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ text: command, session_id: sessionId })
      });

      if (res.ok) {
        const data = await res.json();
        
        // Handle navigation
        if (data.action === 'navigate' && onNavigate) {
          speak(data.response);
          setTimeout(() => {
            onNavigate(data.target);
          }, 1500);
        } 
        // Handle create actions
        else if (data.action === 'create_memory') {
          speak(data.response);
          setTimeout(() => {
            onNavigate('timeline');
          }, 1500);
        }
        else if (data.action === 'create_reminder') {
          speak(data.response);
          setTimeout(() => {
            onNavigate('reminders');
          }, 1500);
        }
        else if (data.action === 'create_family') {
          speak(data.response);
          setTimeout(() => {
            onNavigate('family');
          }, 1500);
        }
        // Just speak the response
        else {
          speak(data.response);
        }
      } else {
        speak("I'm sorry, I couldn't understand that. Could you try again?");
      }
    } catch (error) {
      console.error('Voice command error:', error);
      speak("I'm having trouble connecting. Please try again.");
    } finally {
      setIsProcessing(false);
      setTranscript('');
    }
  };

  // Manual command input (for accessibility)
  const handleManualCommand = () => {
    if (transcript.trim()) {
      processCommand(transcript);
    }
  };

  return (
    <>
      {/* Floating Button */}
      <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end gap-3">
        {/* Expanded Panel */}
        {isExpanded && (
          <div className="bg-background border-2 border-primary rounded-2xl shadow-elevated p-4 w-80 animate-scale-in">
            {/* Header */}
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-primary" />
                <span className="font-semibold text-foreground">Voice Assistant</span>
              </div>
              <Button 
                variant="ghost" 
                size="icon" 
                className="h-8 w-8"
                onClick={() => setIsExpanded(false)}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>

            {/* Status */}
            <div className="bg-muted rounded-xl p-3 mb-3 min-h-[60px]">
              {isProcessing ? (
                <div className="flex items-center gap-2 text-primary">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Thinking...</span>
                </div>
              ) : isSpeaking ? (
                <div className="flex items-center gap-2 text-primary">
                  <Volume2 className="h-4 w-4 animate-pulse" />
                  <span className="text-sm">{response}</span>
                </div>
              ) : wakeWordDetected ? (
                <div className="flex items-center gap-2 text-success">
                  <Mic className="h-4 w-4 animate-pulse" />
                  <span>{transcript || "Listening..."}</span>
                </div>
              ) : isListening ? (
                <div className="text-muted-foreground text-sm">
                  Say <span className="font-bold text-primary">"Hey Memory"</span> to talk to me
                </div>
              ) : (
                <div className="text-muted-foreground text-sm">
                  Tap the microphone to start
                </div>
              )}
            </div>

            {/* Response area */}
            {response && !isSpeaking && (
              <div className="bg-primary/10 rounded-xl p-3 mb-3">
                <p className="text-sm text-foreground">{response}</p>
              </div>
            )}

            {/* Controls */}
            <div className="flex items-center gap-2">
              <Button
                variant={isListening ? "destructive" : "accessible"}
                className="flex-1 gap-2"
                onClick={toggleListening}
              >
                {isListening ? (
                  <>
                    <MicOff className="h-5 w-5" />
                    Stop Listening
                  </>
                ) : (
                  <>
                    <Mic className="h-5 w-5" />
                    Start Listening
                  </>
                )}
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={() => setAudioEnabled(!audioEnabled)}
                className="shrink-0"
              >
                {audioEnabled ? <Volume2 className="h-5 w-5" /> : <VolumeX className="h-5 w-5" />}
              </Button>
              {isSpeaking && (
                <Button
                  variant="outline"
                  size="icon"
                  onClick={stopSpeaking}
                  className="shrink-0"
                >
                  <X className="h-5 w-5" />
                </Button>
              )}
            </div>

            {/* Quick Commands */}
            <div className="mt-3 pt-3 border-t border-border">
              <p className="text-xs text-muted-foreground mb-2">Try saying:</p>
              <div className="flex flex-wrap gap-1">
                {["Show my family", "Who is Maria?", "Play quiz", "What did I do yesterday?"].map((cmd) => (
                  <button
                    key={cmd}
                    onClick={() => processCommand(cmd)}
                    className="text-xs bg-muted hover:bg-muted/80 px-2 py-1 rounded-full transition-colors"
                  >
                    {cmd}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Main Floating Button */}
        <button
          onClick={() => {
            setIsExpanded(!isExpanded);
            if (!isExpanded && !isListening) {
              toggleListening();
            }
          }}
          className={`
            relative flex items-center justify-center
            w-16 h-16 rounded-full shadow-elevated
            transition-all duration-300 hover:scale-105
            ${isListening 
              ? 'bg-primary animate-pulse' 
              : 'bg-primary hover:bg-primary/90'
            }
            ${wakeWordDetected ? 'ring-4 ring-success ring-opacity-50' : ''}
          `}
          data-testid="voice-assistant-button"
        >
          {isProcessing ? (
            <Loader2 className="h-8 w-8 text-primary-foreground animate-spin" />
          ) : isListening ? (
            <Mic className="h-8 w-8 text-primary-foreground" />
          ) : (
            <Mic className="h-8 w-8 text-primary-foreground" />
          )}
          
          {/* Listening indicator */}
          {isListening && (
            <span className="absolute -top-1 -right-1 flex h-4 w-4">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-success opacity-75"></span>
              <span className="relative inline-flex rounded-full h-4 w-4 bg-success"></span>
            </span>
          )}
        </button>

        {/* Hint text */}
        {!isExpanded && (
          <div className="bg-background/90 backdrop-blur px-3 py-1 rounded-full shadow-soft text-sm">
            {isListening ? (
              <span className="text-primary font-medium">Say "Hey Memory"</span>
            ) : (
              <span className="text-muted-foreground">Voice Assistant</span>
            )}
          </div>
        )}
      </div>
    </>
  );
};

export default VoiceAssistant;
