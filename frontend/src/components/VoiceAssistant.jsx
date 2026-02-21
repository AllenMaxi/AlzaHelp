import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  X,
  Loader2,
  Sparkles
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const WAKE_WORD = "hey memory";
const WAKE_COOLDOWN_MS = 3000;
const SESSION_TIMEOUT_MS = 60000;
const RESTART_DELAY_MS = 600;

export const VoiceAssistant = ({ onNavigate, userName = "Friend" }) => {
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [response, setResponse] = useState("");
  const [sessionActive, setSessionActive] = useState(false);
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [sessionId] = useState(() => `voice_${Date.now()}`);

  const recognitionRef = useRef(null);
  const audioRef = useRef(null);
  const sessionTimeoutRef = useRef(null);
  const wakeWordCooldownRef = useRef(false);
  const lastProcessedRef = useRef("");
  const processingLockRef = useRef(false);

  // Refs to access latest state inside recognition callbacks without re-creating recognition
  const stateRef = useRef({
    isListening: false,
    isSpeaking: false,
    isProcessing: false,
    sessionActive: false,
    audioEnabled: true
  });

  useEffect(() => {
    stateRef.current = { isListening, isSpeaking, isProcessing, sessionActive, audioEnabled };
  }, [isListening, isSpeaking, isProcessing, sessionActive, audioEnabled]);

  // Stable ref for callbacks that change
  const onNavigateRef = useRef(onNavigate);
  useEffect(() => { onNavigateRef.current = onNavigate; }, [onNavigate]);

  // Reset session timeout — keeps session alive for 60s after last interaction
  const resetSessionTimeout = useCallback(() => {
    if (sessionTimeoutRef.current) clearTimeout(sessionTimeoutRef.current);
    sessionTimeoutRef.current = setTimeout(() => {
      if (!stateRef.current.isSpeaking && !stateRef.current.isProcessing) {
        setSessionActive(false);
        setTranscript("");
        wakeWordCooldownRef.current = false;
      }
    }, SESSION_TIMEOUT_MS);
  }, []);

  // Text-to-speech using OpenAI TTS (stable — uses refs)
  const speak = useCallback(async (text) => {
    if (!text) return;

    setIsSpeaking(true);
    setResponse(text);

    // Stop recognition while speaking to avoid hearing ourselves
    try { recognitionRef.current?.stop(); } catch (e) {}

    const onComplete = () => {
      setIsSpeaking(false);
      // Resume recognition after TTS finishes
      setTimeout(() => {
        if (stateRef.current.isListening && !stateRef.current.isSpeaking) {
          try { recognitionRef.current?.start(); } catch (e) {}
        }
      }, RESTART_DELAY_MS);
    };

    if (!stateRef.current.audioEnabled) {
      setTimeout(onComplete, Math.min(text.length * 50, 4000));
      return;
    }

    try {
      const res = await fetch(`${BACKEND_URL}/api/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ text, voice: "nova" })
      });

      if (res.ok) {
        const data = await res.json();
        const audio = new Audio(`data:audio/mp3;base64,${data.audio}`);
        audioRef.current = audio;
        audio.onended = onComplete;
        audio.onerror = () => fallbackSpeak(text, onComplete);
        await audio.play();
      } else {
        fallbackSpeak(text, onComplete);
      }
    } catch (error) {
      console.error("TTS error:", error);
      fallbackSpeak(text, onComplete);
    }
  }, []);

  // Fallback browser TTS
  const fallbackSpeak = (text, onComplete) => {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.85;
      utterance.onend = onComplete;
      utterance.onerror = onComplete;
      window.speechSynthesis.speak(utterance);
    } else {
      onComplete();
    }
  };

  // Stop speaking
  const stopSpeaking = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    setIsSpeaking(false);
  }, []);

  // Process voice command (stable — uses refs)
  const processCommand = useCallback(async (command) => {
    if (!command.trim()) return;

    setIsProcessing(true);
    resetSessionTimeout();

    try {
      const res = await fetch(`${BACKEND_URL}/api/voice-command`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ text: command, session_id: stateRef.current.sessionId })
      });

      if (res.ok) {
        const data = await res.json();
        const nav = onNavigateRef.current;

        if (data.action === "navigate" && nav) {
          speak(data.response);
          setTimeout(() => nav(data.target), 1500);
        } else if (data.action === "create_memory") {
          speak(data.response);
          setTimeout(() => nav?.("timeline"), 1500);
        } else if (data.action === "create_reminder") {
          speak(data.response);
          setTimeout(() => nav?.("reminders"), 1500);
        } else if (data.action === "create_family") {
          speak(data.response);
          setTimeout(() => nav?.("family"), 1500);
        } else {
          speak(data.response);
        }
      } else {
        speak("I'm sorry, I couldn't understand that. Could you try again?");
      }
    } catch (error) {
      console.error("Voice command error:", error);
      speak("I'm having trouble connecting. Please try again.");
    } finally {
      setIsProcessing(false);
      setTranscript("");
    }
  }, [speak, resetSessionTimeout]);

  // Initialize speech recognition ONCE (no state in deps — all accessed via refs)
  useEffect(() => {
    if (
      !("webkitSpeechRecognition" in window) &&
      !("SpeechRecognition" in window)
    ) {
      console.warn("Speech recognition not supported");
      return;
    }

    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();

    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
      const { isSpeaking, sessionActive, isProcessing } = stateRef.current;

      // Skip if speaking to avoid processing TTS output
      if (isSpeaking) return;

      let finalTranscript = "";
      let interimTranscript = "";

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += t;
        } else {
          interimTranscript += t;
        }
      }

      const heard = (finalTranscript || interimTranscript).toLowerCase();

      // PHASE 1: Check for wake word if session is NOT active
      if (!sessionActive && !wakeWordCooldownRef.current && heard.includes(WAKE_WORD)) {
        console.log("Wake word detected — activating session");

        // Brief cooldown to prevent double-trigger
        wakeWordCooldownRef.current = true;
        setTimeout(() => { wakeWordCooldownRef.current = false; }, WAKE_COOLDOWN_MS);

        setSessionActive(true);
        setIsExpanded(true);
        setTranscript("");

        // Stop recognition to clear buffer, speak will restart it
        try { recognition.stop(); } catch (e) {}

        speak("Yes, I'm listening. How can I help you?");
        return;
      }

      // PHASE 2: Session is active — process commands
      if (sessionActive && finalTranscript && !isProcessing) {
        const command = finalTranscript.replace(/hey memory/gi, "").trim();

        if (
          command.length > 2 &&
          command !== lastProcessedRef.current &&
          !processingLockRef.current
        ) {
          lastProcessedRef.current = command;
          processingLockRef.current = true;
          setTimeout(() => {
            processingLockRef.current = false;
            lastProcessedRef.current = "";
          }, 3000);

          setTranscript(command);
          processCommand(command);
          // NOTE: Session stays active — no reset! User can keep talking.
        }
      } else if (sessionActive && interimTranscript) {
        setTranscript(interimTranscript.replace(/hey memory/gi, "").trim());
      }
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error);
      if (event.error !== "no-speech") {
        setIsListening(false);
      }
    };

    recognition.onend = () => {
      // Auto-restart if we should still be listening and not speaking
      if (stateRef.current.isListening && !stateRef.current.isSpeaking) {
        setTimeout(() => {
          if (stateRef.current.isListening && !stateRef.current.isSpeaking) {
            try {
              recognition.start();
            } catch (e) {
              console.error("Failed to restart recognition:", e);
            }
          }
        }, RESTART_DELAY_MS);
      }
    };

    recognitionRef.current = recognition;

    return () => {
      try { recognition.stop(); } catch (e) {}
      if (sessionTimeoutRef.current) clearTimeout(sessionTimeoutRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Created ONCE — all mutable state accessed via stateRef

  // Start session timeout when session becomes active
  useEffect(() => {
    if (sessionActive) {
      resetSessionTimeout();
    }
    return () => {
      if (sessionTimeoutRef.current) clearTimeout(sessionTimeoutRef.current);
    };
  }, [sessionActive, resetSessionTimeout]);

  // Start/stop listening
  const toggleListening = useCallback(() => {
    if (stateRef.current.isListening) {
      try { recognitionRef.current?.stop(); } catch (e) {}
      setIsListening(false);
      setSessionActive(false);
      setTranscript("");
      wakeWordCooldownRef.current = false;
    } else {
      try {
        recognitionRef.current?.start();
        setIsListening(true);
        toast.success('Voice assistant activated. Say "Hey Memory" to start!');
      } catch (e) {
        console.error("Failed to start recognition:", e);
        toast.error("Could not start voice recognition");
      }
    }
  }, []);

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
                <span className="font-semibold text-foreground">
                  Voice Assistant
                </span>
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
              ) : sessionActive ? (
                <div className="flex items-center gap-2 text-success">
                  <Mic className="h-4 w-4 animate-pulse" />
                  <span>{transcript || "Listening... go ahead!"}</span>
                </div>
              ) : isListening ? (
                <div className="text-muted-foreground text-sm">
                  Say{" "}
                  <span className="font-bold text-primary">"Hey Memory"</span>{" "}
                  to talk to me
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
                {audioEnabled ? (
                  <Volume2 className="h-5 w-5" />
                ) : (
                  <VolumeX className="h-5 w-5" />
                )}
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
                {[
                  "Show my family",
                  "Play matching game",
                  "Play Sudoku",
                  "What did I do yesterday?",
                  "Open my medications",
                  "Open caregiver portal"
                ].map((cmd) => (
                  <button
                    key={cmd}
                    onClick={() => {
                      setSessionActive(true);
                      processCommand(cmd);
                    }}
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
            ${
              isListening
                ? "bg-primary animate-pulse"
                : "bg-primary hover:bg-primary/90"
            }
            ${sessionActive ? "ring-4 ring-success ring-opacity-50" : ""}
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
            {sessionActive ? (
              <span className="text-success font-medium">Listening...</span>
            ) : isListening ? (
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
