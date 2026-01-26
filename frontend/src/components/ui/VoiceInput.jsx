import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Mic, MicOff, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';

// Voice-to-text hook using Web Speech API
export const useVoiceToText = () => {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [isSupported, setIsSupported] = useState(false);
  const recognitionRef = useRef(null);

  useEffect(() => {
    // Check if browser supports speech recognition
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    setIsSupported(!!SpeechRecognition);

    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onresult = (event) => {
        let finalTranscript = '';
        let interimTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript + ' ';
          } else {
            interimTranscript += transcript;
          }
        }

        if (finalTranscript) {
          setTranscript(prev => prev + finalTranscript);
        }
      };

      recognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
        if (event.error === 'not-allowed') {
          toast.error('Please allow microphone access to use voice input');
        }
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  const startListening = useCallback(() => {
    if (recognitionRef.current && !isListening) {
      setTranscript('');
      recognitionRef.current.start();
      setIsListening(true);
    }
  }, [isListening]);

  const stopListening = useCallback(() => {
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  }, [isListening]);

  const toggleListening = useCallback(() => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  }, [isListening, startListening, stopListening]);

  const resetTranscript = useCallback(() => {
    setTranscript('');
  }, []);

  return {
    isListening,
    transcript,
    isSupported,
    startListening,
    stopListening,
    toggleListening,
    resetTranscript,
  };
};

// Voice Input Button Component
export const VoiceInputButton = ({ 
  onTranscript, 
  isListening, 
  onToggle, 
  isSupported,
  className = '' 
}) => {
  if (!isSupported) {
    return null;
  }

  return (
    <Button
      type="button"
      variant={isListening ? "destructive" : "outline"}
      size="icon"
      onClick={onToggle}
      className={`h-14 w-14 shrink-0 ${className}`}
      title={isListening ? 'Stop recording' : 'Start voice input'}
    >
      {isListening ? (
        <div className="relative">
          <MicOff className="h-6 w-6" />
          <span className="absolute -top-1 -right-1 flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-destructive-foreground opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-destructive-foreground"></span>
          </span>
        </div>
      ) : (
        <Mic className="h-6 w-6" />
      )}
    </Button>
  );
};

// Voice Input Field Component - combines textarea with voice button
export const VoiceInputField = ({
  value,
  onChange,
  placeholder,
  className = '',
  minHeight = '150px',
  label,
  id,
}) => {
  const {
    isListening,
    transcript,
    isSupported,
    toggleListening,
    resetTranscript,
  } = useVoiceToText();

  // Append transcript to value when it changes
  useEffect(() => {
    if (transcript) {
      onChange({ target: { value: value + transcript, name: id } });
      resetTranscript();
    }
  }, [transcript, value, onChange, resetTranscript, id]);

  return (
    <div className="space-y-2">
      {label && (
        <label htmlFor={id} className="text-lg font-semibold flex items-center gap-2">
          {label}
          {isSupported && (
            <span className="text-sm font-normal text-muted-foreground">
              (or tap mic to speak)
            </span>
          )}
        </label>
      )}
      <div className="relative">
        <textarea
          id={id}
          name={id}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          className={`w-full rounded-xl border-2 border-input bg-background px-4 py-3 text-lg focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20 resize-none ${className}`}
          style={{ minHeight }}
        />
        {isSupported && (
          <div className="absolute bottom-3 right-3">
            <VoiceInputButton
              isListening={isListening}
              onToggle={toggleListening}
              isSupported={isSupported}
              className={isListening ? 'animate-pulse' : ''}
            />
          </div>
        )}
      </div>
      {isListening && (
        <p className="text-sm text-primary animate-pulse flex items-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin" />
          Listening... speak now
        </p>
      )}
    </div>
  );
};

export default VoiceInputField;
