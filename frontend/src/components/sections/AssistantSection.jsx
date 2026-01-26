import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, Send, Mic, MicOff, User, Heart, Sparkles, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { chatApi } from '@/services/api';

// Predefined quick questions for easy access
const quickQuestions = [
  { id: 1, text: "Who is my wife?", icon: Heart },
  { id: 2, text: "What are my children's names?", icon: User },
  { id: 3, text: "When did I get married?", icon: Heart },
  { id: 4, text: "Where do I live?", icon: Heart },
  { id: 5, text: "Who are my grandchildren?", icon: User },
  { id: 6, text: "Tell me about my family", icon: Sparkles },
];

export const AssistantSection = ({ userName = 'Friend' }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'assistant',
      text: `Hello ${userName}! I'm here to help you remember. You can ask me anything about your family, your memories, or important events. Just type your question or tap one of the buttons below.`,
      timestamp: new Date()
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [sessionId] = useState(() => `session_${Date.now()}`);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (text = inputText) => {
    if (!text.trim()) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      text: text.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      // Call the real RAG-powered chat API
      const response = await chatApi.send(text.trim(), sessionId);
      
      const assistantMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        text: response.response,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      
      // Fallback message
      const errorMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        text: "I'm sorry, I'm having trouble connecting right now. Please try again in a moment.",
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuickQuestion = (question) => {
    handleSendMessage(question);
  };

  const toggleListening = () => {
    setIsListening(!isListening);
    // Voice recognition would be implemented here with Web Speech API
    // For now, it's a visual indicator
    if (!isListening) {
      // Start listening
      if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        
        recognition.onresult = (event) => {
          const transcript = event.results[0][0].transcript;
          setInputText(transcript);
          setIsListening(false);
        };
        
        recognition.onerror = () => {
          setIsListening(false);
        };
        
        recognition.onend = () => {
          setIsListening(false);
        };
        
        recognition.start();
      }
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <section className="py-8 sm:py-12">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-4xl">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-4 py-2 mb-4">
            <MessageCircle className="h-5 w-5 text-primary" />
            <span className="text-base font-medium text-primary">Your Memory Helper</span>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl font-bold text-foreground mb-4">
            Ask Me Anything
          </h2>
          <p className="text-accessible text-muted-foreground max-w-2xl mx-auto">
            I can help you remember your family, special moments, and important information
          </p>
        </div>

        {/* Quick Questions */}
        <div className="mb-8">
          <p className="text-lg font-semibold text-center mb-4">Quick Questions</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {quickQuestions.map((question) => {
              const Icon = question.icon;
              return (
                <Button
                  key={question.id}
                  variant="accessible-outline"
                  className="justify-start gap-3 h-auto py-4 px-5"
                  onClick={() => handleQuickQuestion(question.text)}
                  disabled={isLoading}
                >
                  <Icon className="h-5 w-5 text-primary shrink-0" />
                  <span className="text-left">{question.text}</span>
                </Button>
              );
            })}
          </div>
        </div>

        {/* Chat Area */}
        <Card className="border-2 border-border shadow-card overflow-hidden">
          <CardContent className="p-0">
            {/* Messages */}
            <div className="h-[400px] sm:h-[500px] overflow-y-auto p-6 space-y-6 bg-muted/30">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex gap-4 ${message.type === 'user' ? 'flex-row-reverse' : ''} animate-fade-in`}
                >
                  {/* Avatar */}
                  <div className={`shrink-0 w-12 h-12 rounded-xl flex items-center justify-center shadow-soft ${
                    message.type === 'user' 
                      ? 'bg-primary' 
                      : 'bg-accent'
                  }`}>
                    {message.type === 'user' ? (
                      <User className="h-6 w-6 text-primary-foreground" />
                    ) : (
                      <Heart className="h-6 w-6 text-accent-foreground" fill="currentColor" />
                    )}
                  </div>
                  
                  {/* Message bubble */}
                  <div className={`flex-1 max-w-[80%] ${message.type === 'user' ? 'text-right' : ''}`}>
                    <div className={`inline-block rounded-2xl p-5 shadow-soft ${
                      message.type === 'user'
                        ? 'bg-primary text-primary-foreground rounded-tr-sm'
                        : 'bg-card border border-border rounded-tl-sm'
                    }`}>
                      <p className="text-accessible whitespace-pre-line">{message.text}</p>
                    </div>
                    <p className="text-sm text-muted-foreground mt-2">
                      {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </p>
                  </div>
                </div>
              ))}
              
              {/* Loading indicator */}
              {isLoading && (
                <div className="flex gap-4 animate-fade-in">
                  <div className="shrink-0 w-12 h-12 rounded-xl bg-accent flex items-center justify-center shadow-soft">
                    <Heart className="h-6 w-6 text-accent-foreground" fill="currentColor" />
                  </div>
                  <div className="bg-card border border-border rounded-2xl rounded-tl-sm p-5 shadow-soft">
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-5 w-5 animate-spin text-primary" />
                      <span className="text-muted-foreground">Thinking...</span>
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-6 bg-card border-t border-border">
              <div className="flex gap-4">
                <div className="flex-1 relative">
                  <Textarea
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Type your question here..."
                    className="min-h-[80px] text-lg resize-none pr-14 rounded-xl border-2 focus:border-primary"
                    disabled={isLoading}
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    className={`absolute right-3 top-3 h-12 w-12 ${isListening ? 'bg-destructive text-destructive-foreground' : ''}`}
                    onClick={toggleListening}
                    aria-label={isListening ? 'Stop listening' : 'Start voice input'}
                    disabled={isLoading}
                  >
                    {isListening ? (
                      <MicOff className="h-6 w-6" />
                    ) : (
                      <Mic className="h-6 w-6" />
                    )}
                  </Button>
                </div>
                <Button
                  variant="accessible"
                  size="lg"
                  className="h-auto px-8"
                  onClick={() => handleSendMessage()}
                  disabled={!inputText.trim() || isLoading}
                >
                  <Send className="h-6 w-6" />
                  <span className="hidden sm:inline ml-2">Send</span>
                </Button>
              </div>
              
              <p className="text-sm text-muted-foreground text-center mt-4">
                Press Enter to send, or tap the microphone to speak
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </section>
  );
};

export default AssistantSection;
