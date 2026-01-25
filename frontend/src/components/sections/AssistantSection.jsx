import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, Send, Mic, MicOff, User, Heart, Sparkles, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

// Predefined quick questions for easy access
const quickQuestions = [
  { id: 1, text: "Who is my wife?", icon: Heart },
  { id: 2, text: "What are my children's names?", icon: User },
  { id: 3, text: "When did I get married?", icon: Heart },
  { id: 4, text: "Where do I live?", icon: Heart },
  { id: 5, text: "Who are my grandchildren?", icon: User },
  { id: 6, text: "Tell me about my family", icon: Sparkles },
];

// Knowledge base for the assistant (mock data that matches family/memories)
const knowledgeBase = {
  wife: {
    keywords: ['wife', 'spouse', 'maria', 'married to'],
    response: "Your wife's name is Maria. You've been married for over 50 years - you got married on June 15, 1972, at St. Mary's Church. She wore her mother's beautiful white wedding dress. Maria loves you very much and you have built a wonderful life together. Her birthday is on March 15."
  },
  children: {
    keywords: ['children', 'kids', 'son', 'daughter', 'michael', 'sarah'],
    response: "You have two wonderful children:\n\n• Michael - your son, born on July 22, 1975. He was your first child. You taught him to ride a bike when he was 6!\n\n• Sarah - your daughter, born on November 8, 1978. She grew up to be a doctor! You cried happy tears at her graduation from medical school."
  },
  grandchildren: {
    keywords: ['grandchildren', 'grandkids', 'emma', 'james', 'lily', 'grandson', 'granddaughter'],
    response: "You have three beautiful grandchildren:\n\n• Emma (12 years old) - Michael's daughter. She loves baking cookies with you every Christmas!\n\n• James (8 years old) - Sarah's son. He calls you 'Papa Bear' because you give the best hugs. He loves fishing with you.\n\n• Lily (5 years old) - Sarah's youngest. She loves when you read her bedtime stories, especially 'The Little Prince'."
  },
  wedding: {
    keywords: ['wedding', 'married', 'anniversary', 'marriage'],
    response: "You and Maria got married on June 15, 1972, at St. Mary's Church on Maple Avenue. It was a beautiful sunny summer day. Maria wore her mother's wedding dress and everyone danced until midnight. In 2022, you celebrated your 50th wedding anniversary with a big party where the whole family came together!"
  },
  home: {
    keywords: ['live', 'house', 'home', 'address'],
    response: "You live at 123 Oak Street in your hometown. You and Maria bought this house in March 1988 after years of saving. It has a big backyard with a beautiful oak tree. You've lived there for over 35 years now and it holds so many wonderful memories."
  },
  family: {
    keywords: ['family', 'everyone', 'loved ones'],
    response: "Your family is your greatest treasure!\n\n• Maria - your loving wife of 50+ years\n• Michael - your son (has a daughter named Emma)\n• Sarah - your daughter, a doctor (has James and Lily)\n• Emma, James, and Lily - your three grandchildren\n\nYou all celebrated your 50th anniversary together in 2022, dancing to the same song from your wedding day."
  },
  paris: {
    keywords: ['paris', 'france', 'eiffel', 'vacation', 'trip'],
    response: "In August 1985, you took your family on a dream vacation to Paris, France! You visited the Eiffel Tower, ate croissants every morning, and took a boat ride on the Seine river. Michael was 10 and Sarah was 7 at the time. They still talk about this wonderful trip today!"
  }
};

export const AssistantSection = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'assistant',
      text: "Hello! I'm here to help you remember. You can ask me anything about your family, your memories, or important events. Just type your question or tap one of the buttons below.",
      timestamp: new Date()
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const findAnswer = (question) => {
    const lowerQuestion = question.toLowerCase();
    
    for (const [key, data] of Object.entries(knowledgeBase)) {
      if (data.keywords.some(keyword => lowerQuestion.includes(keyword))) {
        return data.response;
      }
    }
    
    return "I'm here to help you remember. You can ask me about:\n\n• Your wife Maria\n• Your children Michael and Sarah\n• Your grandchildren Emma, James, and Lily\n• Your wedding and anniversary\n• Where you live\n• Family vacations\n\nJust ask and I'll tell you what I know!";
  };

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

    // Simulate a brief delay for natural feel
    await new Promise(resolve => setTimeout(resolve, 1000));

    const response = findAnswer(text);
    
    const assistantMessage = {
      id: Date.now() + 1,
      type: 'assistant',
      text: response,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, assistantMessage]);
    setIsLoading(false);
  };

  const handleQuickQuestion = (question) => {
    handleSendMessage(question);
  };

  const toggleListening = () => {
    setIsListening(!isListening);
    // Voice recognition would be implemented here
    // For now, it's a visual indicator
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
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    className={`absolute right-3 top-3 h-12 w-12 ${isListening ? 'bg-destructive text-destructive-foreground' : ''}`}
                    onClick={toggleListening}
                    aria-label={isListening ? 'Stop listening' : 'Start voice input'}
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
